"""
anon_fnal: helpers for loading FNAL batch queue data, anonymizing users/IPs,
           building summary payloads, and writing human-readable reports.
"""

import os
from pathlib import Path
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
import duckdb
import pandas as pd
from textwrap import fill, indent
import string

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Default wrap width for human-readable text reports
HUMAN_WRAP = 100

# Default column names in the dataframes we work with
USER_COL   = "User"
IP_COL     = "JobsubClientIpAddress"
FAILED_COL = "DAG_NodesFailed"
NUM_STARTS_COL      = "NumJobStarts"
NUM_COMPLETIONS_COL = "NumJobCompletions"

# Character sets (currently not heavily used, but kept for possible extensions)
DIGITS = string.digits
LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DEFAULT_PUNCT = "!#$%&()*+,-.:;<=>?@[]^_{|}~"
CHAR_TYPE_CHOICES = ["digit", "lower", "upper", "punct"]

# Experiments we treat as "known"; this is used when scanning environment
# strings to detect experiment tokens.
DEFAULT_KNOWN_EXPERIMENTS = {"uboone", "icarus", "pip2", "nova", "dune", "minerva"}

# ---------------------------------------------------------------------------
# Loading config for experiments
# ---------------------------------------------------------------------------

def load_known_experiments(config_path: str | Path | None = None) -> set[str]:
    """
    Load known experiments from config file.
    Searches in:
      1. provided config_path (if any)
      2. project's `tools/anon_fnal_config.json` (relative to this file)
      3. otherwise uses defaults
    """
    experiments = set(DEFAULT_KNOWN_EXPERIMENTS)

    # 1. If user explicitly passed a path, use it.
    if config_path:
        config_path = Path(config_path).expanduser()
    else:
        # 2. Default location in the project
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent  # up one level
        config_path = project_root / "tools" / "anon_fnal_config.json"

    if not config_path.is_file():
        return experiments

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        items = data.get("known_experiments", [])
        experiments |= {str(x).strip() for x in items if str(x).strip()}
    except Exception as e:
        print(f"[load_known_experiments] Warning: {e}; using defaults.")

    return experiments

KNOWN_EXPERIMENTS = load_known_experiments()

# ---------------------------------------------------------------------------
# Data structures for anonymization
# ---------------------------------------------------------------------------

@dataclass
class UserRecord:
    """
    Represents one original value (e.g. user or IP) that has been garbled.

    Attributes
    ----------
    token : str
        The anonymized token assigned to this original value.
    count : int
        How many times this original value appeared.
    valid : bool
        Whether the original value passed validity checks (e.g. non-empty, valid IP).
    """
    token: str
    count: int
    valid: bool


class GarbleTokenMapper:
    """
    Map original strings (usernames, IPs, etc.) to simple sequential tokens.

    Example
    -------
    >>> mapper = GarbleTokenMapper(prefix="UR")
    >>> mapper.add("alice")
    'UR1'
    >>> mapper.add("alice")
    'UR1'   # same token, count incremented
    >>> mapper.add("bob")
    'UR2'
    """

    def __init__(
        self,
        prefix: str = "",
        start: int = 1,
        token_len: int = 8,          # kept for legacy compatibility; not used now
        allow_punctuation: bool = False,  # kept for legacy; not used
        punct_chars: Optional[str] = None,  # kept for legacy; not used
    ):
        """
        Parameters
        ----------
        prefix : str
            String prefix for each token, e.g. "UR" or "IP".
        start : int
            First integer to use for numbering (e.g. 1 -> UR1, UR2, ...).
        token_len, allow_punctuation, punct_chars
            Currently unused, but kept so older code that passes them still runs.
        """
        self.prefix = str(prefix or "")
        self.start = int(start)
        # Map: original string -> UserRecord
        self._by_orig: Dict[str, UserRecord] = {}
        # Map: token -> original string (for reverse lookup)
        self._token_to_orig: Dict[str, str] = {}
        # Set of tokens we've already used (could be useful for collision checks)
        self._seen_tokens = set()
        # Internal counter; next token uses counter+1
        self._counter = self.start - 1

    @staticmethod
    def extract_trailing_int(s: str) -> Optional[int]:
        """
        Extract an integer at the end of a string, if present.

        Used when re-loading JSON to find the largest token number seen so far.
        """
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    def next_token(self) -> str:
        """Increment the internal counter and return the next token string."""
        self._counter += 1
        return f"{self.prefix}{self._counter}"

    def add(self, original: str, valid: bool = True) -> str:
        """
        Register an original string and return its token.

        If the original was seen before, reuses the same token and just bumps
        the count.

        Parameters
        ----------
        original : str
            The original string value (e.g. username, IP).
        valid : bool
            Whether this original value passed validity checks.

        Returns
        -------
        str
            The anonymized token for this original value.
        """
        key = str(original)
        if key in self._by_orig:
            rec = self._by_orig[key]
            rec.count += 1
            return rec.token

        token = self.next_token()
        self._seen_tokens.add(token)
        rec = UserRecord(token=token, count=1, valid=bool(valid))
        self._by_orig[key] = rec
        self._token_to_orig[token] = key
        return token

    def original_from_token(self, token: str) -> Optional[str]:
        """
        Look up the original value given a token.

        Returns None if the token is unknown.
        """
        return self._token_to_orig.get(str(token))

    def record_from_token(self, token: str) -> Optional[UserRecord]:
        """
        Look up the UserRecord (token, count, valid) given a token.

        Returns None if the token is unknown.
        """
        orig = self._token_to_orig.get(str(token))
        return self._by_orig.get(orig) if orig is not None else None

    def export_to_json(self, filepath: str) -> None:
        """
        Save the current mapping state to a JSON file.

        The JSON contains:
        - entries: list of {token, count, valid, original}
        - config: {prefix, start, counter}

        This lets you recreate the same mapping later.
        """
        entries = []
        for orig, rec in self._by_orig.items():
            e = asdict(rec)
            e["original"] = orig
            entries.append(e)
        state = {
            "entries": entries,
            "config": {"prefix": self.prefix, "start": self.start, "counter": self._counter},
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        """
        Load a mapping state previously written by export_to_json().

        This clears any existing mapping before loading.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        cfg = data.get("config", {})

        # Reset internal state
        self._by_orig.clear()
        self._token_to_orig.clear()
        self._seen_tokens.clear()

        # Keep existing prefix/start unless overridden by file config
        self.prefix = str(cfg.get("prefix", self.prefix))
        self.start = int(cfg.get("start", self.start))

        max_num = self.start - 1
        for e in entries:
            orig = str(e["original"])
            token = str(e["token"])
            count = int(e.get("count", 0))
            valid = bool(e.get("valid", True))
            rec = UserRecord(token=token, count=count, valid=valid)
            self._by_orig[orig] = rec
            self._token_to_orig[token] = orig
            self._seen_tokens.add(token)

            # Try to infer the numeric part at the end of the token
            n = self.extract_trailing_int(token)
            if n is not None:
                max_num = max(max_num, n)

        # If config has a saved counter, use that; otherwise use max_num we saw
        self._counter = int(cfg.get("counter", max_num))


# Pre-compiled IPv4 validation regex: "x.x.x.x" where each x is 1–3 digits
_ipv4_re = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


# ---------------------------------------------------------------------------
# Small helpers for shapes / payloads
# ---------------------------------------------------------------------------

def to_jagged_array(ob_dict: Dict[str, Dict[str, object]]) -> List[List[object]]:
    """
    Convert a dictionary like:
        { "alice": {"id": "UR1", "count": 5, "valid": True}, ... }
    into a "jagged array":
        [ ["UR1", 5, True], ... ]

    This is useful when you want a more compact representation for JSON.
    """
    # We don't care about the original key here, only the dict values.
    return [
        [data["id"], data["count"], data["valid"]]
        for _, data in ob_dict.items()
    ]


def make_output_json(
    df: pd.DataFrame,
    users_dict: Dict[str, Dict[str, object]],
    ips_dict: Dict[str, Dict[str, object]],
) -> str:
    """
    Build a JSON string that summarizes user/IP counts plus some metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataframe, used only for total row count.
    users_dict : dict
        Mapping original user -> {"id", "count", "valid"}.
    ips_dict : dict
        Mapping original IP -> {"id", "count", "valid"}.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    users_jagged = to_jagged_array(users_dict)
    ips_jagged   = to_jagged_array(ips_dict)

    payload = {
        "users": users_jagged,
        "ips":   ips_jagged,
        "meta": {
            "total_rows": int(len(df)),
            "distinct_users": int(len(users_dict)),
            "distinct_ips": int(len(ips_dict)),
        },
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_user(u) -> bool:
    """
    Return True if a user value is considered "valid":
    - not NaN
    - non-empty after stripping whitespace
    """
    if pd.isna(u):
        return False
    s = str(u).strip()
    return len(s) > 0


def is_valid_ipv4(ip) -> bool:
    """
    Return True if the value is a syntactically valid IPv4 address.

    We allow dotted-quad strings and then check that each octet is in [0, 255].
    """
    if pd.isna(ip):
        return False

    s = str(ip).strip()
    if not _ipv4_re.match(s):
        return False

    try:
        parts = [int(p) for p in s.split(".")]
    except ValueError:
        return False

    return all(0 <= p <= 255 for p in parts)


def dump_json(obj, path: str):
    """
    Write a Python object as pretty-printed JSON to disk, creating
    parent directories as needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Load all parquet files under data_dir into one dataframe, but only
    select a subset of columns we care about.

    Parameters
    ----------
    data_dir : str
        Directory containing .parquet files.

    Returns
    -------
    pd.DataFrame
        Combined dataframe of selected columns from all parquet files.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    RuntimeError
        If none of the desired columns exist in any of the files.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR does not exist or is not a directory: {data_dir}")

    pattern = f"{data_dir}/*.parquet"

    desired_cols = [
        "User",
        "RequestMemory",
        "CumulativeSlotTime",
        "JobsubClientIpAddress",
        "MATCH_EXP_JOB_Site",
        "DAG_NodesFailed",
        "NumJobCompletions",
        "NumJobStarts",
        "Cmd",
        "Environment",
    ]

    # Read only the schema (no rows) to see which columns exist at all.
    schema_df = duckdb.sql(f"SELECT * FROM read_parquet('{pattern}') LIMIT 0").df()
    available = set(schema_df.columns)

    present = [c for c in desired_cols if c in available]
    missing = [c for c in desired_cols if c not in available]
    if missing:
        print(f"[load_dataframe] Warning: missing columns not found in any file: {missing}")

    if not present:
        raise RuntimeError("None of the desired columns are present in the parquet files.")

    # Build a SELECT statement quoting column names (for safety)
    q = ", ".join([f'"{c}"' for c in present])
    query = f"SELECT {q} FROM read_parquet('{pattern}')"
    return duckdb.sql(query).df()


# ---------------------------------------------------------------------------
# Building anonymization dictionaries for users and IPs
# ---------------------------------------------------------------------------

def build_obfuscations(
    df: pd.DataFrame,
    user_col: str = USER_COL,
    ip_col: str = IP_COL,
) -> Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, object]],
    GarbleTokenMapper,
    GarbleTokenMapper,
]:
    """
    Build anonymization dictionaries and mappers for users and IPs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing user and IP columns.
    user_col : str
        Name of the user column in df.
    ip_col : str
        Name of the IP column in df.

    Returns
    -------
    users_dict : dict
        {original_user: {"id": token, "count": count, "valid": bool}}
    ips_dict : dict
        {original_ip: {"id": token, "count": count, "valid": bool}}
    user_mapper : GarbleTokenMapper
        Mapper used for users.
    ip_mapper : GarbleTokenMapper
        Mapper used for IPs.
    """
    user_mapper = GarbleTokenMapper(prefix="UR", start=1)
    ip_mapper   = GarbleTokenMapper(prefix="IP", start=1)

    for _, row in df.iterrows():
        u = row.get(user_col)
        ip = row.get(ip_col)

        user_mapper.add(str(u), valid=is_valid_user(u))
        ip_mapper.add(str(ip), valid=is_valid_ipv4(ip))

    users_dict = {
        orig: {"id": rec.token, "count": rec.count, "valid": rec.valid}
        for orig, rec in user_mapper._by_orig.items()
    }
    ips_dict = {
        orig: {"id": rec.token, "count": rec.count, "valid": rec.valid}
        for orig, rec in ip_mapper._by_orig.items()
    }
    return users_dict, ips_dict, user_mapper, ip_mapper


def make_summary_payload(
    df: pd.DataFrame,
    users_dict: Dict[str, Dict[str, object]],
    ips_dict: Dict[str, Dict[str, object]],
    user_col: str = USER_COL,
    ip_col: str = IP_COL,
) -> Dict[str, object]:
    """
    Build a summary payload combining users, IPs, and user->IP correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    users_dict : dict
        {orig_user: {"id", "count", "valid"}}
    ips_dict : dict
        {orig_ip: {"id", "count", "valid"}}
    user_col : str
        Column name for users.
    ip_col : str
        Column name for IPs.

    Returns
    -------
    dict
        Payload with:
        - "users": [[user_token, count, valid], ...]
        - "ips": [[ip_token, count, valid], ...]
        - "user_ip_correlations": per-user mode IP and mapping
        - "meta": summary stats
    """
    # Just the anonymized bits (drop original keys)
    users_jagged_anon = [[d["id"], d["count"], d["valid"]] for d in users_dict.values()]
    ips_jagged_anon   = [[d["id"], d["count"], d["valid"]] for d in ips_dict.values()]

    def _pick_mode_ip(series: pd.Series) -> Optional[str]:
        """
        Return the most frequent (mode) value in a series as a string,
        or None if the series is empty.
        """
        ser = series.dropna().astype(str)
        if ser.empty:
            return None
        return ser.value_counts().idxmax()

    # For each user, compute the most common IP they used.
    tmp = df[[user_col, ip_col]].copy()
    tmp[user_col] = tmp[user_col].astype(str)
    top_ip_for_user = tmp.groupby(user_col)[ip_col].apply(_pick_mode_ip)

    user_ip_correlations = []
    for orig_user, udata in users_dict.items():
        key_user = str(orig_user)
        ip_orig = top_ip_for_user.get(key_user, None)
        ip_token = ips_dict.get(ip_orig, {}).get("id") if ip_orig is not None else None
        user_ip_correlations.append([
            key_user,           # original user
            udata["id"],        # garbled user token
            ip_orig,            # mode IP for this user (original)
            ip_token,           # garbled IP token (if any)
            int(udata["count"]),
            bool(udata["valid"]),
        ])

    return {
        "users": users_jagged_anon,
        "ips": ips_jagged_anon,
        "user_ip_correlations": user_ip_correlations,
        "meta": {
            "total_rows": int(len(df)),
            "distinct_users": int(len(users_dict)),
            "distinct_ips": int(len(ips_dict)),
        },
    }


# ---------------------------------------------------------------------------
# Parsing Env / formatting for text reports
# ---------------------------------------------------------------------------

def safe_str(x) -> str:
    """Convert a value to string, treating NaN/None as empty string."""
    return "" if pd.isna(x) else str(x)


def parse_env(env_raw):
    """
    Parse various possible Environment representations into a list of (key, value).

    Handles:
    - JSON dict: {"KEY": "VAL", ...}
    - JSON list of dicts or "KEY=VAL" strings
    - dict-like strings using single quotes
    - delimited "KEY=VAL" strings (by ; , or newline)
    - fallback: returns [("ENV", original_string)]
    """
    def sort_key(pair):
        # Sort keys case-insensitively
        return pair[0].lower()

    s = safe_str(env_raw).strip()
    if not s:
        return []

    # Try strict JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                val = "" if v is None else str(v)
                pairs.append((str(k), val))
            return sorted(pairs, key=sort_key)

        if isinstance(obj, list):
            pairs = []
            for item in obj:
                if isinstance(item, dict):
                    # list of dicts
                    for k, v in item.items():
                        val = "" if v is None else str(v)
                        pairs.append((str(k), val))
                elif isinstance(item, str) and "=" in item:
                    # "KEY=VAL"
                    k, v = item.split("=", 1)
                    pairs.append((k.strip(), v.strip()))
                else:
                    # fallback for weird list entries
                    pairs.append(("ITEM", str(item)))
            return sorted(pairs, key=sort_key)
    except Exception:
        # Not valid JSON; fall through to other heuristics
        pass

    # Try something that *looks* like a dict but isn't strict JSON
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("dict(") and s.endswith(")")):
        try:
            # crude ' to " replacement to coax JSON parser
            s_jsonish = s.replace("'", "\"")
            obj = json.loads(s_jsonish)
            if isinstance(obj, dict):
                pairs = []
                for k, v in obj.items():
                    val = "" if v is None else str(v)
                    pairs.append((str(k), val))
                return sorted(pairs, key=sort_key)
        except Exception:
            pass

    # Try splitting by delimiters (semicolon, comma, newline)
    candidates = []
    for delim in [";", ",", "\n"]:
        if delim in s:
            candidates = [p for p in s.split(delim)]
            break
    if not candidates:
        # Last resort: split on whitespace
        candidates = s.split()

    pairs = []
    for token in candidates:
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    if pairs:
        return sorted(pairs, key=sort_key)

    # Could not parse into pairs; return single fallback entry
    return [("ENV", s)]


def wrap_block(text, width):
    """Wrap text to a given width, preserving whitespace where possible."""
    return fill(safe_str(text), width=width, replace_whitespace=False)


def format_env_block(env_pairs, width, indent_spaces=2):
    """
    Format environment key/value pairs as a pretty, wrapped block of text.

    For each key, we wrap the value so that "KEY=" + value fits into `width`.
    """
    if not env_pairs:
        return "  (none)"
    lines = []
    for k, v in env_pairs:
        if v:
            wrapped_v = fill(
                v,
                # Remaining width after "KEY=" and indentation
                width=width - (len(k) + 1 + indent_spaces),
                subsequent_indent=" " * (len(k) + 1),
            )
            lines.append(f"{k}={wrapped_v}")
        else:
            lines.append(f"{k}=")
    return indent("\n".join(lines), " " * indent_spaces)


def extract_user_handle(user_val: str) -> str:
    """
    Extract a "handle" from a user identifier.

    If it looks like an email (contains '@'), return the part before '@'.
    Otherwise, just return the stripped string.
    """
    s = safe_str(user_val)
    if "@" in s:
        return s.split("@", 1)[0].strip()
    return s.strip()

# ---------------------------------------------------------------------------
# Garbling user handles and experiments inside text fields
# ---------------------------------------------------------------------------

def build_sensitive_mappers_for_df(
    df: pd.DataFrame,
    *,
    user_col: str = USER_COL,
    env_col: str = "Environment",
    user_prefix: str = "UR_",
    exp_prefix: str = "EX_",
) -> tuple[
    GarbleTokenMapper,
    GarbleTokenMapper,
    dict[str, str],
    dict[str, str],
]:
    """
    Scan the dataframe for user handles and experiment tokens, and build
    mappers for them.

    Returns
    -------
    user_mapper : GarbleTokenMapper
    exp_mapper  : GarbleTokenMapper
    user_handle_map : dict
        {original_handle: token}
    experiment_map : dict
        {original_experiment: token}
    """
    user_mapper = GarbleTokenMapper(prefix=user_prefix, start=1)
    exp_mapper  = GarbleTokenMapper(prefix=exp_prefix, start=1)

    handles = set()
    experiments = set()

    # Collect user handles
    if user_col in df.columns:
        for u in df[user_col].dropna().astype(str):
            h = extract_user_handle(u)
            if h:
                handles.add(h)

    # Collect experiments seen in Environment plus KNOWN_EXPERIMENTS anywhere in raw string.
    if env_col in df.columns:
        for raw in df[env_col].dropna().astype(str):
            pairs = parse_env(raw)
            for k, v in pairs:
                if k.upper() == "EXPERIMENT" and v:
                    experiments.add(v.strip())
            # Also scan the raw string for known experiment tokens
            low = raw.lower()
            for ex in KNOWN_EXPERIMENTS:
                if ex in low:
                    experiments.add(ex)

    # Ensure all known experiments are included, even if not found in this df
    experiments |= KNOWN_EXPERIMENTS

    # Build deterministic mappings (sort to have stable token assignments)
    user_handle_map = {}
    for h in sorted(handles, key=lambda s: (len(s), s)):
        tok = user_mapper.add(h, valid=True)
        user_handle_map[h] = tok

    experiment_map = {}
    for ex in sorted(experiments, key=lambda s: (len(s), s)):
        tok = exp_mapper.add(ex, valid=True)
        experiment_map[ex] = tok

    return user_mapper, exp_mapper, user_handle_map, experiment_map


def compile_greedy_sub_regex(literals: list[str]) -> re.Pattern:
    """
    Compile a regex that matches any of the given literal strings, preferring
    longer ones first.

    The trick:
    - sort by length descending so that if you have ["alice", "ali"],
      "alice" is matched before "ali" (greedy behavior).
    - if the list is empty, return a regex that matches nothing: (?!x)x
      (that's a classic "always-false" regex).
    """
    if not literals:
        # Matches nothing: negative lookahead that always fails.
        return re.compile(r"(?!x)x")
    escaped = [re.escape(s) for s in sorted(literals, key=len, reverse=True)]
    return re.compile("(" + "|".join(escaped) + ")")


def greedy_replace(text: str, mapping: dict[str, str], pattern: re.Pattern) -> str:
    """
    Replace substrings in `text` according to `mapping` using a compiled regex
    built from the mapping keys.

    Parameters
    ----------
    text : str
        Input text where replacements will be made.
    mapping : dict
        {original_substring: replacement_substring}
    pattern : re.Pattern
        Compiled pattern that matches any of the keys in mapping.

    Returns
    -------
    str
        Text after performing substitutions.
    """
    if not text:
        return text

    def sub(m):
        orig = m.group(0)
        return mapping.get(orig, orig)

    return pattern.sub(sub, text)


def garble_user_email(email: str, user_handle_map: dict[str, str], pat: re.Pattern) -> str:
    """
    Garble the local-part of an email address using the user_handle_map and regex.

    If the string is not email-like (no '@'), treat the entire string as a
    handle container and replace handles wherever they appear.
    """
    s = safe_str(email)
    if "@" not in s:
        # Not an email; just treat as generic text with possible handles
        return greedy_replace(s, user_handle_map, pat)
    local, domain = s.split("@", 1)
    new_local = greedy_replace(local, user_handle_map, pat)
    return f"{new_local}@{domain}"


def garble_row_fields(
    row: pd.Series,
    *,
    user_col: str = USER_COL,
    cmd_col: str = "Cmd",
    env_col: str = "Environment",
    user_handle_map: dict[str, str],
    experiment_map: dict[str, str],
    pat_user: re.Pattern,
    pat_user_anywhere: re.Pattern,
    pat_exp: re.Pattern,
) -> dict:
    """
    Garble sensitive fields within a row (User, Cmd, Environment).

    Parameters
    ----------
    row : pd.Series
        One row from the dataframe.
    user_col, cmd_col, env_col : str
        Column names to be processed.
    user_handle_map : dict
        {original_handle: token}
    experiment_map : dict
        {original_experiment: token}
    pat_user : re.Pattern
        Pattern for replacing handles in email local-parts.
    pat_user_anywhere : re.Pattern
        Pattern for replacing handles anywhere in text fields.
    pat_exp : re.Pattern
        Pattern for replacing experiment strings.

    Returns
    -------
    dict
        A copy of row.to_dict() with certain fields garbled.
    """
    out = row.to_dict()

    if user_col in row.index:
        out[user_col] = garble_user_email(safe_str(row[user_col]), user_handle_map, pat_user)
        
    if cmd_col in row.index and pd.notna(row[cmd_col]):
        s = safe_str(row[cmd_col])
        s = greedy_replace(s, user_handle_map, pat_user_anywhere)
        s = greedy_replace(s, experiment_map, pat_exp)
        out[cmd_col] = s

    if env_col in row.index and pd.notna(row[env_col]):
        s = safe_str(row[cmd_col])
        s = greedy_replace(s, user_handle_map, pat_user_anywhere)
        s = greedy_replace(s, experiment_map, pat_exp)
        out[env_col] = s

    return out


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_cmd_env_report(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    group_by: str | None = None,
    human_wrap: int = HUMAN_WRAP,
    include_meta: bool = True,
    meta_cols: tuple[str, ...] = (
        "User",
        "JobsubClientIpAddress",
        "CumulativeSlotTime",
        "DAG_NodesFailed",
        "NumJobStarts",
        "NumJobCompletions",
    ),
    cmd_col: str = "Cmd",
    env_col: str = "Environment",
) -> Path:
    """
    Write a human-readable text report showing Cmd and Environment fields,
    optionally grouped by some column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least cmd_col and env_col.
    out_path : str or Path
        Where to write the text report.
    group_by : str or None
        Optional column name to group jobs by (e.g. "User").
    human_wrap : int
        Wrap width for long text fields.
    include_meta : bool
        Whether to show metadata columns for each job.
    meta_cols : tuple[str, ...]
        Metadata column names to include in the header of each job block.
    cmd_col, env_col : str
        Column names for the command and environment text.

    Returns
    -------
    Path
        The path where the report was written.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Make sure required columns exist; if not, we warn but continue with what we have.
    cols_needed = set([cmd_col, env_col]) | (set(meta_cols) if include_meta else set())
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"[write_cmd_env_report] Warning: missing columns {missing}; proceeding with what exists.")

    def format_one(idx, row) -> str:
        """
        Format a single job row into a text block.
        """
        parts = []

        # Header line for the job
        header = f"— Job #{idx} —"
        parts.append(header)

        # Meta block (User, IP, etc.)
        if include_meta:
            for c in meta_cols:
                if c in row.index:
                    val = safe_str(row[c])
                    if c == cmd_col or c == env_col:
                        # Don't duplicate these; they have their own sections below.
                        continue
                    # Wrap really long metadata values so they don't blow up formatting
                    if len(val) > human_wrap:
                        val = wrap_block(val, human_wrap)
                    parts.append(f"{c}: {val}")

        # Cmd section
        if cmd_col in row.index:
            parts.append("Cmd:")
            parts.append(indent(wrap_block(row[cmd_col], human_wrap), "  "))

        # Environment section
        if env_col in row.index:
            parts.append("Environment:")
            env_pairs = parse_env(row[env_col])
            parts.append(format_env_block(env_pairs, width=human_wrap, indent_spaces=2))

        return "\n".join(parts)

    # Build the full report as a list of lines
    lines_out = []
    title = "Job Command & Environment Report"
    meta_summary = f"Total rows: {len(df)}"
    lines_out += [title, meta_summary, "=" * max(28, len(title)), ""]

    if group_by and group_by in df.columns:
        # Group by e.g. User, so each group has a mini-header
        for gval, gdf in df.groupby(group_by, dropna=False):
            header = f"## {group_by}: {safe_str(gval)}  (jobs: {len(gdf)})"
            lines_out += [header, "-" * len(header)]
            for i, (_, row) in enumerate(gdf.iterrows(), start=1):
                lines_out.append(format_one(i, row))
                lines_out.append("")  # blank line between jobs
            lines_out.append("")      # blank line between groups
    else:
        # No grouping; just list every job
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            lines_out.append(format_one(i, row))
            lines_out.append("")

    txt = "\n".join(lines_out).rstrip() + "\n"
    p.write_text(txt, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Site selection helpers
# ---------------------------------------------------------------------------

def canonicalize_site(
    df: pd.DataFrame,
    site_col: str,
    requested: str,
    case_insensitive: bool,
):
    """
    Try to match a requested site name to one of the actual site values
    present in df[site_col].

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the site column.
    site_col : str
        Column name holding site names.
    requested : str
        The site requested by the user.
    case_insensitive : bool
        Whether to try case-insensitive matches if exact match fails.

    Returns
    -------
    (is_valid, canonical_site, note) : (bool, str or None, str)
        is_valid : True if we found a match.
        canonical_site : matched site value (or None).
        note : explanation ("exact", "case-insensitive", "not found", etc.).
    """
    if site_col not in df.columns:
        return False, None, f"Missing column: {site_col}"

    series = df[site_col].dropna().astype(str).map(lambda s: s.strip())
    uniques = series.unique().tolist()
    if not uniques:
        return False, None, "No sites found in data."

    req = str(requested).strip()
    if not req:
        return False, None, "Empty site argument."

    # Exact match first
    if req in uniques:
        return True, req, "exact"

    # Case-insensitive matching if allowed
    if case_insensitive:
        matches = [u for u in uniques if u.casefold() == req.casefold()]
        if len(matches) == 1:
            return True, matches[0], "case-insensitive"
        elif len(matches) > 1:
            # Ambiguous: multiple entries only differing by case
            matches_sorted = sorted(matches)
            return True, matches_sorted[0], f"ambiguous ({len(matches)} ci-matches)"

    return False, None, "not found"


def site_jobs_payload(
    df: pd.DataFrame,
    site_name: str,
    *,
    site_col: str = "MATCH_EXP_JOB_Site",
    case_insensitive: bool = True,
    garble: bool = True,
    user_col: str = USER_COL,
    cmd_col: str = "Cmd",
    env_col: str = "Environment",
) -> Dict[str, object]:
    """
    Build a JSON-ready payload of jobs for a specific site.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe.
    site_name : str
        Requested site name (as given by user/CLI).
    site_col : str
        Column in df containing site names.
    case_insensitive : bool
        Whether to allow case-insensitive site matching.
    garble : bool
        If True, anonymize user handles and experiments in text fields.
    user_col, cmd_col, env_col : str
        Column names used for garbling.

    Returns
    -------
    dict
        {
            "jobs_at_site": [ ... list of (possibly garbled) job dicts ... ],
            "meta": { ... metadata including validity, canonical_site, maps, etc. }
        }
    """
    # Try to match requested site to something that actually appears in df
    is_valid, canonical_site, match_note = canonicalize_site(
        df, site_col=site_col, requested=site_name, case_insensitive=case_insensitive
    )

    meta_common = {
        "requested_site": str(site_name),
        "canonical_site": canonical_site,
        "is_valid_site": bool(is_valid),
        "site_column": site_col,
        "match_note": match_note,
        "garbled": bool(garble),
    }

    if not is_valid:
        # Return an "empty" result with a note if the site is invalid
        return {
            "jobs_at_site": [],
            "meta": {
                **meta_common,
                "total_jobs_at_site": 0,
                "columns_included": [],
                "note": "Requested site is not valid; returning empty result.",
            },
        }

    # Filter rows that belong to the canonical site
    series = df[site_col].astype(str).map(lambda s: s.strip())
    mask = series == canonical_site
    df_site = df.loc[mask].copy()

    if not garble:
        # Return raw records, no anonymization
        return {
            "jobs_at_site": df_site.to_dict(orient="records"),
            "meta": {
                **meta_common,
                "total_jobs_at_site": int(len(df_site)),
                "columns_included": list(df_site.columns),
            },
        }

    # Build mappers based only on rows for this site
    user_mapper, exp_mapper, user_handle_map, experiment_map = build_sensitive_mappers_for_df(
        df_site, user_col=user_col, env_col=env_col, user_prefix="UR_", exp_prefix="EX_"
    )
    pat_user_local = compile_greedy_sub_regex(list(user_handle_map.keys()))
    pat_user_anywhere = pat_user_local  # for now we reuse the same pattern
    pat_exp = compile_greedy_sub_regex(list(experiment_map.keys()))

    garbled_rows = []
    for _, row in df_site.iterrows():
        garbled_rows.append(
            garble_row_fields(
                row,
                user_col=user_col,
                cmd_col=cmd_col,
                env_col=env_col,
                user_handle_map=user_handle_map,
                experiment_map=experiment_map,
                pat_user=pat_user_local,
                pat_user_anywhere=pat_user_anywhere,
                pat_exp=pat_exp,
            )
        )

    return {
        "jobs_at_site": garbled_rows,
        "meta": {
            **meta_common,
            "total_jobs_at_site": int(len(df_site)),
            "columns_included": list(df_site.columns),
            "maps": {
                "user_handles": user_handle_map,   # {original_handle: "UR_n"}
                "experiments": experiment_map,     # {original_exp: "EX_n"}
            },
            "token_prefixes": {"user": "UR_", "experiment": "EX_"},
        },
    }


# ---------------------------------------------------------------------------
# Failed users summary
# ---------------------------------------------------------------------------

def failed_users_payload(
    df: pd.DataFrame,
    user_mapper: GarbleTokenMapper,
    user_col: str = USER_COL,
    starts_col: str = NUM_STARTS_COL,
    completions_col: str = NUM_COMPLETIONS_COL,
) -> Dict[str, object]:
    """
    Build a payload summarizing "failed" users based on start/completion counts.

    A row is considered a "failure row" if:
        NumJobStarts > 0 and NumJobCompletions == 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing user, starts, and completions columns.
    user_mapper : GarbleTokenMapper
        Mapper to use for garbling user identifiers.
    user_col, starts_col, completions_col : str
        Column names to use.

    Returns
    -------
    dict
        {
          "failed_users": [
              {"token", "failure_count", "valid"}, ...
          ],
          "meta": {
              "distinct_failed_users": int,
              "total_failure_rows": int,
              ...
          }
        }
    """
    if not {user_col, starts_col, completions_col} <= set(df.columns):
        return {
            "failed_users": [],
            "meta": {
                "distinct_failed_users": 0,
                "total_failure_rows": 0,
                "note": "Required columns missing; cannot compute failed users.",
            },
        }

    # "Failure row": some starts but zero completions
    mask_fail = (df[completions_col].astype("int") == 0) & (df[starts_col] > 0)
    failed_df = df.loc[mask_fail, [user_col]]

    # Count failure rows per user
    fail_counts = failed_df.groupby(user_col)[user_col].count().rename("failure_count")

    records = []
    total_failure_rows = int(fail_counts.sum()) if not fail_counts.empty else 0

    for orig_user, fcount in fail_counts.items():
        token = user_mapper.add(str(orig_user), valid=is_valid_user(orig_user))
        records.append({
            "token": token,
            "failure_count": int(fcount),
            "valid": is_valid_user(orig_user),
        })

    payload = {
        "failed_users": records,
        "meta": {
            "distinct_failed_users": int(len(records)),
            "total_failure_rows": total_failure_rows,
        },
    }
    return payload


# ---------------------------------------------------------------------------
# Selected job fields dump
# ---------------------------------------------------------------------------

def dump_selected_job_fields(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    garble: bool = True,
    user_col: str = "User",
) -> Path:
    """
    Dump a subset of job fields to JSON, optionally garbling the user column.

    This is meant to be a more compact "record dump" for further analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    out_path : str or Path
        Where to write the JSON file.
    garble : bool
        If True, anonymize the user column by mapping users to tokens.
    user_col : str
        Column name for the user field to be garbled.

    Returns
    -------
    Path
        The path where the JSON was written.
    """
    # Fields we care about; we only keep the ones actually present in df.
    wanted_cols = [
        "User",
        "AccountingGroup",
        "JobCurrentStartDate",
        "JobTotalTime",
        "JobStatus",
        "ExitCode",
        "HoldReason",
        "RequestMemory",
        "RequestCpus",
        "RequestDisk",
        "RequestSlots",
        "JOB_EXPECTED_MAX_LIFETIME",
        "MemoryUsage",
        "DiskUsage",
        "CumulativeRemoteUserCpu",
        "CumulativeRemoteSysCpu",
        "TotalSubmitProcs",
        "ProcId",
        "ClusterId",
    ]
    present_cols = [c for c in wanted_cols if c in df.columns]

    # Local user mapper; this is separate from the global one used elsewhere.
    # It's only used for this specific selected-fields dump.
    user_mapper = GarbleTokenMapper(prefix="UR_", start=1)

    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in present_cols:
            val = row[col]
            if pd.isna(val):
                val = None

            if garble and col == user_col and val is not None:
                # Garble only the user column
                val = user_mapper.add(str(val), valid=is_valid_user(val))

            rec[col] = val
        records.append(rec)

    # Build payload including both the records and the mapping used
    payload = {
        "records": records,
        "meta": {
            "total_rows": len(records),
            "columns_included": present_cols,
        },
        "maps": {
            "users": [
                {
                    "original": orig,
                    "token": rec.token,
                    "count": rec.count,
                    "valid": rec.valid,
                }
                for orig, rec in user_mapper._by_orig.items()
            ]
        },
    }

    out_path = Path(out_path)
    dump_json(payload, str(out_path))
    return out_path

