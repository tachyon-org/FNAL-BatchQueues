#!/usr/bin/env python
# coding: utf-8

# In[300]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

# Replace with your filename if needed
file_path = Path("CodeAttemptFNAL_SeqVer.py").resolve()
print(file_path)


# ##### Note: Code split between two parts for me. First part is gathering data to put in a way to use, second part is transforming it. Also, headers are below code. ##### 

# ## Methods ##

# 
# (Garble Methods)
# 
# 
# 
# add — returns a stable token for an original and increments its count.
# 
# original_from_token — reverse-maps a token back to the original string.
# 
# record_from_token — fetches the UserRecord associated with a token.
# 
# export_to_json — saves the mapper’s current state to disk.
# 
# load_from_json — restores the mapper’s state from a saved JSON file.
# 
# 
# 
# 
# (Helper Methods)
# is_valid_user — checks that a user value is non-null and non-empty.
# 
# is_valid_ipv4 — validates IPv4 dotted-quad format and range.
# 
# to_jagged_array — builds [[original, token, count, valid], ...] for non-anonymous use.
# 
# dump_json — writes a Python object to a pretty-printed JSON file.
# 
# 
# 
# 
# (Data Methods)
# load_dataframe — selects required columns from all Parquet files in DATA_DIR.
# 
# build_obfuscations — iterates rows to create user/IP token maps with counts/validity.
# 
# make_summary_payload — returns anonymized users/IPs jagged arrays plus meta.
# 
# failed_users_payload — returns anonymized records for users with failed jobs plus meta.

# In[301]:


import os
import re
import json
import string
import secrets
from dataclasses import dataclass, asdict #asdict for json
from typing import Dict, Optional, List, Tuple #tuple for serializing
#For transforming data into what we want.
import duckdb
import pandas as pd
#For reading data and putting it into something usable.

from pathlib import Path
from textwrap import fill, indent
#For readable texts


# In[302]:


DATA_DIR = "../data"
OUTPUT_DIR = "./Output"
HUMAN_WRAP = 100 # wrap width for text output


# ### Import stuff ###

# In[303]:


USER_COL   = "User"
IP_COL     = "JobsubClientIpAddress"
FAILED_COL = "DAG_NodesFailed"  # “boolean-ish”
NUM_STARTS_COL     = "NumJobStarts"
NUM_COMPLETIONS_COL= "NumJobCompletions"


# ### Config / Column Names ###

# In[304]:


DIGITS = string.digits
LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DEFAULT_PUNCT = "!#$%&()*+,-.:;<=>?@[]^_{|}~"
CHAR_TYPE_CHOICES = ["digit", "lower", "upper", "punct"]
#defines UserRecord and GarbleTokenMapper for obfuscation.

@dataclass #shortcut class go brrr
class UserRecord:
    token: str
    count: int
    valid: bool


# In[305]:


class GarbleTokenMapper:
    """
    Sequential (non-random) token mapper.

    - Users:  tokens like "UR1", "UR2", ...
    - IPs:    tokens like "IP1", "IP2", ...

    Keeps the same public API as the previous GarbleTokenMapper:
    """

    def __init__(
        self,
        prefix: str = "",
        start: int = 1,
        # legacy args kept for drop-in compatibility; ignored
        token_len: int = 8,
        allow_punctuation: bool = False,
        punct_chars: Optional[str] = None,
    ):
        self.prefix = str(prefix or "")
        self.start = int(start)
        # original -> UserRecord(token, count, valid)
        self._by_orig: Dict[str, UserRecord] = {}
        # token   -> original
        self._token_to_orig: Dict[str, str] = {}
        # issued tokens (not strictly needed for sequential, kept for parity)
        self._seen_tokens = set()
        # counter points to the LAST issued number (so next is _counter + 1)
        self._counter = self.start - 1

    @staticmethod
    def _extract_trailing_int(s: str) -> Optional[int]:
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    def _next_token(self) -> str:
        self._counter += 1
        return f"{self.prefix}{self._counter}"

    def add(self, original: str, valid: bool = True) -> str:
        key = str(original)
        if key in self._by_orig:
            rec = self._by_orig[key]
            rec.count += 1
            return rec.token

        token = self._next_token()
        self._seen_tokens.add(token)
        rec = UserRecord(token=token, count=1, valid=bool(valid))
        self._by_orig[key] = rec
        self._token_to_orig[token] = key
        return token

    def original_from_token(self, token: str) -> Optional[str]:
        return self._token_to_orig.get(str(token))

    def record_from_token(self, token: str) -> Optional[UserRecord]:
        orig = self._token_to_orig.get(str(token))
        return self._by_orig.get(orig) if orig is not None else None

    def export_to_json(self, filepath: str) -> None:
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
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        cfg = data.get("config", {})

        # reset
        self._by_orig.clear()
        self._token_to_orig.clear()
        self._seen_tokens.clear()

        # keep existing prefix/start unless provided in file
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
            n = self._extract_trailing_int(token)
            if n is not None:
                max_num = max(max_num, n)

        # resume counting AFTER the largest seen number
        self._counter = int(cfg.get("counter", max_num))


# ### Token Mapper (Part 2) ###

# In[306]:


def export_to_json(self, filepath: str):
        entries = []
        for orig, rec in self._by_orig.items():
            e = asdict(rec)
            e["original"] = orig
            entries.append(e)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=2)


# In[307]:


def load_from_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])

        # clear current
        self._by_orig.clear()
        self._token_to_orig.clear()
        self._seen_tokens.clear()

        for e in entries:
            orig = e["original"]
            token = e["token"]
            count = int(e.get("count", 0))
            valid = bool(e.get("valid", True))
            rec = UserRecord(token=token, count=count, valid=valid)
            self._by_orig[orig] = rec
            self._token_to_orig[token] = orig
            self._seen_tokens.add(token)


# #### Json "export" ####

# In[308]:


_ipv4_re = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def is_valid_user(u) -> bool:
    if pd.isna(u):
        return False
    s = str(u).strip()
    return len(s) > 0


# In[309]:


def is_valid_ipv4(ip) -> bool:
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


# In[310]:


def to_jagged_array(ob_dict: Dict[str, Dict[str, object]]) -> List[List[object]]:
    """
    Anonymized jagged array builder:
      input: { original: {"id": token, "count": int, "valid": bool}, ... }
      output: [[token, count, valid], ...]   # NO original included
    """
    return [[data["id"], data["count"], data["valid"]]
            for _, data in ob_dict.items()] #throwaway with keys to get values in tuple.


# In[311]:


def dump_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# #### Helpers/Secondary ####

# In[312]:


def load_dataframe(data_dir: str) -> pd.DataFrame:
    """
    Reads all *.parquet in data_dir with DuckDB and returns a DataFrame.
    Only selects columns that actually exist across the files.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR does not exist or is not a directory: {data_dir}")

    pattern = f"{data_dir}/*.parquet"

    # Desired columns (with the corrected "Environment" name)
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

    # Peek schema (LIMIT 0) to learn available columns across all files
    # NOTE: read_parquet() properly handles glob patterns.
    schema_df = duckdb.sql(f"SELECT * FROM read_parquet('{pattern}') LIMIT 0").df()
    available = set(schema_df.columns)

    present = [c for c in desired_cols if c in available]
    missing = [c for c in desired_cols if c not in available]
    if missing:
        print(f"[load_dataframe] Warning: missing columns not found in any file: {missing}")

    if not present:
        raise RuntimeError("None of the desired columns are present in the parquet files.")

    # Quote identifiers to preserve any case-sensitive names
    q = ", ".join([f'"{c}"' for c in present])
    query = f"SELECT {q} FROM read_parquet('{pattern}')"
    return duckdb.sql(query).df()


# #### Loading Data ####

# In[313]:


def build_obfuscations(
    df: pd.DataFrame,
    user_col: str = USER_COL,
    ip_col: str = IP_COL,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], GarbleTokenMapper, GarbleTokenMapper]:
    """
    Build two dicts:
      users_dict = { original_user: {"id": token, "count": n, "valid": bool}, ... }
      ips_dict   = { original_ip:   {"id": token, "count": n, "valid": bool}, ... }
    Returns also the underlying mappers (useful if you want to export the maps).

    Tokens are sequential:
      - Users: UR1, UR2, ...
      - IPs:   IP1, IP2, ...
    """
    user_mapper = GarbleTokenMapper(prefix="UR", start=1)
    ip_mapper   = GarbleTokenMapper(prefix="IP", start=1)

    # iterate rows to add and count
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


# #### Transform ####

# In[314]:


def make_output_json(
    df: pd.DataFrame,
    users_dict: Dict[str, Dict[str, object]],
    ips_dict: Dict[str, Dict[str, object]],
) -> str:
    """
    Build the final JSON payload that includes jagged arrays and minimal metadata.
    """
    users_jagged = to_jagged_array(users_dict)
    ips_jagged   = to_jagged_array(ips_dict)

    payload = {
        "users": users_jagged,   # [[original, token, count, valid], ...]
        "ips":   ips_jagged,     # [[original, token, count, valid], ...]
        "meta": {
            "total_rows": int(len(df)),
            "distinct_users": int(len(users_dict)),
            "distinct_ips": int(len(ips_dict)),
        },
    }
    return json.dumps(payload, indent=2)


# #### Generic User json (Below) ####

# In[315]:


def make_summary_payload(
    df: pd.DataFrame,
    users_dict: Dict[str, Dict[str, object]],
    ips_dict: Dict[str, Dict[str, object]],
    user_col: str = USER_COL,
    ip_col: str = IP_COL,
) -> Dict[str, object]:
    """
    Build the summary payload and include per-user/IP correlation records:
      - users: [[token, count, valid], ...]               (no originals)
      - ips:   [[token, count, valid], ...]               (no originals)
      - user_ip_correlations: [
            [original_user, user_token, original_ip, ip_token, frequency, user_valid],
            ...
        ]
    """

    # Anonymized jagged arrays (no originals)
    users_jagged_anon = [[d["id"], d["count"], d["valid"]] for d in users_dict.values()]
    ips_jagged_anon   = [[d["id"], d["count"], d["valid"]] for d in ips_dict.values()]

    # Compute, for each user, the most frequent (mode) IP they used
    def _pick_mode_ip(series: pd.Series) -> Optional[str]:
        ser = series.dropna().astype(str)
        if ser.empty:
            return None
        return ser.value_counts().idxmax()

    tmp = df[[user_col, ip_col]].copy()
    tmp[user_col] = tmp[user_col].astype(str)
    top_ip_for_user = tmp.groupby(user_col)[ip_col].apply(_pick_mode_ip)

    # Build correlated records
    user_ip_correlations = []
    for orig_user, udata in users_dict.items():
        key_user = str(orig_user)
        ip_orig = top_ip_for_user.get(key_user, None)
        ip_token = ips_dict.get(ip_orig, {}).get("id") if ip_orig is not None else None
        user_ip_correlations.append([
            key_user,                 # original user
            udata["id"],              # garbled user
            ip_orig,                  # user's (mode) IP original
            ip_token,                 # garbled IP
            int(udata["count"]),      # frequency (user count)
            bool(udata["valid"]),     # user validity
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


# In[316]:


def _s(x):
    #in case of NaN/none
    return "" if pd.isna(x) else str(x)


# In[317]:


def _parse_env(env_raw):
    """
    Parses Environment column in sorted list by [key, value].
    Accepts:
      -Jsons, strings, key valu
    creates [env, string]
    """
    def _sort_key(pair):
        #Case sens.
        return pair[0].lower()

    s = _s(env_raw).strip()
    if not s:
        return []

    # Json
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                val = "" if v is None else str(v)
                pairs.append((str(k), val))
            return sorted(pairs, key=_sort_key)

        # Json arrays of KEY=VAL or dicts.
        if isinstance(obj, list):
            pairs = []
            for item in obj:
                if isinstance(item, dict):
                    for k, v in item.items():
                        val = "" if v is None else str(v)
                        pairs.append((str(k), val))
                elif isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    pairs.append((k.strip(), v.strip()))
                else:
                    pairs.append(("ITEM", str(item)))
            return sorted(pairs, key=_sort_key)
    except Exception:
        pass

    # Dict.
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("dict(") and s.endswith(")")):
        try:
            s_jsonish = s.replace("'", "\"")
            obj = json.loads(s_jsonish)
            if isinstance(obj, dict):
                pairs = []
                for k, v in obj.items():
                    val = "" if v is None else str(v)
                    pairs.append((str(k), val))
                return sorted(pairs, key=_sort_key)
        except Exception:
            pass

    # KEY=VAL pars.
    candidates = []
    for delim in [";", ",", "\n"]:
        if delim in s:
            candidates = [p for p in s.split(delim)]
            break
    if not candidates:
        # space-separated tokens; keep tokens that look like KEY=VAL
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
        return sorted(pairs, key=_sort_key)

    # Fallback
    return [("ENV", s)]


# In[318]:


def _wrap_block(text, width):
    return fill(_s(text), width=width, replace_whitespace=False)


# In[319]:


def _format_env_block(env_pairs, width, indent_spaces=2):
    if not env_pairs:
        return "  (none)"
    lines = []
    for k, v in env_pairs:
        # "KEY=VALUE" with wrapping of thevalue
        if v:
            wrapped_v = fill(v, width=width - (len(k) + 1 + indent_spaces),
                             subsequent_indent=" " * (len(k) + 1))
            lines.append(f"{k}={wrapped_v}")
        else:
            lines.append(f"{k}=")
    return indent("\n".join(lines), " " * indent_spaces)


# ### Helpers ###

# In[320]:


def write_cmd_env_report(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    group_by: str | None = None,   # e.g., "User" or None for flat list
    human_wrap: int = HUMAN_WRAP,
    include_meta: bool = True,
    meta_cols: tuple[str, ...] = ("User", "JobsubClientIpAddress",
                                  "CumulativeSlotTime", "DAG_NodesFailed",
                                  "NumJobStarts", "NumJobCompletions"),
    cmd_col: str = "Cmd",
    env_col: str = "Environment",
) -> Path:

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required cols exist (silently skip if missing)
    cols_needed = set([cmd_col, env_col]) | (set(meta_cols) if include_meta else set())
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"[write_cmd_env_report] Warning: missing columns {missing}; proceeding with what exists.")

    def format_one(idx, row) -> str:
        parts = []

        # Header line
        header = f"— Job #{idx} —"
        parts.append(header)

        # Meta block (compact, only present columns)
        if include_meta:
            for c in meta_cols:
                if c in row.index:
                    val = _s(row[c])
                    if c == cmd_col or c == env_col:
                        # don't duplicate
                        continue
                    # keep meta short; wrap very long fields
                    if len(val) > human_wrap:
                        val = _wrap_block(val, human_wrap)
                    parts.append(f"{c}: {val}")

        # Cmd
        if cmd_col in row.index:
            parts.append("Cmd:")
            parts.append(indent(_wrap_block(row[cmd_col], human_wrap), "  "))

        # Environment
        if env_col in row.index:
            parts.append("Environment:")
            env_pairs = _parse_env(row[env_col])
            parts.append(_format_env_block(env_pairs, width=human_wrap, indent_spaces=2))

        return "\n".join(parts)

    lines_out = []

    title = "Job Command & Environment Report"
    meta_summary = f"Total rows: {len(df)}"
    lines_out += [title, meta_summary, "=" * max(28, len(title)), ""]

    if group_by and group_by in df.columns:
        for gval, gdf in df.groupby(group_by, dropna=False):
            header = f"## {group_by}: {_s(gval)}  (jobs: {len(gdf)})"
            lines_out += [header, "-" * len(header)]
            for i, (_, row) in enumerate(gdf.iterrows(), start=1):
                lines_out.append(format_one(i, row))
                lines_out.append("")  # blank line between jobs
            lines_out.append("")      # blank line between groups
    else:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            lines_out.append(format_one(i, row))
            lines_out.append("")

    txt = "\n".join(lines_out).rstrip() + "\n"
    p.write_text(txt, encoding="utf-8")
    return p


# ### main writer ###

# ## cmd and envior. ##

# #### Failed User json (Below) ####

# In[321]:


def failed_users_payload(
    df: pd.DataFrame,
    user_mapper: GarbleTokenMapper,
    user_col: str = USER_COL,
    starts_col: str = NUM_STARTS_COL,
    completions_col: str = NUM_COMPLETIONS_COL,
) -> Dict[str, object]:
    """
    Create a payload listing JUST users with failed jobs, WITHOUT originals.
    Each record: { "token": <str>, "failure_count": <int>, "valid": <bool> }
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

    mask_fail = (df[completions_col].astype("int") == 0) & (df[starts_col] > 0)
    failed_df = df.loc[mask_fail, [user_col]]

    # count failure rows per user
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


# In[322]:


def site_jobs_payload(
    df: pd.DataFrame,
    site: str = "FermiGrid",
    *,
    site_col: str = "MATCH_EXP_JOB_Site",
    case_insensitive: bool = False,
) -> Dict[str, object]:
    """
    Payload of jobs given site
    """
    # Check required column
    if site_col not in df.columns:
        return {
            "jobs_at_site": [],
            "meta": {
                "site": site,
                "total_jobs_at_site": 0,
                "note": f"Missing column: {site_col}",
            },
        }

    # Build mask to select the site
    site_series = df[site_col].astype(str)
    if case_insensitive:
        mask = site_series.str.strip().str.casefold() == str(site).strip().casefold()
    else:
        mask = site_series.str.strip() == str(site).strip()

    # Filter dataframe
    df_site = df.loc[mask].copy()

    # Convert to list of dictionaries for json for readability
    job_records = df_site.to_dict(orient="records")

    payload = {
        "jobs_at_site": job_records,
        "meta": {
            "site": site,
            "total_jobs_at_site": int(len(df_site)),
            "columns_included": list(df_site.columns),
        },
    }
    return payload


# #### sites JSON ####

# #### Full output ####

# In[323]:


if __name__ == "__main__":
    # Load data
    df = load_dataframe(DATA_DIR)

    # Build obfuscations
    users_dict, ips_dict, user_mapper, ip_mapper = build_obfuscations(
        df, user_col=USER_COL, ip_col=IP_COL
    )

    # --- Create payloads / JSON strings ---
    summary_obj = make_summary_payload(df, users_dict, ips_dict)
    summary_json = json.dumps(summary_obj, indent=2)

    # Explicit JSON for the jagged arrays (as requested)
    users_jagged = to_jagged_array(users_dict)
    ips_jagged   = to_jagged_array(ips_dict)
    users_jagged_obj = {
        "users": users_jagged,
        "meta": {
            "distinct_users": int(len(users_dict)),
            "total_rows": int(len(df)),
        },
    }
    ips_jagged_obj = {
        "ips": ips_jagged,
        "meta": {
            "distinct_ips": int(len(ips_dict)),
            "total_rows": int(len(df)),
        },
    }

    # Failed users payload (JUST the users with failed jobs)
    failed_obj = failed_users_payload(
        df,
        user_mapper=user_mapper,
        user_col=USER_COL,
        starts_col=NUM_STARTS_COL,
        completions_col=NUM_COMPLETIONS_COL,
    )

    # --- Write files ---
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    users_jagged_path = os.path.join(OUTPUT_DIR, "users_jagged.json")
    ips_jagged_path = os.path.join(OUTPUT_DIR, "ips_jagged.json")
    failed_users_path = os.path.join(OUTPUT_DIR, "failed_users.json")
    out_path = os.path.join(OUTPUT_DIR, "jobs_at_FermiGrid.json")

    jobs_at_FermiGrid = site_jobs_payload(df, site="FermiGrid")
    dump_json(jobs_at_FermiGrid, out_path)

    dump_json(summary_obj, summary_path)
    dump_json(users_jagged_obj, users_jagged_path)
    dump_json(ips_jagged_obj, ips_jagged_path)
    dump_json(failed_obj, failed_users_path)
    
    # --- Console peeks ---
    #print("\n=== Wrote JSON files ===")
    #print("summary         :", summary_path)
    #print("users_jagged    :", users_jagged_path)
    #print("ips_jagged      :", ips_jagged_path)
    #print("failed_users    :", failed_users_path)

    print("\n=== Small samples ===")
    print("users_dict sample:", json.dumps(dict(list(users_dict.items())[:3]), indent=2))
    print("ips_dict sample  :", json.dumps(dict(list(ips_dict.items())[:3]), indent=2))
    print("\nsummary.json preview:\n", summary_json[:800], "...\n")

    # --- Failure stats (for context) ---
    total_starts = int(df[NUM_STARTS_COL].sum()) if NUM_STARTS_COL in df else 0
    total_completions = int(df[NUM_COMPLETIONS_COL].astype("int").sum()) if NUM_COMPLETIONS_COL in df else 0
    n_job_failures = total_starts - total_completions
    job_failure_frac = (n_job_failures / total_starts) if total_starts else 0.0
    print(f"Job failure fraction %: {job_failure_frac:.3%}, job failure abs number: {n_job_failures}") #Job failures = (Jobs Started - Jobs finished)/ Jobs Started

 #Text file stuff
    out_txt = Path(OUTPUT_DIR) / "cmd_env_report.txt"
    write_cmd_env_report(df, out_txt)
    print("Wrote:", out_txt)


# #### Main ####
