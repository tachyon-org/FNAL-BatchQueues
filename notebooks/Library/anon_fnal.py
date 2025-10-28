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


# Defaults / constants used by helpers
HUMAN_WRAP = 100
USER_COL   = "User"
IP_COL     = "JobsubClientIpAddress"
FAILED_COL = "DAG_NodesFailed"
NUM_STARTS_COL      = "NumJobStarts"
NUM_COMPLETIONS_COL = "NumJobCompletions"

DIGITS = string.digits
LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DEFAULT_PUNCT = "!#$%&()*+,-.:;<=>?@[]^_{|}~"
CHAR_TYPE_CHOICES = ["digit", "lower", "upper", "punct"]

KNOWN_EXPERIMENTS = {"uboone", "icarus", "pip2", "nova", "dune"}


@dataclass
class UserRecord:
    token: str
    count: int
    valid: bool


class GarbleTokenMapper:
    def __init__(
        self,
        prefix: str = "",
        start: int = 1,
        token_len: int = 8,
        allow_punctuation: bool = False,
        punct_chars: Optional[str] = None,
    ):
        self.prefix = str(prefix or "")
        self.start = int(start)
        self._by_orig: Dict[str, UserRecord] = {}
        self._token_to_orig: Dict[str, str] = {}
        self._seen_tokens = set()
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

        self._by_orig.clear()
        self._token_to_orig.clear()
        self._seen_tokens.clear()

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

        self._counter = int(cfg.get("counter", max_num))


_ipv4_re = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def is_valid_user(u) -> bool:
    if pd.isna(u):
        return False
    s = str(u).strip()
    return len(s) > 0


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


def dump_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_dataframe(data_dir: str) -> pd.DataFrame:
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

    schema_df = duckdb.sql(f"SELECT * FROM read_parquet('{pattern}') LIMIT 0").df()
    available = set(schema_df.columns)

    present = [c for c in desired_cols if c in available]
    missing = [c for c in desired_cols if c not in available]
    if missing:
        print(f"[load_dataframe] Warning: missing columns not found in any file: {missing}")

    if not present:
        raise RuntimeError("None of the desired columns are present in the parquet files.")

    q = ", ".join([f'"{c}"' for c in present])
    query = f"SELECT {q} FROM read_parquet('{pattern}')"
    return duckdb.sql(query).df()


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
    users_jagged_anon = [[d["id"], d["count"], d["valid"]] for d in users_dict.values()]
    ips_jagged_anon   = [[d["id"], d["count"], d["valid"]] for d in ips_dict.values()]

    def _pick_mode_ip(series: pd.Series) -> Optional[str]:
        ser = series.dropna().astype(str)
        if ser.empty:
            return None
        return ser.value_counts().idxmax()

    tmp = df[[user_col, ip_col]].copy()
    tmp[user_col] = tmp[user_col].astype(str)
    top_ip_for_user = tmp.groupby(user_col)[ip_col].apply(_pick_mode_ip)

    user_ip_correlations = []
    for orig_user, udata in users_dict.items():
        key_user = str(orig_user)
        ip_orig = top_ip_for_user.get(key_user, None)
        ip_token = ips_dict.get(ip_orig, {}).get("id") if ip_orig is not None else None
        user_ip_correlations.append([
            key_user,
            udata["id"],
            ip_orig,
            ip_token,
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


def _s(x):
    return "" if pd.isna(x) else str(x)


def _parse_env(env_raw):
    def _sort_key(pair):
        return pair[0].lower()

    s = _s(env_raw).strip()
    if not s:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                val = "" if v is None else str(v)
                pairs.append((str(k), val))
            return sorted(pairs, key=_sort_key)

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

    candidates = []
    for delim in [";", ",", "\n"]:
        if delim in s:
            candidates = [p for p in s.split(delim)]
            break
    if not candidates:
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

    return [("ENV", s)]


def _wrap_block(text, width):
    return fill(_s(text), width=width, replace_whitespace=False)


def _format_env_block(env_pairs, width, indent_spaces=2):
    if not env_pairs:
        return "  (none)"
    lines = []
    for k, v in env_pairs:
        if v:
            wrapped_v = fill(
                v,
                width=width - (len(k) + 1 + indent_spaces),
                subsequent_indent=" " * (len(k) + 1),
            )
            lines.append(f"{k}={wrapped_v}")
        else:
            lines.append(f"{k}=")
    return indent("\n".join(lines), " " * indent_spaces)


def _extract_user_handle(user_val: str) -> str:
    s = _s(user_val)
    if "@" in s:
        return s.split("@", 1)[0].strip()
    return s.strip()


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
    user_mapper = GarbleTokenMapper(prefix=user_prefix, start=1)
    exp_mapper  = GarbleTokenMapper(prefix=exp_prefix, start=1)

    handles = set()
    experiments = set()

    if user_col in df.columns:
        for u in df[user_col].dropna().astype(str):
            h = _extract_user_handle(u)
            if h:
                handles.add(h)

    if env_col in df.columns:
        for raw in df[env_col].dropna().astype(str):
            pairs = _parse_env(raw)
            for k, v in pairs:
                if k.upper() == "EXPERIMENT" and v:
                    experiments.add(v.strip())
            low = raw.lower()
            for ex in KNOWN_EXPERIMENTS:
                if ex in low:
                    experiments.add(ex)

    experiments |= KNOWN_EXPERIMENTS

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
    if not literals:
        return re.compile(r"(?!x)x")
    escaped = [re.escape(s) for s in sorted(literals, key=len, reverse=True)]
    return re.compile("(" + "|".join(escaped) + ")")


def greedy_replace(text: str, mapping: dict[str, str], pattern: re.Pattern) -> str:
    if not text:
        return text

    def _sub(m):
        orig = m.group(0)
        return mapping.get(orig, orig)

    return pattern.sub(_sub, text)


def garble_user_email(email: str, user_handle_map: dict[str, str], pat: re.Pattern) -> str:
    s = _s(email)
    if "@" not in s:
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
    out = row.to_dict()

    if user_col in row.index:
        out[user_col] = garble_user_email(_s(row[user_col]), user_handle_map, pat_user)

    if cmd_col in row.index and pd.notna(row[cmd_col]):
        s = _s(row[cmd_col])
        s = greedy_replace(s, user_handle_map, pat_user_anywhere)
        s = greedy_replace(s, experiment_map, pat_exp)
        out[cmd_col] = s

    if env_col in row.index and pd.notna(row[env_col]):
        s = _s(row[env_col])
        s = greedy_replace(s, user_handle_map, pat_user_anywhere)
        s = greedy_replace(s, experiment_map, pat_exp)
        out[env_col] = s

    return out


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
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cols_needed = set([cmd_col, env_col]) | (set(meta_cols) if include_meta else set())
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"[write_cmd_env_report] Warning: missing columns {missing}; proceeding with what exists.")

    def format_one(idx, row) -> str:
        parts = []
        header = f"— Job #{idx} —"
        parts.append(header)

        if include_meta:
            for c in meta_cols:
                if c in row.index:
                    val = _s(row[c])
                    if c == cmd_col or c == env_col:
                        continue
                    if len(val) > human_wrap:
                        val = _wrap_block(val, human_wrap)
                    parts.append(f"{c}: {val}")

        if cmd_col in row.index:
            parts.append("Cmd:")
            parts.append(indent(_wrap_block(row[cmd_col], human_wrap), "  "))

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
                lines_out.append("")
            lines_out.append("")
    else:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            lines_out.append(format_one(i, row))
            lines_out.append("")

    txt = "\n".join(lines_out).rstrip() + "\n"
    p.write_text(txt, encoding="utf-8")
    return p


def _canonicalize_site(
    df: pd.DataFrame,
    site_col: str,
    requested: str,
    case_insensitive: bool,
):
    if site_col not in df.columns:
        return False, None, f"Missing column: {site_col}"

    series = df[site_col].dropna().astype(str).map(lambda s: s.strip())
    uniques = series.unique().tolist()
    if not uniques:
        return False, None, "No sites found in data."

    req = str(requested).strip()
    if not req:
        return False, None, "Empty site argument."

    if req in uniques:
        return True, req, "exact"

    if case_insensitive:
        matches = [u for u in uniques if u.casefold() == req.casefold()]
        if len(matches) == 1:
            return True, matches[0], "case-insensitive"
        elif len(matches) > 1:
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
    is_valid, canonical_site, match_note = _canonicalize_site(
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
        return {
            "jobs_at_site": [],
            "meta": {
                **meta_common,
                "total_jobs_at_site": 0,
                "columns_included": [],
                "note": "Requested site is not valid; returning empty result.",
            },
        }

    series = df[site_col].astype(str).map(lambda s: s.strip())
    mask = series == canonical_site
    df_site = df.loc[mask].copy()

    if not garble:
        return {
            "jobs_at_site": df_site.to_dict(orient="records"),
            "meta": {
                **meta_common,
                "total_jobs_at_site": int(len(df_site)),
                "columns_included": list(df_site.columns),
            },
        }

    user_mapper, exp_mapper, user_handle_map, experiment_map = build_sensitive_mappers_for_df(
        df_site, user_col=user_col, env_col=env_col, user_prefix="UR_", exp_prefix="EX_"
    )
    pat_user_local = compile_greedy_sub_regex(list(user_handle_map.keys()))
    pat_user_anywhere = pat_user_local
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
                "user_handles": user_handle_map,
                "experiments": experiment_map,
            },
            "token_prefixes": {"user": "UR_", "experiment": "EX_"},
        },
    }
