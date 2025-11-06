# Job Analysis Toolkit — README

This README explains what the project does, how to run it, what it outputs, and how to interpret or extend its results. It uses a table-based layout for clarity.

---

##  General Purpose

| Aspect | Description |
|---|---|
| **Goal** | Analyze, summarize, and export Condor job data from `.parquet` logs into anonymized JSON and readable text reports. |
| **Core Idea** | Sequentially garble sensitive information (usernames, experiments) using deterministic token mapping (`UR_1`, `EX_1`) and produce both human- and machine-readable outputs. |
| **Primary Outputs** | (1) JSON summaries and job listings  (2) Text-based job reports  (3) Per-site job exports. |
| **Designed For** | Analysts or engineers handling job metrics across multiple grid sites who need privacy-safe exports. |

---

##  How It Works (Overview)

| Step | Description |
|---|---|
| **1. Load data** | Reads `.parquet` files from a given directory using DuckDB. Dynamically selects available columns. |
| **2. Build mappings** | Creates sequential user/IP tokens and maps for sensitive data. |
| **3. Garble selectively** | Replaces sensitive strings (user handles and experiment names) with tokens using greedy regex substitution. |
| **4. Generate payloads** | Produces multiple JSON summaries: users, IPs, failed users, jobs per site. |
| **5. Write reports** | Outputs JSON to disk and a text summary of job commands & environments. |

---

##  Requirements

| Type | Details |
|---|---|
| **Python** | 3.9+ |
| **Dependencies** | `duckdb`, `pandas` |
| **Input Data** | `.parquet` files in `--data-dir` |
| **Environment** | Works in CLI or Jupyter (ignores `-f` args automatically) |

---

##  Critical Functions

| Function | Purpose | Input | Output |
|---|---|---|---|
| `load_dataframe(data_dir)` | Load parquet files and select valid columns | Path to data dir | DataFrame |
| `build_obfuscations(df, user_col, ip_col)` | Create sequential user/IP tokens | DataFrame | (users_dict, ips_dict, user_mapper, ip_mapper) |
| `make_summary_payload(df, users_dict, ips_dict)` | Build summary with user↔IP correlation | DataFrame & dicts | Summary JSON object |
| `failed_users_payload(df, user_mapper, user_col, starts_col, completions_col)` | Detect users with failed jobs | DataFrame & cols | JSON object of failed users |
| `site_jobs_payload(df, site_name, ...)` | Export jobs at given site (validate + garble) | DataFrame + site name | JSON of jobs at that site |
| `write_cmd_env_report(df, out_path, ...)` | Generate readable job command report | DataFrame | `.txt` file |

---

##  Helper Functions

| Function | Description |
|---|---|
| `_canonicalize_site` | Checks if a requested site name exists in the data and normalizes casing. |
| `_parse_env` | Converts environment strings or JSONs into key-value pairs. |
| `_compile_greedy_sub_regex` | Builds regex patterns for efficient replacements. |
| `greedy_replace` | Performs greedy replacements using longest-first ordering. |
| `garble_row_fields` | Applies garbling to User, Cmd, and Environment fields. |
| `GarbleTokenMapper` | Generates consistent sequential tokens with optional persistence. |
| `_s`, `_wrap_block`, `_format_env_block` | Utility functions for safe string handling and text wrapping. |

---

##  Project Structure

| Path | Contents |
|---|---|
| `your_script.py` | Main logic (all functions + CLI) |
| `../data/` | Source `.parquet` job data files |
| `./Output/` | Generated JSON and TXT output files |
| `README.md` | This file |

---

##  Command-Line Usage

| Example | Description |
|---|---|
| `python your_script.py` | Runs default configuration (`../data`, `./Output`, site=`FermiGrid`). |
| `python your_script.py --site DUNE --no-garble` | Exports raw jobs from site DUNE (no anonymization). |
| `python your_script.py --all-sites` | Exports one JSON per detected site. |
| `python your_script.py --wrap 120 --report-group-by User` | Wider text wrap; group report per user. |
| `python your_script.py --data-dir /mnt/logs --output-dir ./out` | Custom data and output directories. |

###  Common Flags

| Flag | Purpose | Default |
|---|---|---|
| `--data-dir` | Input folder | `../data` |
| `--output-dir` | Output folder | `./Output` |
| `--site` | Target site to export | `FermiGrid` |
| `--all-sites` | Export every site | False |
| `--garble / --no-garble` | Toggle anonymization | Garble ON |
| `--case-insensitive / --case-sensitive` | Site matching mode | Insensitive |
| `--wrap` | Text wrap width | 100 |
| `--report-group-by` | Group TXT output (e.g., `User`) | None |
| `--experiments` | List of experiment names | `uboone,icarus,pip2,nova,dune` |

---

##  Output Examples

### `jobs_at_FermiGrid.json` (Valid Site, Garbled)
```json
{
  "jobs_at_site": [
    {
      "User": "UR_1@example.org",
      "Cmd": "/storage/.../EX_1_stage_1.sh",
      "Environment": "EXPERIMENT=EX_1; GRID_USER=UR_1"
    }
  ],
  "meta": {
    "canonical_site": "FermiGrid",
    "is_valid_site": true,
    "garbled": true,
    "maps": {
      "user_handles": {"alice": "UR_1"},
      "experiments": {"dune": "EX_1"}
    }
  }
}

_______________________________________________________________________________________________________________________________________________________________________________________

## cmd_env_report.txt

cmd_env_report.txt (Snippet)
Job Command & Environment Report
Total rows: 211
===============================

— Job #1 —
User: UR_1@example.org
JobsubClientIpAddress: 131.225.240.146
NumJobStarts: 1
NumJobCompletions: 1
Cmd:
  /storage/.../EX_1_stage_1.sh
Environment:
  EXPERIMENT=EX_1
  GRID_USER=UR_1
_______________________________________________________________________________________________________________________________________________________________________________________

## failed_users.json

{
  "failed_users": [
    {"token": "UR_3", "failure_count": 4, "valid": true}
  ],
  "meta": {
    "distinct_failed_users": 8,
    "total_failure_rows": 14
  }
}
_______________________________________________________________________________________________________________________________________________________________________________________
## jobs_at_InvalidSite.json

{
  "jobs_at_site": [],
  "meta": {
    "requested_site": "Nowhere",
    "is_valid_site": false,
    "note": "Requested site is not valid; returning empty result."
  }
}
________________________________________________________________________________________________________________________________________________________________________________________
## How to Read the Outputs

| Section | Meaning |
|---|---|
| `meta.is_valid_site` | Confirms if the requested site exists in the dataset. |
| `meta.canonical_site` | Shows how the site name appears in the original data. |
| `meta.garbled` | Indicates whether usernames and experiments were replaced with tokens. |
| `meta.maps` | Contains the mapping dictionaries `{original → token}` for reference. |
| `meta.columns_included` | Lists the DataFrame columns included in the payload. |
| `jobs_at_site` | Array of job records (one per row). |
| `failed_users` | List of users who had failed jobs. |

---

## Reading & Extending the Code

| Area | Key Concepts | How to Modify |
|---|---|---|
| **Entrypoint (`__main__`)** | Parses CLI arguments and triggers exports | Change defaults or add new flags here |
| **Data Loading** | Uses DuckDB to auto-detect available columns | Add or remove columns in `desired_cols` |
| **Validation** | `_canonicalize_site` ensures safe site lookups | Extend with fuzzy matching or site aliases |
| **Garbling** | Sequential deterministic mapping | Add new mappers (e.g., for IPs or IDs) |
| **Output Structure** | Handles JSON and TXT writing | Add new formats via `dump_json()` |
| **Readability** | `_parse_env`, `_format_env_block` | Modify for different environment formats or wrapping styles |

---

## File Summary

| Output File | Purpose |
|---|---|
| `summary.json` | Dataset-wide summary, correlations, and counts |
| `users_jagged.json` | Compact anonymized user list for analysis |
| `ips_jagged.json` | Compact anonymized IP list for analysis |
| `failed_users.json` | Users with failed jobs |
| `jobs_at_<SITE>.json` | Jobs filtered by site (garbled or raw) |
| `cmd_env_report.txt` | Readable job list with commands and environment variables |

---

## Typical Workflow

| Task | Command |
|---|---|
| **Generate all reports (default site)** | `python your_script.py` |
| **Export one site's jobs** | `python your_script.py --site DUNE` |
| **Export all sites** | `python your_script.py --all-sites` |
| **Get raw (no garble)** | `python your_script.py --no-garble` |
| **Change data/output directories** | `python your_script.py --data-dir ../alt_data --output-dir ./out` |
