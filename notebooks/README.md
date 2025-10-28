Data Anonymization / Reporting Tool

This tool ingests job/accounting data from one or more .parquet files and produces anonymized and shareable summaries, diagnostics, and reports.

1. Purpose

This tool ingests job/accounting data from one or more .parquet files and produces:

An anonymized view of sensitive identifiers:

User identifiers (e.g. User column, which may include email handles)

IP addresses (JobsubClientIpAddress)

Experiment / site hints in things like Environment and Cmd

Summary/diagnostic outputs suitable for sharing, debugging, or downstream analysis:

summary.json: high-level stats, “jagged arrays” of anonymized users and IPs, and correlations

failed_users.json: which users’ jobs are failing to complete

jobs_at_<site>.json: per-site listing of jobs, optionally garbled / scrubbed

cmd_env_report.txt: a readable, human-friendly text dump of command lines + runtime environments

In the future, this tool is also intended to support compression of those outputs so they’re smaller to store/ship.

2. Approach / Design Reasoning
First attempt: Hashing

The original idea was to hash user names, emails, and IPs.
That solves reversibility (mostly) but creates a few practical issues:

Long hash strings aren’t nice to read in a report.

Collisions aren’t likely, but you can’t easily tell “UR_12 appeared 18,000 times” without extra bookkeeping.

We sometimes want to retain a reversible mapping locally (for internal debugging).

Final approach: Sequential Token Dictionaries

Instead of hashing, we keep a small in-memory mapping that assigns each seen sensitive value a short, incrementing token:

Type	Example Tokens
Users	UR1, UR2, UR3, …
IPs	IP1, IP2, IP3, …
Experiments	EX_1, EX_2, …

This is driven by the GarbleTokenMapper class.

What this gives us

Stable, consistent tokens over the run (so you can correlate UR1 across files).

Compact output, easy to scan.

A count of how many times we saw each original value.

Optional persistence/export of the mapping to JSON (for rehydration or audit).

We then “garble” high-risk text fields like User, Cmd, and Environment by replacing user handles and experiment names with these sequential tokens.

3. High-Level Pseudocode

This is a simplified sketch of the current codepath, not exact code.

# 1. Parse CLI args
args = parse_args()

# 2. Load parquet data
df = load_dataframe(args.data_dir)

# 3. Build anonymization maps
users_dict, ips_dict, user_mapper, ip_mapper = build_obfuscations(
    df,
    user_col=args.user_col,
    ip_col=args.ip_col,
)

# 4. Generate summary-style payloads
summary_obj = make_summary_payload(df, users_dict, ips_dict, user_col=args.user_col, ip_col=args.ip_col)
failed_obj = failed_users_payload(df, user_mapper=user_mapper, user_col=args.user_col, starts_col=args.starts_col, completions_col=args.completions_col)

# 5. Generate per-site jobs payload (garbled)
site_payload = site_jobs_payload(
    df,
    site_name=args.site,
    site_col=args.site_col,
    case_insensitive=args.case_insensitive,
    garble=args.garble,
    user_col=args.user_col,
    cmd_col=args.cmd_col,
    env_col=args.env_col,
)

# 6. Write JSONs to output dir
dump_json(summary_obj, "<output>/summary.json")
dump_json(failed_obj, "<output>/failed_users.json")
dump_json(site_payload, "<output>/jobs_at_<site>.json")

# 7. Write human-readable TXT report of Cmd + Environment
write_cmd_env_report(df, "<output>/cmd_env_report.txt", group_by=args.report_group_by, wrap_width=args.wrap)

4. How to Use
Running It

You run this from a terminal, pointing it at:

a directory of parquet files (--data-dir)

an output directory (--output-dir)

Example:

python3 tool.py \
  --data-dir ../data \
  --output-dir ./Output \
  --site FermiGrid \
  --report-file cmd_env_report.txt

Required Arguments
Argument	Description
--data-dir	Path to a directory containing one or more .parquet files. The loader will scan them via DuckDB, select a known subset of columns, and concatenate.
--output-dir	Where all reports / JSON files will be written.
Optional / Tuning Arguments
Argument	Description
--wrap <int>	How wide to wrap long text blocks in the human-readable report. Defaults to 100.
--user-col, --ip-col, --site-col, --cmd-col, --env-col	Override column names if your parquet schema differs.
--starts-col, --completions-col	Columns used to decide if a job “failed”. Default: NumJobStarts, NumJobCompletions.
--site <str>	Which site you want to export into jobs_at_<site>.json.
--all-sites	Emit a JSON for each distinct site in the data.
--case-insensitive / --case-sensitive	Control how site matching works.
--garble / --no-garble	Whether to scrub/replace sensitive fields in the per-site dump.
--report-file <filename>	Filename for the text report.
--report-group-by <colname>	Group the text report by a column (e.g. --report-group-by User).
--include-meta / --no-meta	Whether to include per-job metadata lines in the text report.
--user-prefix, --exp-prefix, --experiments	Control the prefixes (UR_, EX_) and seed list of experiment keywords.
5. Expected Outputs

After a run, ./Output will contain several files.

cmd_env_report.txt

Human-readable block-per-job view. Each block shows who ran it, where, and with what environment.

— Job #1 —
User: uboonepro@fnal.gov
JobsubClientIpAddress: 131.225.240.146
CumulativeSlotTime: 727
...
Environment:
  BEARER_TOKEN_FILE=.condor_creds/uboone_production_6795daa7dc.use
  EXPERIMENT=uboone
  GRID_USER=uboonepro


Long values are line-wrapped for readability.

jobs_at_<site>.json

Per-site dump of rows, with sensitive strings replaced.

{
  "jobs_at_site": [
    {
      "User": "UR_29@fnal.gov",
      "RequestMemory": 2000,
      "JobsubClientIpAddress": "131.225.240.146",
      "Environment": "POMS_TASK_ID=1688108 ... GRID_USER=UR_29 ... EXPERIMENT=EX_8"
    }
  ],
  "meta": {
    "requested_site": "FermiGrid",
    "canonical_site": "FermiGrid",
    "token_prefixes": { "user": "UR_", "experiment": "EX_" }
  }
}


Notable details:

Only the handle part is garbled; domain is kept.

Environment swaps usernames and experiment names for placeholders.

Metadata under "meta" helps analysts trace anonymized mappings.

summary.json

Overview of distinct users, IPs, and correlations.

{
  "users": [["UR1", 186007, true], ["UR2", 42015, true]],
  "ips": [["IP1", 174541, true], ["IP2", 9982, true]],
  "user_ip_correlations": [["uboonepro@fnal.gov", "UR1", "131.225.240.146", "IP1", 186007, true]],
  "meta": { "total_rows": 206874, "distinct_users": 34, "distinct_ips": 33 }
}


Notes:

“Jagged arrays”: each entry = [token, count, valid].

"user_ip_correlations" link users and IPs with frequency stats.

Original values may remain for internal audits (not for publication).

failed_users.json

Who is starting jobs but never finishing them.

{
  "failed_users": [
    {"token": "UR7", "failure_count": 42, "valid": true},
    {"token": "UR19", "failure_count": 5, "valid": false}
  ],
  "meta": {"distinct_failed_users": 2, "total_failure_rows": 47}
}


Definition of “failed”:

NumJobStarts > 0
NumJobCompletions == 0

6. Reading / Navigating the Code
A. Constants / Imports / Dataclasses

Global constants like DATA_DIR, OUTPUT_DIR, default column names, wrap width.

UserRecord dataclass — struct for token, count, valid.

GarbleTokenMapper — hands out sequential tokens and tracks counts.

B. Validation / Helpers

is_valid_user, is_valid_ipv4 — quick sanity checks.

to_jagged_array, dump_json, load_dataframe — reshape and IO utilities.

_parse_env, _wrap_block, _format_env_block — environment parsing and report formatting helpers.

C. Core Builders / Payload Generators

build_obfuscations(df, ...) — assigns tokens and returns dicts + mappers.

make_summary_payload(...), failed_users_payload(...) — JSON-friendly payloads.

site_jobs_payload(...) — filters rows by site, garbles sensitive fields.

greedy_replace(...), garble_row_fields(...) — core garbling logic.

D. Human-Oriented Report Writer

write_cmd_env_report(df, out_path, ...)

Groups jobs

Writes readable blocks with wrapped Cmd and parsed Environment.

E. CLI Interface

build_arg_parser() — defines command-line arguments.

if __name__ == "__main__": — main orchestration:

Parse args

Load data

Generate anonymization + summaries

Write outputs

Print stats and preview

F. Internal Mechanics / Site Resolution

_canonicalize_site() — matches provided --site with actual dataframe entries (case-insensitive).
Used by site_jobs_payload() for site resolution.