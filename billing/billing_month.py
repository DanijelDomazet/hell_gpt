#!/usr/bin/env python3
"""
org_tokens_price_pretty_hr_month.py
Human-readable usage & cost for your OpenAI organization for a custom date span.
Default = current calendar month.
Examples:
  billing_month.py            # current month
  billing_month.py 30         # last 30 days incl. today
  billing_month.py --start 2024-04-01   # from 1 Apr 2024 until now
"""
import os, sys, datetime, argparse, requests
from zoneinfo import ZoneInfo           # Python ≥3.9

# ────────── config ──────────
ADMIN_KEY = os.getenv("OPENAI_ADMIN_KEY")
if not ADMIN_KEY:
    sys.exit("OPENAI_ADMIN_KEY not set!")

CRO_TZ = ZoneInfo("Europe/Zagreb")

USD_PER_M = {
    "input":        2.00,
    "cached_input": 0.50,
    "output":       8.00,
}
RATE = {k: v / 1_000_000 for k, v in USD_PER_M.items()}

URL     = "https://api.openai.com/v1/organization/usage/completions"
HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}",
           "Content-Type":  "application/json"}

# ────────── arg-parsing ──────────
parser = argparse.ArgumentParser(description="Org usage & cost summary for a custom span (default-current month)")
parser.add_argument("days", nargs="?", type=int, help="Number of past days to include (e.g. 30). Overrides --start.")
parser.add_argument("--start", "-s", type=str, help="ISO date YYYY-MM-DD to begin counting from (local midnight)")
args = parser.parse_args()

now_local = datetime.datetime.now(CRO_TZ)

# determine start_local
if args.days:
    start_local = (now_local - datetime.timedelta(days=args.days-1)).replace(hour=0, minute=0, second=0, microsecond=0)
elif args.start:
    try:
        y, m, d = map(int, args.start.split("-"))
        start_local = datetime.datetime(y, m, d, tzinfo=CRO_TZ)
    except Exception as e:
        sys.exit(f"Invalid --start date: {e}")
else:
    # first day of current month
    start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

start_ts = int(start_local.astimezone(datetime.timezone.utc).timestamp())
end_ts   = int(now_local.astimezone(datetime.timezone.utc).timestamp())

# ────────── helpers ──────────

def fetch_totals(start_ts, end_ts):
    params  = {"start_time": start_ts, "end_time": end_ts,
               "bucket_width": "1d", "limit": 31}
    totals  = {"input": 0, "cached_input": 0, "output": 0, "req": 0}
    while True:
        res = requests.get(URL, headers=HEADERS, params=params, timeout=30)
        res.raise_for_status()
        page = res.json()

        for bucket in page.get("data", []):
            for r in bucket.get("results", []):
                totals["input"]        += r.get("input_tokens", 0)
                totals["cached_input"] += r.get("input_cached_tokens", 0)
                totals["output"]       += r.get("output_tokens", 0)
                totals["req"]          += r.get("num_model_requests", 0)

        if not page.get("has_more"):
            break
        params["page"] = page["next_page"]
    return totals

fmt_int = lambda n: f"{n:,}"
fmt_usd = lambda n: f"${n:,.2f}"

def cost(counts):
    return (
        counts["input"]        * RATE["input"]        +
        counts["cached_input"] * RATE["cached_input"] +
        counts["output"]       * RATE["output"]
    )

# ────────── gather & print ──────────
span_counts = fetch_totals(start_ts, end_ts)
span_tokens = sum(span_counts.values())
span_cost   = cost(span_counts)
span_reqs   = span_counts["req"]

start_str = start_local.strftime("%Y-%m-%d")
end_str   = now_local.strftime("%Y-%m-%d")
print(f"\n📅  Usage {start_str} → {end_str} (Europe/Zagreb)")
line = "─" * 46
print(line)
print(f"{'Category':<18}{'Tokens':>14}{'Cost':>14}")
for key, label in (("input", "Input"),
                   ("cached_input", "Cached input"),
                   ("output", "Output")):
    print(f"{label:<18}{fmt_int(span_counts[key]):>14}{fmt_usd(span_counts[key]*RATE[key]):>14}")
print(line)
print(f"{'TOTAL':<18}{fmt_int(span_tokens):>14}{fmt_usd(span_cost):>14}")
print(f"{'Requests':<18}{fmt_int(span_reqs):>14}\n")
