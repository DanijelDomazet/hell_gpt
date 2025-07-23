#!/usr/bin/env python3
"""
org_tokens_price_pretty_hr.py
Human-readable **daily** usage & cost for your OpenAI organization.
• Requires:  requests   (pip install requests)
• Env     :  OPENAI_ADMIN_KEY  (org-level key)

Pricing (o3 – 2025-04-16):
  input           → $2.00  / 1 000 000 tokens
  cached input    → $0.50  / 1 000 000 tokens
  output          → $8.00  / 1 000 000 tokens
"""

import os, sys, datetime, requests
from zoneinfo import ZoneInfo           # Python ≥3.9

# ────────── config ──────────
ADMIN_KEY = os.getenv("OPENAI_ADMIN_KEY")
if not ADMIN_KEY:
    sys.exit("OPENAI_ADMIN_KEY not set!")

CRO_TZ = ZoneInfo("Europe/Zagreb")      # 🇭🇷 local timezone

USD_PER_M = {                           # $ per 1M tokens
    "input":        2.00,
    "cached_input": 0.50,
    "output":       8.00,
}
RATE = {k: v / 1_000_000 for k, v in USD_PER_M.items()}

URL     = "https://api.openai.com/v1/organization/usage/completions"
HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}",
           "Content-Type":  "application/json"}

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

# ────────── time span (Croatia) ──────────
now_local   = datetime.datetime.now(CRO_TZ)
start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

start_ts = int(start_local.astimezone(datetime.timezone.utc).timestamp())
now_ts   = int(now_local .astimezone(datetime.timezone.utc).timestamp())

# ────────── gather data ──────────
today_counts = fetch_totals(start_ts, now_ts)
today_tokens = sum(today_counts.values())
today_cost   = cost(today_counts)
today_reqs   = today_counts["req"]

# ────────── pretty output ──────────
date_str   = start_local.strftime("%Y-%m-%d")
line       = "─" * 46

print(f"\n📅  Usage for {date_str} (Europe/Zagreb)\n{line}")
print(f"{'Category':<18}{'Tokens':>14}{'Cost':>14}")
for key, label in (("input", "Input"),
                   ("cached_input", "Cached input"),
                   ("output", "Output")):
    print(f"{label:<18}{fmt_int(today_counts[key]):>14}{fmt_usd(today_counts[key]*RATE[key]):>14}")
print(line)
print(f"{'TOTAL':<18}{fmt_int(today_tokens):>14}{fmt_usd(today_cost):>14}")
print(f"{'Requests':<18}{fmt_int(today_reqs):>14}\n")