"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Extensions in this file:
  * Ext 3 — cache_is_worth_it(): cache savings only counted when reads/prefix pay back the write.
  * Ext 4 — reasoning budget: $ and Wh of is_reasoning traffic + a routing cap.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Ext 3 assumptions. Anthropic-style cache write = 1.25x input price -> 0.25 extra.
CACHE_WRITE_COST = 0.25
# What-if: explicit-storage caches (Gemini-style). Illustrative $/1M-token/hour, 1h TTL.
STORAGE_PER_M_HR = {"small": 1.00, "large": 4.50}
CACHE_TTL_HR = 1.0

REASONING_CAPS = (0.10, 0.05, 0.02)  # fraction of traffic allowed to use reasoning


def _implied_reads(cached_tokens: float, input_tokens: float) -> float:
    """Proxy for avg reads per cached prefix from the cache-hit ratio h: n = h / (1 - h).

    (A prefix written once and read n times has hit ratio n / (n + 1).)
    """
    h = cached_tokens / input_tokens if input_tokens else 0.0
    return h / (1.0 - h) if h < 1.0 else float("inf")


def cache_economics(rows) -> dict:
    """Ext 3 — per-tier break-even reads vs the reads actually observed in the data."""
    tot = defaultdict(lambda: {"in": 0.0, "cached": 0.0})
    for r in rows:
        t = tot[r["route_tier"]]
        t["in"] += num(r["input_tokens"])
        t["cached"] += num(r["cached_input_tokens"])
    out = {}
    for tier, t in tot.items():
        price_in = MODEL_PRICES[tier][0]
        reads = _implied_reads(t["cached"], t["in"])
        be = pricing.cache_breakeven_reads(CACHE_WRITE_COST)
        # storage-billed what-if: express storage $ as a multiple of the input price
        storage_write = CACHE_WRITE_COST + STORAGE_PER_M_HR[tier] * CACHE_TTL_HR / price_in
        be_storage = pricing.cache_breakeven_reads(storage_write)
        out[tier] = {
            "hit_ratio": t["cached"] / t["in"] if t["in"] else 0.0,
            "avg_reads": reads,
            "breakeven_reads": be,
            "worth_it": pricing.cache_is_worth_it(reads, CACHE_WRITE_COST),
            "breakeven_reads_storage": be_storage,
            "worth_it_storage": pricing.cache_is_worth_it(reads, storage_write),
        }
    return out


def reasoning_budget(rows, opt_cost_fn, caps=REASONING_CAPS) -> dict:
    """Ext 4 — $ / Wh split of reasoning traffic, and the effect of capping it.

    Routing rule modelled: reasoning is only allowed for the `cap` share of traffic with the
    longest prompts (complexity proxy); the rest is served as a normal completion whose output
    length falls back to the tier's mean non-reasoning output (no hidden thinking tokens).
    """
    mean_out = {}
    for tier in MODEL_PRICES:
        outs = [num(r["output_tokens"]) for r in rows if r["route_tier"] == tier and not int(num(r["is_reasoning"]))]
        mean_out[tier] = sum(outs) / len(outs) if outs else 0.0

    def totals(reasoning_ids):
        d = {"n": 0, "cost": 0.0, "wh": 0.0, "rn": 0, "rcost": 0.0, "rwh": 0.0}
        for i, r in enumerate(rows):
            was_reasoning = bool(int(num(r["is_reasoning"])))
            reasoning = was_reasoning if reasoning_ids is None else i in reasoning_ids
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            if was_reasoning and not reasoning:
                out = min(out, int(mean_out[r["route_tier"]]))
            cost = opt_cost_fn(r, inp, out)
            wh = sustainability.wh_per_query(inp + out, is_reasoning=reasoning)
            d["n"] += 1; d["cost"] += cost; d["wh"] += wh
            if reasoning:
                d["rn"] += 1; d["rcost"] += cost; d["rwh"] += wh
        return d

    cur = totals(None)
    n_all = len(rows)
    reasoning_idx = [i for i, r in enumerate(rows) if int(num(r["is_reasoning"]))]
    reasoning_idx.sort(key=lambda i: num(rows[i]["input_tokens"]), reverse=True)
    avg_wh_normal = (cur["wh"] - cur["rwh"]) / max(1, cur["n"] - cur["rn"])
    avg_wh_reason = cur["rwh"] / max(1, cur["rn"])
    scenarios = []
    for cap in caps:
        keep = set(reasoning_idx[: int(cap * n_all)])
        s = totals(keep)
        scenarios.append({"cap": cap, "reasoning_share": s["rn"] / n_all,
                          "cost": s["cost"], "wh": s["wh"],
                          "cost_saved": cur["cost"] - s["cost"], "wh_saved": cur["wh"] - s["wh"],
                          "wh_saved_pct": (1 - s["wh"] / cur["wh"]) * 100 if cur["wh"] else 0.0})
    return {
        "traffic_share": cur["rn"] / n_all, "cost_share": cur["rcost"] / cur["cost"] if cur["cost"] else 0.0,
        "wh_share": cur["rwh"] / cur["wh"] if cur["wh"] else 0.0,
        "cost_reasoning": cur["rcost"], "cost_normal": cur["cost"] - cur["rcost"],
        "wh_reasoning": cur["rwh"], "wh_normal": cur["wh"] - cur["rwh"],
        "wh_per_query_ratio": avg_wh_reason / avg_wh_normal if avg_wh_normal else 0.0,
        "scenarios": scenarios,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_info = cache_economics(rows)
    cache_ok = {t: v["worth_it"] for t, v in cache_info.items()}

    base_cost = opt_cost = opt_no_cache = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (only if it pays), batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        used_cache = cached if cache_ok.get(r["route_tier"], False) else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=used_cache, batch=is_batch)
        opt_no_cache += pricing.request_cost(inp, out, pin, pout, cached_in=0, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    def opt_fn(r, inp, out):
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cached = min(int(num(r["cached_input_tokens"])), inp) if cache_ok.get(r["route_tier"], False) else 0
        return pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=bool(int(num(r["is_batch"]))))

    reasoning = reasoning_budget(rows, opt_fn)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: is prompt caching worth it? (cache_is_worth_it) --")
        print(f"cache lever alone saves ${opt_no_cache - opt_cost:,.2f}/day "
              f"({(opt_no_cache - opt_cost) / opt_no_cache * 100:.1f}% of the post-cascade/batch bill)")
        print(f"{'tier':7}{'hit%':>7}{'avg reads':>11}{'break-even':>12}{'worth it':>10}"
              f"{'| storage-billed BE':>20}{'worth it':>10}")
        for tier, v in cache_info.items():
            print(f"{tier:7}{v['hit_ratio']*100:>6.1f}%{v['avg_reads']:>11.2f}{v['breakeven_reads']:>12.2f}"
                  f"{str(v['worth_it']):>10}{v['breakeven_reads_storage']:>20.2f}{str(v['worth_it_storage']):>10}")
        print("(avg reads = proxy h/(1-h) from the cache-hit ratio; storage-billed = what-if, illustrative $/M-tok/hr)")

        print("\n-- Extension 4: reasoning budget --")
        print(f"reasoning = {reasoning['traffic_share']*100:.1f}% of traffic, "
              f"{reasoning['cost_share']*100:.1f}% of $ (${reasoning['cost_reasoning']:,.2f}/day), "
              f"{reasoning['wh_share']*100:.1f}% of energy ({reasoning['wh_reasoning']/1000:,.1f} of "
              f"{(reasoning['wh_reasoning']+reasoning['wh_normal'])/1000:,.1f} kWh/day)")
        print(f"energy per reasoning query is {reasoning['wh_per_query_ratio']:.0f}x a normal one "
              f"(80x multiplier x ~2.2x tokens)")
        for s in reasoning["scenarios"]:
            note = "  (already below cap: no change)" if s["cost_saved"] <= 1e-9 and s["wh_saved"] <= 1e-9 else ""
            print(f"cap reasoning at {s['cap']*100:>4.0f}% traffic -> share {s['reasoning_share']*100:.1f}%  "
                  f"saves ${s['cost_saved']:.2f}/day (${s['cost_saved']*30:,.0f}/mo) and "
                  f"{s['wh_saved']/1000:,.2f} kWh/day ({s['wh_saved_pct']:.0f}%){note}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_info": cache_info, "cache_savings_daily": round(opt_no_cache - opt_cost, 2),
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
