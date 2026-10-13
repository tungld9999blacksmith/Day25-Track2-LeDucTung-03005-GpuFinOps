"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def analysis_sections(r1, r2, r3, levers, baseline, cat) -> str:
    """Narrative + extension results appended to the generated report (rubric C.1/C.2)."""
    L = []
    total = sum(levers.values())
    first_cache = next(iter(r2["cache_info"].values()))
    # --- unit economics ---
    L += ["", "## Unit economics ($/1M-token)", "",
          f"- Baseline (all-large model, no cache, no batch): **${r2['baseline_per_m']:.3f}/1M-token** (${r2['baseline_daily']:,.2f}/day)",
          f"- Optimized (cascade + caching + batch): **${r2['optimized_per_m']:.3f}/1M-token** (${r2['optimized_daily']:,.2f}/day)",
          f"- Inference savings: **{r2['savings_pct']}%** over {r2['total_tokens']:,} tokens", ""]
    # --- lever table ---
    L += ["## Lever breakdown and priority", "",
          "| Rank | Lever | Savings / month | % of baseline | % of total savings |", "|---|---|---|---|---|"]
    for i, (name, amt) in enumerate(sorted(levers.items(), key=lambda kv: -kv[1]), 1):
        L.append(f"| {i} | {name} | ${amt:,.0f} | {amt / baseline * 100:.1f}% | {amt / total * 100:.0f}% |")
    L += ["", "Ranked by dollars. By effort: *Kill idle GPUs* is a scheduler change (hours), *Purchasing* needs a commitment "
          "decision (days; spot is reversible, reserved is not), *Right-size* needs a migration test, *Inference* needs "
          "routing/caching plumbing. Start with idle + spot (cheap, immediate), then reserved for the steady inference fleet.", ""]
    # --- GPU-Util lie ---
    L += ["## Why GPU-Util is a lie", "",
          "`nvidia-smi` GPU-Util is the share of sample periods in which *at least one kernel was running*. It says nothing "
          "about how many SMs were busy or whether the tensor cores were fed. A GPU stalled on HBM reads, waiting on "
          "all-reduce or data loading, or running tiny kernels with launch overhead still reports ~100%. "
          "MFU (achieved FLOPs / peak FLOPs) is what you are actually paying for.", "",
          "| GPU | Type | GPU-Util | MFU | Wasted $/hr | Wasted $/month |", "|---|---|---|---|---|---|"]
    waste_total = 0.0
    for lie in r1["lies"]:
        hr = num(cat[lie["gpu_type"]]["on_demand_hr"])
        w = (1 - lie["mfu"]) * hr
        waste_total += w * 24 * DAYS
        L.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['gpu_util_pct']}% | {lie['mfu']:.2f} | ${w:.2f} | ${w * 24 * DAYS:,.0f} |")
    L += ["", f"These GPUs look healthy on a dashboard yet leak **${waste_total:,.0f}/month** in unused FLOPs. "
          "Fix the bottleneck (batch size, dataloader, comm overlap, fused kernels) before buying more GPUs; "
          "if it cannot be fixed, right-size down (lever above).", ""]
    # --- extensions ---
    v2 = r3["policy_v2"]
    cache = r2["cache_info"]
    rs = r2["reasoning"]
    rs_moves = "; ".join(f"{x['gpu_id']} {x['current']} -> {x['proposed']} (${x['monthly_savings']:,.0f}/mo)"
                         for x in r1["rightsize"] if x["monthly_savings"] > 0)
    L += ["## Extension results", "",
          "**Ext 1: interruption/term-aware tier policy** (`pricing.recommend_tier_v2`). "
          f"The old policy reported {v2['old_reported_pct']}% savings but bills reserved only for used hours; re-priced with the true 24h commitment it is "
          f"{v2['old_true_pct']}%. The new policy (per-GPU interruption rate, 1yr vs 3yr eligibility) gives {v2['new_pct']}% with realistic accounting, "
          f"and {r3['policy_v2_12m_pct']}% if only 1-year commitments are safe. Most jobs keep the same tier; the value is an honest number and a decision matrix.", "",
          f"**Ext 2: right-sizing by MBU** (M1). {rs_moves or 'No move found'}. Total ${r1['rightsize_monthly']:,.0f}/month. "
          "Cheapest $/hr is not enough: the SKU must hold the working set and supply the bandwidth at <=60% MBU (L4 is cheapest but has 0.3 TB/s).", "",
          "**Ext 3: `cache_is_worth_it`**. Observed cache-hit ratio "
          + ", ".join(f"{t} {v['hit_ratio']*100:.0f}%" for t, v in cache.items())
          + f" gives ~{first_cache['avg_reads']:.2f} reads/prefix vs a break-even of {first_cache['breakeven_reads']:.2f} "
          f"with 1.25x writes: worth it (cache saves ${r2['cache_savings_daily']:.2f}/day). With storage-billed caching the break-even rises to "
          + ", ".join(f"{v['breakeven_reads_storage']:.1f} ({t})" for t, v in cache.items())
          + " reads, and caching would lose money at this hit rate.", "",
          f"**Ext 4: reasoning budget**. Reasoning is {rs['traffic_share']*100:.1f}% of traffic but {rs['cost_share']*100:.1f}% of dollars and "
          f"{rs['wh_share']*100:.0f}% of energy (about {rs['wh_per_query_ratio']:.0f}x Wh per query: 80x multiplier times ~2.2x tokens). "
          "Rule: allow reasoning only for the longest/hardest prompts (<=5% of traffic) or when the cheap answer fails validation. "
          + "; ".join(f"cap {s['cap']*100:.0f}% saves ${s['cost_saved']*30:,.0f}/mo and {s['wh_saved_pct']:.0f}% of energy" for s in rs["scenarios"][1:]) + ".", ""]
    # --- sustainability ---
    c = r3["carbon"]
    L += ["## Sustainability: region and carbon", "",
          "| Region | $/kWh | gCO2/kWh | Energy $ | CO2 kg | Carbon-adjusted $ (@ $100/t) |", "|---|---|---|---|---|---|"]
    for r in c["table"]:
        L.append(f"| {r['region']} | {r['usd_per_kwh']:.3f} | {r['g_per_kwh']} | ${r['energy_usd']:,.2f} | {r['co2_kg']:,.1f} | ${r['carbon_adjusted_usd']:,.2f} |")
    L += ["", f"Interruptible jobs use {c['kwh']:,.0f} kWh. Cheapest electricity: **{c['cheapest']}**; cleanest: **{c['cleanest']}**; "
          f"best balance with an internal carbon price: **{c['balanced']}**. Moving them from us-east-1 to {c['cleanest']} avoids "
          f"{c['saved_kg']:,.0f} kgCO2e ({c['saved_pct']:.0f}%). Strictly, the cheapest grid is {c['cheapest']}, so the 'cheapest+cleanest' line above means cleanest. "
          "Trade-off: batch/training tolerates distance; latency-sensitive inference should stay near users. "
          "GPU rental prices already include power, so the electricity column is an indicator, while carbon is the real saving.", ""]
    # --- actions ---
    L += ["## Recommended actions (priority order)", "",
          f"1. **Move steady inference to 3yr reserved and training to spot** (${levers['Purchasing (spot/reserved)']:,.0f}/mo, largest lever).",
          "2. **Auto-stop idle GPUs** (zero risk, immediate).",
          f"3. **Deploy cascade + caching + batch routing** ({r2['savings_pct']}% off the inference bill, also lowers carbon).",
          "4. **Replace GPU-Util with MFU/MBU alerts**; fix or right-size the util-lie GPUs.",
          "5. **Cap reasoning to <=5% of traffic** and schedule interruptible jobs in the cleanest/cheapest region.",
          "6. **Tag everything, then chargeback** (the 80% coverage gate is already open)."]
    return "\n".join(L)


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    md = report.build_report(baseline, optimized, levers, sustainability=sust)
    md += "\n" + analysis_sections(r1, r2, r3, levers, baseline, cat) + "\n"
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
