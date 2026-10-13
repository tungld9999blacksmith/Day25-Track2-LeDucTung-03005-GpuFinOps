"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


CARBON_PRICE_PER_T = 100.0  # internal carbon price (USD / tCO2e) — assumption
CLEANEST = "europe-north1"
BASELINE_REGION = "us-east-1"


def policy_comparison(jobs, cat, horizon_months: int = 36) -> dict:
    """Extension 1 — old recommend_tier vs interruption/term-aware recommend_tier_v2.

    The old M3 bills reserved only for USED hours (optimistic). To compare fairly we also
    re-price the old policy with the commitment billed 24h/day ("old, true cost").
    """
    od_total = old_reported = old_true = new_total = 0.0
    rows = []
    for j in jobs:
        c = cat[j["gpu_type"]]
        ngpu, hpd, days = int(num(j["num_gpus"])), num(j["hours_per_day"]), num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        prices = {k: num(c[k]) for k in ("on_demand_hr", "spot_hr", "reserved_1yr_hr", "reserved_3yr_hr")}

        old_tier = pricing.recommend_tier(hpd, interruptible)
        if old_tier == "spot":
            old_rep = old_tr = pricing.spot_checkpoint_cost(gpu_hours, prices["spot_hr"], od)["spot_cost"]
        elif old_tier == "reserved":
            old_rep = gpu_hours * prices["reserved_3yr_hr"]
            old_tr = 24 * DAYS * ngpu * prices["reserved_3yr_hr"]
        else:
            old_rep = old_tr = gpu_hours * od

        v2 = pricing.recommend_tier_v2(hpd, interruptible, j["gpu_type"], prices,
                                       job_days=days, horizon_months=horizon_months)
        new_cost = gpu_hours * v2["cost_per_used_hr"]
        od_total += gpu_hours * od
        old_reported += old_rep
        old_true += old_tr
        new_total += new_cost
        rows.append({"job_id": j["job_id"], "old": old_tier,
                     "new": v2["tier"] + (f"-{v2['term']}" if v2["term"] else ""),
                     "old_true": round(old_tr), "new_cost": round(new_cost)})
    pct = lambda x: (1 - x / od_total) * 100 if od_total else 0.0
    return {"rows": rows, "old_reported_pct": round(pct(old_reported), 1),
            "old_true_pct": round(pct(old_true), 1), "new_pct": round(pct(new_total), 1)}


def tier_matrix(cat, gpus=("H100", "A100", "A10G", "L4"), hours=(3, 8, 14, 18, 24)) -> list[dict]:
    """GPU type x duty cycle x interruptible -> v2 tier (steady 30-day service)."""
    out = []
    for g in gpus:
        prices = {k: num(cat[g][k]) for k in ("on_demand_hr", "spot_hr", "reserved_1yr_hr", "reserved_3yr_hr")}
        for intr in (False, True):
            cells = []
            for h in hours:
                v = pricing.recommend_tier_v2(h, intr, g, prices)
                cells.append(v["tier"] + (f"-{v['term']}" if v["term"] else ""))
            out.append({"gpu": g, "interruptible": intr, "cells": dict(zip(hours, cells))})
    return out


def carbon_schedule(jobs, cat) -> dict:
    """Extension 5 — move interruptible jobs to a cleaner / cheaper region."""
    kwh = 0.0
    for j in jobs:
        if int(num(j["interruptible"])):
            kwh += sustainability.job_energy_kwh(num(cat[j["gpu_type"]]["watts"]), int(num(j["num_gpus"])),
                                                 num(j["hours_per_day"]), num(j["days"]))
    table = sustainability.region_comparison(kwh, CARBON_PRICE_PER_T)
    base = next(r for r in table if r["region"] == BASELINE_REGION)
    cleanest = min(table, key=lambda r: r["co2_kg"])
    cheapest = min(table, key=lambda r: r["energy_usd"])
    balanced = min(table, key=lambda r: r["carbon_adjusted_usd"])
    return {"kwh": kwh, "table": table, "baseline": base, "cleanest": cleanest["region"],
            "cheapest": cheapest["region"], "balanced": balanced["region"],
            "saved_kg": base["co2_kg"] - cleanest["co2_kg"],
            "saved_pct": (1 - cleanest["co2_kg"] / base["co2_kg"]) * 100 if base["co2_kg"] else 0.0}


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    pol = policy_comparison(jobs, cat)
    pol12 = policy_comparison(jobs, cat, horizon_months=12)
    carbon = carbon_schedule(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- Extension 1: interruption/term-aware tier policy (recommend_tier_v2) --")
        print(f"{'job':18}{'old':11}{'new':14}{'old $ (24h commit)':>20}{'new $':>9}")
        for r in pol["rows"]:
            print(f"{r['job_id']:18}{r['old']:11}{r['new']:14}{r['old_true']:>20,}{r['new_cost']:>9,}")
        print(f"savings vs on-demand: old policy as reported {pol['old_reported_pct']}%  |  "
              f"old re-priced with true commitment cost {pol['old_true_pct']}%  |  new v2 {pol['new_pct']}%")
        print(f"if reservations are only safe for 12 months: new v2 {pol12['new_pct']}%")
        print("tier matrix (steady service; columns = hours/day):")
        for m in tier_matrix(cat):
            cells = "  ".join(f"{h}h:{v:<11}" for h, v in m["cells"].items())
            print(f"  {m['gpu']:5} {'interruptible' if m['interruptible'] else 'on-demand-ok':14} {cells}")

        print("\n-- Extension 5: carbon-aware scheduling of interruptible jobs --")
        print(f"interruptible energy: {carbon['kwh']:,.0f} kWh/run-set")
        print(f"{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'energy $':>10}{'CO2 kg':>9}{'carbon-adj $':>14}")
        for r in carbon["table"]:
            print(f"{r['region']:17}{r['usd_per_kwh']:>7.3f}{r['g_per_kwh']:>10}{r['energy_usd']:>10,.2f}"
                  f"{r['co2_kg']:>9,.1f}{r['carbon_adjusted_usd']:>14,.2f}")
        print(f"cheapest $: {carbon['cheapest']} | cleanest CO2: {carbon['cleanest']} | "
              f"balanced (carbon @ ${CARBON_PRICE_PER_T:.0f}/t): {carbon['balanced']}")
        print(f"{BASELINE_REGION} -> {carbon['cleanest']}: saves {carbon['saved_kg']:,.0f} kgCO2e ({carbon['saved_pct']:.0f}%)")
        print("latency trade-off: batch/training tolerates distance, so move it freely; keep latency-sensitive "
              "inference near users (europe-north1 is far from US users, +100ms RTT).")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "policy_v2": pol, "policy_v2_12m_pct": pol12["new_pct"], "carbon": carbon}


if __name__ == "__main__":
    run()
