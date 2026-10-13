"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


# Right-sizing targets (Ext 2): leave headroom so the new GPU isn't pinned at 100%.
MBU_TARGET = 0.60
MFU_TARGET = 0.60
VRAM_HEADROOM = 1.15


def vram_price_table(cat: dict) -> list[dict]:
    """$/GB-VRAM and bandwidth economics for every GPU in the catalog."""
    rows = []
    for g, c in cat.items():
        od, hbm, bw = num(c["on_demand_hr"]), num(c["hbm_gb"]), num(c["peak_bw_tbs"])
        rows.append({"gpu_type": g, "on_demand_hr": od, "hbm_gb": hbm, "peak_bw_tbs": bw,
                     "usd_per_gb_vram_hr": od / hbm if hbm else 0.0,
                     "usd_per_tbs_hr": od / bw if bw else 0.0})
    return sorted(rows, key=lambda r: r["usd_per_gb_vram_hr"])


def rightsize_by_mbu(needs: dict, cat: dict) -> list[dict]:
    """Ext 2 — for each non-training (decode / memory-bound) GPU pick the cheapest SKU that fits.

    `needs[gpu_id]` = {type, workload, mem_gb, tflops, bw_tbs} measured from telemetry. A candidate must
    (a) hold the working set + headroom, (b) deliver the measured bandwidth at <= MBU_TARGET and the
    measured FLOPs at <= MFU_TARGET. Cheapest $/hr among those wins — NOT cheapest overall, because the
    cheapest SKU (L4: $0.80/hr) has 0.3 TB/s and would be bandwidth-starved for most decode workloads.
    """
    out = []
    for gid, n in sorted(needs.items()):
        if n["workload"] == "train":
            continue  # compute-bound training is handled by the util-lie / purchasing levers
        cur = cat[n["type"]]
        fits = []
        for g, c in cat.items():
            if (num(c["hbm_gb"]) >= n["mem_gb"] * VRAM_HEADROOM
                    and num(c["peak_bw_tbs"]) * MBU_TARGET >= n["bw_tbs"]
                    and num(c["peak_tflops_fp16"]) * MFU_TARGET >= n["tflops"]):
                fits.append(g)
        best = min(fits, key=lambda g: num(cat[g]["on_demand_hr"])) if fits else n["type"]
        cur_hr, new_hr = num(cur["on_demand_hr"]), num(cat[best]["on_demand_hr"])
        out.append({
            "gpu_id": gid, "current": n["type"], "proposed": best,
            "mem_gb": n["mem_gb"], "bw_tbs": n["bw_tbs"],
            "cur_hr": cur_hr, "new_hr": new_hr,
            "cur_usd_gb": cur_hr / num(cur["hbm_gb"]), "new_usd_gb": new_hr / num(cat[best]["hbm_gb"]),
            "new_bw": num(cat[best]["peak_bw_tbs"]),
            "monthly_savings": max(0.0, cur_hr - new_hr) * 24 * 30,
            "savings_pct": max(0.0, (1 - new_hr / cur_hr)) * 100 if cur_hr else 0.0,
        })
    return out


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    needs = defaultdict(lambda: {"mem_gb": 0.0, "tflops": [], "bw_tbs": [], "type": None, "workload": None})
    for r in tel:
        n = needs[r["gpu_id"]]
        n["type"], n["workload"] = r["gpu_type"], r["workload"]
        n["mem_gb"] = max(n["mem_gb"], num(r["mem_used_gb"]))
        n["tflops"].append(num(r["achieved_tflops"]))
        n["bw_tbs"].append(num(r["achieved_bw_tbs"]))
    for n in needs.values():
        n["tflops"] = sum(n["tflops"]) / len(n["tflops"])
        n["bw_tbs"] = sum(n["bw_tbs"]) / len(n["bw_tbs"])
    vram_table = vram_price_table(cat)
    rightsize = rightsize_by_mbu(needs, cat)

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Extension 2: right-sizing by MBU / $ per GB-VRAM --")
        print(f"{'GPU':8}{'$/hr':>6}{'HBM GB':>8}{'$/GB-VRAM-hr':>14}{'TB/s':>6}{'$/(TB/s)-hr':>13}")
        for v in vram_table:
            print(f"{v['gpu_type']:8}{v['on_demand_hr']:>6.2f}{v['hbm_gb']:>8.0f}{v['usd_per_gb_vram_hr']:>14.4f}"
                  f"{v['peak_bw_tbs']:>6.1f}{v['usd_per_tbs_hr']:>13.3f}")
        print(f"\nrequirements: VRAM >= used x{VRAM_HEADROOM}, bandwidth at <= {MBU_TARGET:.0%} MBU, FLOPs at <= {MFU_TARGET:.0%} MFU")
        print(f"{'GPU':12}{'now':>6}{'->':>3}{'proposed':>9}{'mem GB':>8}{'bw TB/s':>9}{'new peak bw':>12}{'$/hr':>12}{'saved/mo':>10}")
        for r in rightsize:
            print(f"{r['gpu_id']:12}{r['current']:>6}{'->':>3}{r['proposed']:>9}{r['mem_gb']:>8.1f}{r['bw_tbs']:>9.2f}"
                  f"{r['new_bw']:>12.1f}{r['cur_hr']:>6.2f}->{r['new_hr']:<4.2f}${r['monthly_savings']:>8,.0f}")
        print(f"right-sizing all memory-bound GPUs saves ${sum(r['monthly_savings'] for r in rightsize):,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "vram_table": vram_table, "rightsize": rightsize,
            "rightsize_monthly": round(sum(r["monthly_savings"] for r in rightsize))}


if __name__ == "__main__":
    run()
