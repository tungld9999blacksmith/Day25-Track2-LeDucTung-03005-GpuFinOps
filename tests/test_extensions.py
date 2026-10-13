"""Tests for the 'Your Turn' extensions (new file; the graded tests are untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing
from missions._common import catalog_by_type, num

H100 = {"on_demand_hr": 2.5, "spot_hr": 1.5, "reserved_1yr_hr": 2.0, "reserved_3yr_hr": 1.4}


# Ext 1
def test_v2_interruptible_prefers_spot():
    assert pricing.recommend_tier_v2(8, True, "H100", H100)["tier"] == "spot"


def test_v2_steady_service_goes_reserved_3yr():
    v = pricing.recommend_tier_v2(24, False, "H100", H100)
    assert (v["tier"], v["term"]) == ("reserved", "3yr")


def test_v2_term_depends_on_horizon():
    v = pricing.recommend_tier_v2(24, False, "H100", H100, horizon_months=12)
    assert (v["tier"], v["term"]) == ("reserved", "1yr")
    assert pricing.recommend_tier_v2(24, False, "H100", H100, horizon_months=6)["tier"] == "on_demand"


def test_v2_low_duty_or_short_job_is_on_demand():
    assert pricing.recommend_tier_v2(4, False, "H100", H100)["tier"] == "on_demand"
    assert pricing.recommend_tier_v2(24, False, "H100", H100, job_days=7)["tier"] == "on_demand"


def test_v2_bills_reserved_for_full_day():
    v = pricing.recommend_tier_v2(18, False, "H100", H100)
    assert abs(v["candidates"]["reserved-3yr"] - 1.4 * 24 / 18) < 1e-3


def test_policy_comparison_runs_and_is_below_old_reported():
    from missions._common import load_csv
    p = m3_purchasing.policy_comparison(load_csv("workloads.csv"), catalog_by_type())
    assert 0 < p["new_pct"] <= p["old_reported_pct"]


# Ext 2
def test_vram_table_sorted_by_price_per_gb():
    t = m1_efficiency_audit.vram_price_table(catalog_by_type())
    vals = [r["usd_per_gb_vram_hr"] for r in t]
    assert vals == sorted(vals) and t[0]["gpu_type"] == "MI300X"


def test_rightsize_respects_bandwidth_and_vram():
    cat = catalog_by_type()
    needs = {"g1": {"type": "A10G", "workload": "infer", "mem_gb": 20.0, "tflops": 30.0, "bw_tbs": 0.14},
             "g2": {"type": "A10G", "workload": "infer", "mem_gb": 20.0, "tflops": 30.0, "bw_tbs": 0.3},
             "g3": {"type": "H100", "workload": "train", "mem_gb": 60.0, "tflops": 400.0, "bw_tbs": 1.5}}
    out = {r["gpu_id"]: r for r in m1_efficiency_audit.rightsize_by_mbu(needs, cat)}
    assert out["g1"]["proposed"] == "L4"          # fits and is cheaper
    assert out["g2"]["proposed"] == "A10G"        # L4's 0.3 TB/s cannot serve 0.3 TB/s at <=60% MBU
    assert "g3" not in out                        # training is skipped


# Ext 3
def test_cache_breakeven_and_gate():
    assert abs(pricing.cache_breakeven_reads(0.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert pricing.cache_breakeven_reads(0.0) == 0.0
    assert pricing.cache_is_worth_it(1.0, 0.25, 0.10)
    assert not pricing.cache_is_worth_it(0.2, 0.25, 0.10)
    assert not pricing.cache_is_worth_it(2.0, write_cost=5.0)  # expensive storage


def test_cache_economics_on_dataset():
    from missions._common import load_csv
    info = m2_inference_levers.cache_economics(load_csv("token_usage.csv"))
    assert set(info) == {"small", "large"} and all(v["worth_it"] for v in info.values())
    assert all(not v["worth_it_storage"] for v in info.values())


# Ext 4
def test_reasoning_budget_shares_and_caps():
    r = m2_inference_levers.run(verbose=False)["reasoning"]
    assert r["wh_share"] > r["cost_share"] > r["traffic_share"] > 0
    s5 = next(s for s in r["scenarios"] if s["cap"] == 0.05)
    assert s5["cost_saved"] > 0 and s5["wh_saved"] > 0
    s10 = next(s for s in r["scenarios"] if s["cap"] == 0.10)
    assert s10["cost_saved"] == 0  # already below a 10% cap


# Ext 5
def test_region_comparison_and_carbon_schedule():
    kwh = sustainability.job_energy_kwh(700, 8, 20, 14)
    assert abs(kwh - 1568.0) < 1e-6
    rows = {r["region"]: r for r in sustainability.region_comparison(kwh)}
    assert rows["europe-north1"]["co2_kg"] < rows["us-east-1"]["co2_kg"] < rows["europe-central2"]["co2_kg"]
    from missions._common import load_csv
    c = m3_purchasing.carbon_schedule(load_csv("workloads.csv"), catalog_by_type())
    assert c["cleanest"] == "europe-north1" and c["cheapest"] == "us-east-wa" and c["saved_kg"] > 0
