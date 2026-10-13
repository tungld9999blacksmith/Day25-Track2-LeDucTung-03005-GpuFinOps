"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ---------------------------------------------------------------------------
# Extension 1 — interruption-aware, term-aware tier policy
# ---------------------------------------------------------------------------
# Illustrative per-hour spot reclaim probability by GPU type (scarcer / smaller
# SKUs get reclaimed more often). Assumption — re-baseline against real data.
SPOT_INTERRUPT_RATE = {
    "B200": 0.03, "H100": 0.04, "H200": 0.05, "MI300X": 0.06,
    "A100": 0.06, "L4": 0.10, "A10G": 0.12,
}
DEFAULT_INTERRUPT_RATE = 0.05


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str,
    prices: dict,
    job_days: float = 30,
    horizon_months: int = 36,
) -> dict:
    """Pick the cheapest tier per *used* GPU-hour, with real commitment costs.

    Improvements over recommend_tier():
      1. spot is costed with a per-GPU-type interruption rate (rework + checkpoints),
         not assumed to always win;
      2. reserved is billed for ALL 24h of the day (a commitment is not pay-per-use),
         so its cost per used hour is rate * 24 / hours_per_day;
      3. 1yr vs 3yr: a term is only eligible if the capacity is needed for at least
         that long (``horizon_months``); bursty jobs (< 30 days) get no reservation.

    `prices` needs on_demand_hr, spot_hr, reserved_1yr_hr, reserved_3yr_hr.
    Returns {"tier", "term", "cost_per_used_hr", "candidates"}; tier is one of
    'spot' | 'reserved' | 'on_demand' (term is '1yr'/'3yr' when reserved).
    """
    hpd = max(1e-9, float(hours_per_day))
    od = float(prices["on_demand_hr"])
    cands = {("on_demand", None): od}
    if interruptible:
        sim = spot_checkpoint_cost(
            100.0, float(prices["spot_hr"]), od,
            interrupt_rate=SPOT_INTERRUPT_RATE.get(gpu_type, DEFAULT_INTERRUPT_RATE))
        cands[("spot", None)] = sim["spot_cost"] / 100.0
    if job_days >= 30:  # steady service, otherwise no point committing
        for term, months, key in (("1yr", 12, "reserved_1yr_hr"), ("3yr", 36, "reserved_3yr_hr")):
            if horizon_months >= months and key in prices:
                cands[("reserved", term)] = float(prices[key]) * 24.0 / hpd
    (tier, term), cost = min(cands.items(), key=lambda kv: kv[1])
    return {"tier": tier, "term": term, "cost_per_used_hr": round(cost, 4),
            "candidates": {f"{t}{'-' + m if m else ''}": round(c, 4) for (t, m), c in cands.items()}}


# ---------------------------------------------------------------------------
# Extension 3 — is prompt caching actually worth it?
# ---------------------------------------------------------------------------
def cache_breakeven_reads(write_cost: float = 0.25, read_discount: float = 0.10) -> float:
    """Reads per cached prefix needed before caching beats not caching.

    `write_cost` is the EXTRA cost of creating the cache, as a multiple of the normal
    input price (Anthropic writes bill 1.25x -> 0.25; storage-billed caches add
    storage $ on top). Over 1 write + n reads:
        no cache : (n + 1) * P
        cache    : (1 + write_cost) * P + n * read_discount * P
    -> worth it when n > write_cost / (1 - read_discount).
    """
    if read_discount >= 1.0:
        return float("inf")
    return max(0.0, write_cost) / (1.0 - read_discount)


def cache_is_worth_it(avg_reads: float, write_cost: float = 0.25, read_discount: float = 0.10) -> bool:
    """True when the average reads per cached prefix exceed the break-even."""
    return avg_reads > cache_breakeven_reads(write_cost, read_discount)
