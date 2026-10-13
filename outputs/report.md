# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

## Unit economics ($/1M-token)

- Baseline (all-large model, no cache, no batch): **$6.488/1M-token** ($48.87/day)
- Optimized (cascade + caching + batch): **$1.126/1M-token** ($8.48/day)
- Inference savings: **82.6%** over 7,533,027 tokens

## Lever breakdown and priority

| Rank | Lever | Savings / month | % of baseline | % of total savings |
|---|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | 37.0% | 80% |
| 2 | Inference (cascade/cache/batch) | $1,212 | 4.5% | 10% |
| 3 | Right-size util-lies | $655 | 2.4% | 5% |
| 4 | Kill idle GPUs | $600 | 2.2% | 5% |

Ranked by dollars. By effort: *Kill idle GPUs* is a scheduler change (hours), *Purchasing* needs a commitment decision (days; spot is reversible, reserved is not), *Right-size* needs a migration test, *Inference* needs routing/caching plumbing. Start with idle + spot (cheap, immediate), then reserved for the steady inference fleet.

## Why GPU-Util is a lie

`nvidia-smi` GPU-Util is the share of sample periods in which *at least one kernel was running*. It says nothing about how many SMs were busy or whether the tensor cores were fed. A GPU stalled on HBM reads, waiting on all-reduce or data loading, or running tiny kernels with launch overhead still reports ~100%. MFU (achieved FLOPs / peak FLOPs) is what you are actually paying for.

| GPU | Type | GPU-Util | MFU | Wasted $/hr | Wasted $/month |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.19 | $2.02 | $1,451 |
| gpu-a10g-1 | A10G | 96.9% | 0.27 | $0.73 | $527 |

These GPUs look healthy on a dashboard yet leak **$1,978/month** in unused FLOPs. Fix the bottleneck (batch size, dataloader, comm overlap, fused kernels) before buying more GPUs; if it cannot be fixed, right-size down (lever above).

## Extension results

**Ext 1: interruption/term-aware tier policy** (`pricing.recommend_tier_v2`). The old policy reported 39.1% savings but bills reserved only for used hours; re-priced with the true 24h commitment it is 37.9%. The new policy (per-GPU interruption rate, 1yr vs 3yr eligibility) gives 38.0% with realistic accounting, and 29.5% if only 1-year commitments are safe. Most jobs keep the same tier; the value is an honest number and a decision matrix.

**Ext 2: right-sizing by MBU** (M1). gpu-a10g-0 A10G -> L4 ($144/mo). Total $144/month. Cheapest $/hr is not enough: the SKU must hold the working set and supply the bandwidth at <=60% MBU (L4 is cheapest but has 0.3 TB/s).

**Ext 3: `cache_is_worth_it`**. Observed cache-hit ratio small 32%, large 32% gives ~0.47 reads/prefix vs a break-even of 0.28 with 1.25x writes: worth it (cache saves $1.17/day). With storage-billed caching the break-even rises to 5.8 (small), 1.9 (large) reads, and caching would lose money at this hit rate.

**Ext 4: reasoning budget**. Reasoning is 8.4% of traffic but 16.5% of dollars and 94% of energy (about 173x Wh per query: 80x multiplier times ~2.2x tokens). Rule: allow reasoning only for the longest/hardest prompts (<=5% of traffic) or when the cheap answer fails validation. cap 5% saves $14/mo and 31% of energy; cap 2% saves $23/mo and 68% of energy.

## Sustainability: region and carbon

| Region | $/kWh | gCO2/kWh | Energy $ | CO2 kg | Carbon-adjusted $ (@ $100/t) |
|---|---|---|---|---|---|
| us-east-1 | 0.120 | 380 | $214.68 | 679.8 | $282.66 |
| us-west-2 | 0.070 | 120 | $125.23 | 214.7 | $146.70 |
| europe-north1 | 0.090 | 30 | $161.01 | 53.7 | $166.38 |
| europe-central2 | 0.180 | 660 | $322.02 | 1,180.7 | $440.09 |
| us-east-wa | 0.055 | 90 | $98.39 | 161.0 | $114.50 |

Interruptible jobs use 1,789 kWh. Cheapest electricity: **us-east-wa**; cleanest: **europe-north1**; best balance with an internal carbon price: **us-east-wa**. Moving them from us-east-1 to europe-north1 avoids 626 kgCO2e (92%). Strictly, the cheapest grid is us-east-wa, so the 'cheapest+cleanest' line above means cleanest. Trade-off: batch/training tolerates distance; latency-sensitive inference should stay near users. GPU rental prices already include power, so the electricity column is an indicator, while carbon is the real saving.

## Recommended actions (priority order)

1. **Move steady inference to 3yr reserved and training to spot** ($10,040/mo, largest lever).
2. **Auto-stop idle GPUs** (zero risk, immediate).
3. **Deploy cascade + caching + batch routing** (82.6% off the inference bill, also lowers carbon).
4. **Replace GPU-Util with MFU/MBU alerts**; fix or right-size the util-lie GPUs.
5. **Cap reasoning to <=5% of traffic** and schedule interruptible jobs in the cleanest/cheapest region.
6. **Tag everything, then chargeback** (the 80% coverage gate is already open).
