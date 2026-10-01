# Gryphon current tuning summary

Updated: 2026-10-02

## Engine boundary

The integrated game remains inside the Numba framework. Tuning code parses
targets, constructs configuration values, and measures the production kernels;
it does not implement an alternate base game or Hold-and-Spin flow.

The production route is:

```text
simulations/full_game.py
  -> simulations/base_game.py
  -> simulations/free_game.py
  -> core/hold_and_spin_kernels.py
```

## Selected controls

| Control | Selected value |
|---|---:|
| Base reelset weights | 0.1711049402 / 0.8273723490 / 0.0015227108 |
| Plain route relative weight | 0 |
| Single-Bag relative weights | 1 / 1 / 1 / 1 / 1 / 1 |
| Base pay-table scale | 2.05046 |
| Base Coin-drop probabilities | 0.0868970066 / 0.90 / 0.90 |
| Conditional Coin-drop mean | 3.90 cells |
| Base Coin-credit mean | 0.361x |
| Jackpot token probability | 0.9999999362 |
| Jackpot type weights | 0.8294530347 / 0.1321823832 / 0.0336458856 / 0.0047186966 |
| Hard round cap | 15,000x |

Feature payout multipliers, in reporting order, are:

```text
Splitter   1.720929400749
Grow       0.941021135063
Boost      0.564453709225
Multiplier 1.066429853626
Collect    1.566930088682
Expansion  1.729551533546
Mega       0.343879700019
Plain      3.155013434     # retained path; live probability is zero
```

## Trigger solve

`ReelSet_3.csv` remains the rare Mega-capable reelset. It uses the ordinary
reel selector, 3x5 board generator, generic-SC conversion, and all-six-Bag
detection. There is no post-spin Mega gate.

The target sheet requests mutually exclusive rates totaling 1 in 71.43 while
also requesting an overall 1 in 75 rate. The fallback fixes overall H&S at 1
in 75, Mega at 1 in 1,500, and Plain at zero; each single Bag receives the
remaining 0.211111% paid-spin probability, or 1 in 473.68.

The independent 10M routing run measured:

| Route | Observed | Target |
|---|---:|---:|
| Splitter | 0.20958% | 0.211111% |
| Grow | 0.20916% | 0.211111% |
| Boost | 0.21230% | 0.211111% |
| Multiplier | 0.21217% | 0.211111% |
| Collect | 0.21077% | 0.211111% |
| Expansion | 0.21008% | 0.211111% |
| Mega | 0.06639% | 0.066667% |
| Plain | 0 | 0 |
| **Any H&S** | **1.33045%** | **1.333333%** |

## Base calibration

The exact natural Collect window probability is 1 in 24.83. A 90% drop chance
when a Collector is active and a 3.90-cell conditional drop mean jointly meet
the requested 3.5-5 coins per Collector without exceeding the 30% base-hit
ceiling.

The 5M base run measured 55.0589% RTP and 29.8955% hit rate. The earlier 3M
mechanic diagnostic measured 3.5128 collected cells per Collector and 3.8994
cells per successful drop.

## Conditional mechanic measurements

The 20K-session-per-route pass measured:

| Route | Respins | Final Coins | Key mechanic result |
|---|---:|---:|---|
| Splitter | 5.72 | 10.03 | 5.21 generated split coins |
| Grow | 11.29 | 8.30 | 4.07 Grow resolutions |
| Boost | 13.44 | 8.64 | 3.78 Boost resolutions |
| Multiplier | 13.08 | 8.78 | 3.31 Multiplier resolutions |
| Collect | 12.36 | 8.25 | 2.25 successful collections |
| Expansion | 17.80 | 17.36 | 2.03 distinct rows; 40.56% all rows |
| Mega | 17.32 | 14.88 | 18.92 visible backing symbols; 43.56% all rows |

Multiplier Bags place persistent 2x/3x/4x values onto empty active cells;
landing Coins consume those cells. Expansion and Mega use the same 6x5
backing board with three initially locked rows.

## Jackpot feasibility and calibration

The one-token-attempt-per-respin mechanic cannot satisfy the requested Mini,
Minor, Major, and Grand rates simultaneously. The constrained production-
opportunity fit reaches approximately:

| Jackpot | Fitted paid-spin odds | Requested |
|---|---:|---:|
| Mini | 1 in 108.30 | 1 in 93.75 |
| Minor | 1 in 1,507.84 | 1 in 1,500 |
| Major | 1 in 50,053 | 1 in 50,000 |
| Grand | 1 in 15,002,106 | 1 in 15,000,000 |

This is the closest fit without changing token eligibility or feature flow.
The jackpot kernel also consumes a draw that selects an already-awarded tier;
it no longer retries candidates and unintentionally redistributes that mass to
rarer jackpots.

## Integrated validation

The retained final run used 10,000,000 rounds, 10,770,660 paid spins, and seed
`20261001`.

| Metric | Observed | Reconciled target |
|---|---:|---:|
| Total RTP, no Grand observed | 93.8741% | 94.0201% expected |
| Total plus modeled Grand | 93.9741% | 94.0201% |
| Base RTP | 55.0155% | 55.0000% |
| Base hit rate | 29.8721% | 25%-30% |
| Non-jackpot feature RTP | 30.3756% | 30.4812% |
| Hold-and-Spin trigger rate | 1.32922% | 1.33333% |
| Mini RTP | 5.8432% | 5.8347% modeled |
| Minor RTP | 1.6214% | 1.6126% modeled |
| Major RTP | 1.0184% | 0.9916% modeled |
| Grand RTP | 0 observed | 0.1000% modeled |

Using the fixed 55% base fallback, exact configured route rates, 20K
conditional feature means, fitted jackpot hit rates, and 10M progressive award
means gives 94.0260% expected RTP, 0.0059 percentage points above target.

## Reproduction

```bash
python -m Numba_Engine.Tuning.evaluate --base-only --base-spins 5m
python -m Numba_Engine.Tuning.evaluate --routing-only --routing-spins 10m
python -m Numba_Engine.Tuning.evaluate --features-only --feature-sessions 20k
python -m Numba_Engine.Tuning.tune_jackpots --sessions-per-route 20k
python -m Numba_Engine.Tuning.evaluate --full-only --full-rounds 10m
```

Final evidence is retained in `Tuning/results/new_targets_*_final_*.json` plus
`new_targets_features_20k.json` and `new_targets_jackpot_fit_20k.json`.
