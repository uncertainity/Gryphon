# Gryphon RTP targets

This is the tuning target, not the currently achieved simulation result. RTP values use a paid-spin bet denominator.

## Game-level targets

| Metric | Target |
|---|---:|
| Total RTP | 94.0201% |
| Base RTP | 55.1746% |
| Base-game hit rate | 23.7585% |
| Hold-and-Spin trigger rate | 1.0000% (1 in 100 paid spins) |
| Non-jackpot feature RTP | 37.5414% |
| Progressive-jackpot RTP | 1.3041% |

Base-game hit rate means the percentage of paid spins with a positive base-game award. Hold-and-Spin trigger rate is the chance that a paid spin launches the feature; it is not an internal feature-lane probability.

## RTP allocation

| Component | Target RTP |
|---|---:|
| Base paylines | 35.1196% |
| Base Collect | 20.0551% |
| Plain Hold-and-Spin | 4.4389% |
| Splitter | 4.1215% |
| Grow | 4.8395% |
| Boost | 3.4524% |
| Multiplier | 3.9567% |
| Feature Collect | 6.1750% |
| Expansion | 7.1616% |
| Mega Combo | 3.3957% |
| Mini jackpot | 0.7038% |
| Minor jackpot | 0.3550% |
| Major jackpot | 0.1852% |
| Grand jackpot | 0.0601% |
| **Total** | **94.0201%** |

## Feature-lane allocation

After the 1-in-100 Hold-and-Spin trigger is selected, exactly one route runs. The target allocates 4% of triggers to Mega Combo. Of the other 96%, 13% are Plain and 87% are split equally across the six single-Bag routes. This preserves the legacy 96% single/4% Mega split while correctly interpreting the former seventh conversion weight as the no-Bag Plain route, rather than as a seventh Bag.

The route choice is constrained by the Bags actually visible on the triggering base board: Plain is always eligible; a single-Bag route is eligible only when that Bag is visible; Mega requires all six distinct Bags. Route weights and base-reel weights are tuned so the aggregate paid-spin probabilities converge on the targets below.

| Lane | Paid-spin probability | Required conditional mean win |
|---|---:|---:|
| Plain | 0.1248% (1 in 801.28) | 35.5684x |
| Splitter | 0.1392% (1 in 718.39) | 29.6087x |
| Grow | 0.1392% (1 in 718.39) | 34.7665x |
| Boost | 0.1392% (1 in 718.39) | 24.8021x |
| Multiplier | 0.1392% (1 in 718.39) | 28.4246x |
| Collect | 0.1392% (1 in 718.39) | 44.3604x |
| Expansion | 0.1392% (1 in 718.39) | 51.4483x |
| Mega Combo | 0.0400% | 84.8922x |

The corrected routing changes how the existing non-jackpot feature budget is distributed, not its 37.5414% subtotal. The six named-feature conditional means are retained; their paid-spin frequencies move from 0.1600% to 0.1392%. The resulting 4.4389% remainder is assigned to Plain Hold-and-Spin.

Progressive jackpot awards are accounted for separately from feature wins and must not receive the feature payout multiplier. Their hit rates are calibrated against Gryphon's 3/3/3/3 token-collection rules; the rare Grand lane uses the opportunity-profile calculation described below.

The machine-readable source is `rtp_targets.csv`. The matching presentation copy is `config_render/gryphon_rtp_targets.csv`.

## Current calibrated validation

The current production configuration was validated with 10,000,000 full-game
rounds (11,124,765 paid spins) using seed `20261014`. The evaluator called the
production Numba full-game, base-game, and shared Hold-and-Spin kernels; only
the history storage was replaced by an aggregate tuning sink.

| Metric | Observed | Target | Difference |
|---|---:|---:|---:|
| Total RTP | 93.9892% | 94.0201% | -0.0309 pp |
| Base RTP | 55.2860% | 55.1746% | +0.1114 pp |
| Base-game hit rate | 23.7548% | 23.7585% | -0.0038 pp |
| Hold-and-Spin trigger rate | 0.9984% | 1.0000% | -0.0016 pp |
| Non-jackpot feature RTP | 37.4351% | 37.5414% | -0.1063 pp |
| Observed jackpot RTP | 1.2680% | 1.3041% | -0.0361 pp |

No Grand jackpot occurred in this sample, which is expected for its modeled
paid-spin probability of approximately 1 in 38.25 million. Adding the 0.0601%
modeled Grand contribution to the sampled result gives 94.0493% expected total
RTP, 0.0292 percentage points above target. Mini, Minor, and Major measured
0.7203%, 0.3617%, and 0.1861% respectively. Grand therefore requires the
opportunity-profile calculation or a much larger rare-event certification run;
a zero-Grand 10-million-round sample must not be interpreted as zero Grand RTP.

The retained validation artifacts are in `results/`. The selected working
tolerances for this pass are +/-0.10 percentage points for total expected RTP,
+/-0.15 points for the base and non-jackpot feature subtotals, +/-0.05 points
for base hit rate, and +/-0.02 points for overall Hold-and-Spin frequency.
