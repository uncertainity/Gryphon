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
| Splitter | 4.7374% |
| Grow | 5.5626% |
| Boost | 3.9683% |
| Multiplier | 4.5479% |
| Feature Collect | 7.0977% |
| Expansion | 8.2317% |
| Mega Combo | 3.3957% |
| Mini jackpot | 0.7038% |
| Minor jackpot | 0.3550% |
| Major jackpot | 0.1852% |
| Grand jackpot | 0.0601% |
| **Total** | **94.0201%** |

## Feature-lane allocation

After the 1-in-100 Hold-and-Spin trigger is selected, the target model allocates six single-feature lanes at 0.16% of paid spins each (1 in 625) and Mega Combo at 0.04% (1 in 2,500). Together they equal the 1.00% overall trigger rate.

| Lane | Paid-spin probability | Required conditional mean win |
|---|---:|---:|
| Splitter | 0.1600% | 29.6087x |
| Grow | 0.1600% | 34.7665x |
| Boost | 0.1600% | 24.8021x |
| Multiplier | 0.1600% | 28.4246x |
| Collect | 0.1600% | 44.3604x |
| Expansion | 0.1600% | 51.4483x |
| Mega Combo | 0.0400% | 84.8922x |

Progressive jackpot awards are accounted for separately from feature wins and must not receive the feature payout multiplier. Jackpot hit rates still require tuning for Gryphon's 3/3/3/3 token-collection rules.

The machine-readable source is `rtp_targets.csv`. The matching presentation copy is `config_render/gryphon_rtp_targets.csv`.
