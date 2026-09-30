# Gryphon Base Math Tuning Summary

Updated: 2026-09-28

## Targets

| Metric | Target |
|---|---:|
| Base line win RTP | 35% |
| Collect feature RTP | 20% |
| Collect symbol frequency | 1 in 20 spins |
| Overlay coins on window | 1-3 coins per spin |
| Coin value average | approx. 1.2 |
| Coin count when drop triggers | 2-6 coins |
| Average coin count per Collect symbol | 4-6 coins |
| Validation cycle | 10,000,000 spins |

## Implemented Mechanics

1. Base reel strips use a generic `SC` symbol only. `SC1`-`SC6` remain feature-only bag symbols and do not appear on base reel strips.
2. Before each paid base spin, the game checks whether to drop overlay Coins.
3. Coin-drop probability uses three buckets:
   - `P1` when there are 0 Collect symbols on screen.
   - `P2` when there is exactly 1 Collect symbol on screen.
   - `P3` when there is more than 1 Collect symbol on screen.
4. If a coin drop triggers, the game drops 2-6 overlay Coins by weighted count.
5. Overlay Coins cannot land on Wild, Collect, existing Coin, or generic `SC` cells.
6. Each visible Collect symbol independently collects the full overlay coin total.
7. Reel 3 and reel 4 each contain a stack of 3 Collect symbols. Other reels use stack-2 and single Collect placements.

## Current Tuned Configuration

| Area | Parameter | Current value |
|---|---|---|
| Reelsets | `ReelSet_1.csv`, `ReelSet_2.csv` | Generated identically from `Numba_Engine/Tuning/tune_base_reels.py` |
| Reel length | all reels | `1000` |
| Reelset weights | `BASE_GAME_CONFIG.reelset_probabilities` | `[0.50, 0.50]` |
| Paytable | H1 | `(5OAK=10, 4OAK=3, 3OAK=1)` |
| Paytable | H2 | `(5OAK=5, 4OAK=2, 3OAK=0.6)` |
| Paytable | H3 | `(5OAK=4, 4OAK=1.6, 3OAK=0.4)` |
| Paytable | H4/H5 | `(5OAK=3, 4OAK=1, 3OAK=0.4)` |
| Paytable | L1-L6 | `(5OAK=2, 4OAK=0.6, 3OAK=0.2)` |
| Coin-drop probabilities | `P1, P2, P3` | `[0.30, 0.45, 0.65]` |
| Coin-drop count weights | counts `2, 3, 4, 5, 6` | `[0.08, 0.16, 0.34, 0.28, 0.14]` |
| Coin values | value set | `[0.2, 0.3, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]` |
| Coin value weights | probabilities | `[0.030, 0.040, 0.070, 0.100, 0.100, 0.150, 0.170, 0.170, 0.120, 0.050]` |
| Coin value average | weighted average | `1.197x` |

## 10M Validation

Saved to `Numba_Engine/Tuning/latest_base_tuning.json`.

| Metric | Measured |
|---|---:|
| Spins | 10,000,000 |
| Base line RTP | 35.1505% |
| Collect RTP | 20.0754% |
| Base total RTP | 55.2259% |
| Collect window frequency | 5.2912%, approx. 1 in 18.9 |
| Average Collect symbols on Collect spin | 1.3626 |
| Coin-drop spin frequency | 31.1192% |
| Average dropped coin count when triggered | 4.2390 |
| Average overlay coin count per spin | 1.3191 |
| Generic SC window frequency | 4.9083% |

## Reel Constraint Check

The generated reelsets passed the following checks:

1. Every reel has at least one of every line-symbol kind.
2. Reel 1 has no Wild symbol.
3. No base reel contains `SC1`, `SC2`, `SC3`, `SC4`, `SC5`, or `SC6`.
4. No base reel contains natural `COIN`; Coins are now overlay drops.
5. Generic `SC` appears on the base reels.
6. There is at least one symbol between any two generic `SC` symbols.
7. There are at least two symbols between every Collect and generic `SC` symbol.
8. Reel 3 and reel 4 each include a Collect stack of 3.
9. Reels 1, 2, and 5 use stack-2/single Collect placements.
10. Each reel can show up to 2 generic SC symbols in a 3-row window, so 6-8 scatters on a full 5-reel window is possible.

## Verification

`pytest` is not installed in this Python environment, so `python -m pytest -q` could not run.

The existing pytest-style test functions were imported and called directly; they passed:

```text
manual pytest-style tests passed
```

## Remaining Notes

1. The line and Collect targets are close on the 10M sample. For final certification, run an independent larger validation seed.
2. Generic `SC` currently acts as the base trigger symbol. The later conversion from `SC` to `SC1`-`SC6` by weights still needs an explicit selection layer when the feature is launched from base game.
3. The Hold-and-Spin launch path is still not wired directly into `base_game.py`; this pass focused on base reel/line/Collect math.
