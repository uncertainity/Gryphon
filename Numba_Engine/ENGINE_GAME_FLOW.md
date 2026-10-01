# Gryphon Numba Engine Game Flow

## Scope

This document describes the game flow executed by the integrated Numba engine.
It describes runtime mechanics, not the current RTP tuning, target allocation,
or search strategy.

The integrated path is:

```text
Numba_Engine/simulations/full_game.py
    -> Numba_Engine/simulations/base_game.py
    -> Numba_Engine/simulations/free_game.py
    -> Numba_Engine/core/hold_and_spin_kernels.py
```

The integrated full game does not import or call `legacy/`. The older
per-feature kernels remain available as standalone compatibility helpers, but
they are not dispatched by `full_game.py`.

## Flow at a glance

```text
begin paid round
    -> resolve one 3x5 base spin
    -> evaluate paylines
    -> resolve base Coin/Collector behavior
    -> apply base jackpot overlays
    -> if generic SC is present, convert every SC to SC1-SC6
    -> identify the distinct visible Bag types
    -> select exactly one eligible route
         -> Plain, or one visible SC1-SC6 single-Bag route
         -> Mega Combo when all six distinct Bags are visible
    -> start exactly one Hold-and-Spin session
         -> Plain and SC1-SC5 use a logical 3x5 board
         -> Expansion and Mega use a 6x5-capable board
         -> one respin counter and one set of jackpot meters
         -> only Bags enabled by the selected route may land
         -> Expansion unlocks rows on the same board
         -> resolve Bags in the configured order
    -> pay non-jackpot feature win and any progressive jackpots
    -> move walking base Collectors
    -> repeat a paid spin while a Collector remains
    -> finish the round
```

## 1. Paid base spin

Each paid spin builds a 3-row by 5-reel window from the configured base reels.
Walking Collectors carried from the preceding paid spin are applied before the
spin is evaluated, and natural Collectors on the new window are registered.

The base award is resolved in this order:

1. Evaluate the 20 configured paylines from left to right.
2. Build the separate Coin-credit overlay.
3. Multiply the visible Coin-credit total by the active Collector count.
4. Apply jackpot overlays and increment the persistent progressive values.
5. Test the base window for a generic `SC` trigger.

Line symbols and the Coin-credit overlay are kept as separate data. Jackpot
overlays increment progressives; they do not award a jackpot in the base game.

## 2. Convert the base trigger

When at least one generic `SC` is visible, every generic `SC` cell is converted
independently using `BaseGameConfig.scatter_feature_symbols` and its probability
table. The converted base window is stored for inspection.

The integrated feature runner records which of the six Bag types are visible:

| Symbol | Bag power |
|---|---|
| `SC1` | Splitter |
| `SC2` | Grower |
| `SC3` | Booster |
| `SC4` | Multiplier |
| `SC5` | Collector |
| `SC6` | Expansion |

Duplicate symbols do not create extra sessions or extra route entries. For
example, `SC1, SC2, SC1` makes three routes eligible: Plain, Splitter, or
Grower. Configured relative weights choose exactly one of them.

There is no `SC7` conversion outcome. A partial set of different Bags is never
combined. Mega Combo is eligible only when all six distinct symbols `SC1`
through `SC6` are present; the current base configuration then launches it
deterministically.

## 3. One routed Hold-and-Spin session

The selected route is passed to one call to `hold_and_free_spin`. There is no
loop that launches one feature session per visible Bag type.

Every session owns exactly one of each of the following:

- one logical feature board;
- persistent Coin mask;
- locked-row boundary;
- remaining-respins counter;
- Collector meter;
- four jackpot-token meters; and
- detailed session/respin/step storage history.

The eight reporting routes are Splitter, Grow, Boost, Multiplier, Collect,
Expansion, Mega Combo, and Plain. One and only one route is recorded for each
triggered session.

## 4. Logical board size and fixed-shape storage

Numba storage arrays use one fixed maximum shape of 6x5. This is a storage
envelope, not a rule that every route plays on six rows.

For Plain and the SC1-SC5 routes:

- the logical board is 3x5;
- the upper three storage rows are inactive padding and are never spun,
  occupied, counted, or paid; and
- Plain starts without a Bag, while a single-Bag route starts with exactly its
  selected Bag.

For Expansion and Mega Combo:

- the logical backing board is 6x5;
- rows 0-2 begin locked and rows 3-5 begin active;
- Expansion starts with SC6, while Mega starts with one of every SC1-SC6; and
- locked rows can stage symbols until Expansion unlocks them.

For every route, the configured number of starting Coins is placed into active
cells and three respins are awarded by default. Base positions are not reused
as fixed feature positions; starting symbols are placed into random available
feature cells.

## 5. Respin and landing flow

At the start of each respin, the kernel counts occupied cells in the route's
playable region. Each playable empty position then uses:

- the locked or active scalar landing probability; or
- an occupied-count override when that table entry is configured.

The occupied count is updated as symbols land, so later cell trials in the same
respin can use a different state entry. A landed symbol is chosen from the
configured type table after filtering it to the selected route:

- Plain permits only natural Coins;
- a single-Bag route permits natural Coins and its selected Bag; and
- Mega permits natural Coins and all six Bags.

An occupied-count-specific type distribution can override the default
distribution before the same route filter is applied.

Only Expansion and Mega have locked rows. Those rows can accumulate symbols,
but the symbols do not act or pay while locked. Expansion moves the boundary
past a row, making its staged symbols active and eligible for the final award.

## 6. Jackpot tokens

New natural Coins are eligible for the feature jackpot-token overlay. A token:

- advances one of the four session-local meters;
- does not replace or remain on the board; and
- does not participate in Bag transformations.

Reaching the configured collection target marks that progressive jackpot as
awarded. The full-game runner pays the current progressive value and resets the
awarded jackpot to its configured seed.

Jackpot types are always sampled from the configured fixed type weights. Once
a type has awarded in the current session, a later draw of that type is
non-awarding; its probability mass is not reassigned to rarer types.

## 7. Bag resolution order

After landing symbols, the engine scans the routed board and resolves every
active Bag in this configured order:

1. Expansion (`SC6`)
2. Splitter (`SC1`)
3. Booster (`SC3`)
4. Grower (`SC2`)
5. Multiplier (`SC4`)
6. Collector (`SC5`)

This order is data in `HoldAndSpinConfig.bag_resolution_order`; it is not a
tuning-script branch.

### Expansion

Each active Expansion unlocks the configured number of rows, one by default,
then applies its configured post-resolution action. Expansion is processed
bottom-up so a newly unlocked row can expose another stored Expansion in the
same resolution pass.

### Splitter

Splitter selects existing active Coins and copies their values into available
active positions. Generated Coins do not recursively split during that same
resolution step.

### Booster

Booster adds a sampled value to every supplied active Coin, subject to the
configured Coin-value cap.

### Grower

Grower selects a configured number of unique active Coins and adds a sampled
value to them.

### Multiplier

Multiplier multiplies every supplied active Coin by a sampled value, subject
to the configured Coin-value cap.

### Collector

Collector adds the current active Coin total to the persistent session
Collector meter. The configured maximum number of Collector events limits how
often this can happen in one session.

## 8. Post-resolution Bag actions and respin reset

Each Bag type has a configurable action after it resolves:

- `0`: remain as the Bag symbol;
- `1`: disappear; or
- `2`: convert to a normal credit Coin.

The default actions are:

| Bag | Default action |
|---|---|
| Splitter | convert to Coin |
| Grower | remain |
| Booster | disappear |
| Multiplier | disappear |
| Collector | convert to Coin |
| Expansion | convert to Coin |

Reset behavior is also config data aligned with the landing-symbol types. By
default, a newly landed natural Coin, Splitter, Grower, or Collector resets the
counter to three when its position is active after Expansion resolution.
Booster, Multiplier, and Expansion do not reset under the configured reset
flags. Mechanic-created Coins do not create reset chains.

If no qualifying symbol landed, the remaining-respins counter decreases by
one. The feature ends when the counter reaches zero or the active area has no
empty position.

## 9. Feature award and reporting categories

At feature end:

```text
raw non-jackpot feature win
    = sum(final persistent Coin values)
    + Collector meter
```

The full-game runner multiplies this non-jackpot component by the configured
factor for its starting-Bag reporting category: Splitter, Grow, Boost,
Multiplier, Collect, Expansion, Mega Combo, or Plain. This does not route into
another feature implementation or alter board play. Progressive jackpot
awards are paid separately and are never passed through these multipliers.

The same feature result is written to:

- detailed `HoldAndSpinStorage` for board/respin/mechanic inspection; and
- `FullGameStorage` for paid-spin, round, RTP-component, and reporting totals.

## 10. Configuration and tuning boundary

Runtime mechanics consume configuration only. The engine does not import a
tuning script or decide how a candidate configuration should be searched.

The shared Hold-and-Spin config exposes, among other controls:

- Plain and per-Bag route-selection weights;
- scalar locked/active landing probabilities;
- landing probabilities indexed by occupied-cell count;
- default and occupied-count-specific landed-symbol distributions;
- landed-symbol respin-reset flags;
- starting Coin distributions and Coin values;
- Bag resolution order and post-resolution actions;
- Splitter/Grower/Booster/Multiplier strength tables;
- Collector event limit and Expansion row count; and
- jackpot-token probabilities and collection targets.

Tuning code may construct or replace these arrays later, but the Numba runtime
remains the sole implementation of the game flow.
