import numpy as np
from numba import njit
from typing import NamedTuple

from .kernels import probChoice

@njit
def splitter_bag(
    pay_window,
    coin_mask,
    coin_positions,
    source_count_probabilities,
    source_counts,
    copy_count_probabilities,
    copy_counts,
    locked_row_idx,
):
    """Copy newly landed coins into random empty, unlocked positions.

    ``coin_positions`` contains flattened positions of qualifying coins from
    the current spin.  It must be captured before calling this function so a
    generated coin cannot become another splitter source on the same spin.
    """
    num_rows, num_reels = pay_window.shape
    num_candidates = len(coin_positions)

    if num_candidates == 0 or locked_row_idx + 1 >= num_rows:
        return pay_window

    available_positions = np.empty(num_rows * num_reels, dtype=np.int32)
    num_available = 0
    for row in range(locked_row_idx + 1, num_rows):
        for col in range(num_reels):
            if pay_window[row, col] == 0:
                available_positions[num_available] = row * num_reels + col
                num_available += 1

    if num_available == 0:
        return pay_window

    chosen_num_to_split = int(
        probChoice(
            source_count_probabilities,
            source_counts,
        )
    )
    chosen_num_to_split = min(chosen_num_to_split, num_candidates)

    # Shuffle a local index buffer so source coins are chosen without
    # replacement.  This is supported consistently by Numba and avoids a
    # Python list or set in the compiled kernel.
    candidate_indices = np.arange(num_candidates, dtype=np.int32)
    for idx in range(chosen_num_to_split):
        swap_idx = np.random.randint(idx, num_candidates)
        temp = candidate_indices[idx]
        candidate_indices[idx] = candidate_indices[swap_idx]
        candidate_indices[swap_idx] = temp

        source_position = int(coin_positions[candidate_indices[idx]])
        source_row = source_position // num_reels
        source_col = source_position % num_reels

        # Ignore malformed or stale candidate entries rather than copying a
        # non-coin/empty value onto the board.
        if (
            source_position < 0
            or source_position >= num_rows * num_reels
            or source_row <= locked_row_idx
            or pay_window[source_row, source_col] <= 0
        ):
            continue

        chosen_num_splits = int(
            probChoice(copy_count_probabilities, copy_counts)
        )
        chosen_num_splits = min(chosen_num_splits, num_available)

        for _ in range(chosen_num_splits):
            available_idx = np.random.randint(0, num_available)
            target_position = available_positions[available_idx]
            target_row = target_position // num_reels
            target_col = target_position % num_reels
            pay_window[target_row, target_col] = pay_window[
                source_row,
                source_col,
            ]
            coin_mask[target_position] = True

            # Remove the filled cell from the available pool in O(1), which
            # also prevents two split coins selecting the same destination.
            num_available -= 1
            available_positions[available_idx] = available_positions[
                num_available
            ]

        if num_available == 0:
            break

    return pay_window


@njit
def grower_bag(
    pay_window,
    coin_positions,
    locked_row_idx,
    coin_count_probabilities,
    coin_counts,
    increment_probabilities,
    increment_values,
    max_coin_value,
):
    """Increase a weighted number of random unlocked credit coins."""
    num_rows, num_reels = pay_window.shape
    num_candidates = len(coin_positions)
    if num_candidates == 0:
        return pay_window

    num_coins_to_grow = min(
        int(probChoice(coin_count_probabilities, coin_counts)),
        num_candidates,
    )
    candidate_indices = np.arange(num_candidates, dtype=np.int32)
    for idx in range(num_coins_to_grow):
        swap_idx = np.random.randint(idx, num_candidates)
        temp = candidate_indices[idx]
        candidate_indices[idx] = candidate_indices[swap_idx]
        candidate_indices[swap_idx] = temp

        source_position = int(coin_positions[candidate_indices[idx]])
        source_row = source_position // num_reels
        source_col = source_position % num_reels
        if (
            source_position < 0
            or source_position >= num_rows * num_reels
            or source_row <= locked_row_idx
            or pay_window[source_row, source_col] <= 0
        ):
            continue

        increment = int(
            probChoice(increment_probabilities, increment_values)
        )
        pay_window[source_row, source_col] = min(
            max_coin_value,
            pay_window[source_row, source_col] + increment,
        )

    return pay_window


@njit
def booster_bag(
    pay_window,
    coin_positions,
    increment_probabilities,
    increment_values,
    max_coin_value,
):
    """Apply one additive boost to every supplied credit coin."""
    num_rows, num_reels = pay_window.shape
    boost_value = int(probChoice(increment_probabilities, increment_values))
    for position in coin_positions:
        if position < 0 or position >= num_rows * num_reels:
            continue
        row = position // num_reels
        col = position % num_reels
        if pay_window[row, col] > 0:
            pay_window[row, col] = min(
                max_coin_value,
                pay_window[row, col] + boost_value,
            )

    return pay_window


@njit
def multiplier_bag(
    pay_window,
    coin_positions,
    multiplier_probabilities,
    multiplier_values,
    max_coin_value,
):
    """Apply one multiplier to every supplied credit coin."""
    num_rows, num_reels = pay_window.shape
    multiplier = int(probChoice(multiplier_probabilities, multiplier_values))
    for position in coin_positions:
        if position < 0 or position >= num_rows * num_reels:
            continue
        row = position // num_reels
        col = position % num_reels
        if pay_window[row, col] > 0:
            pay_window[row, col] = min(
                max_coin_value,
                pay_window[row, col] * multiplier,
            )

    return pay_window


@njit
def collector_bag(
    pay_window,
    coin_positions,
    collector_meter,
):
    """Add all supplied credit coin values to the collector meter."""
    num_rows, num_reels = pay_window.shape
    for position in coin_positions:
        if position < 0 or position >= num_rows * num_reels:
            continue
        row = position // num_reels
        col = position % num_reels
        if pay_window[row, col] > 0:
            collector_meter += pay_window[row, col]
    return collector_meter


@njit
def expansion_bag(
    locked_row_idx,
    rows_to_unlock,
):
    """Move the locked-row boundary upward without passing row ``-1``."""
    return max(locked_row_idx - rows_to_unlock, -1)


@njit
def collect_active_coin_positions(coin_mask, locked_row_idx, num_reels):
    """Return flattened positions of all credit coins in unlocked rows."""
    positions = np.empty(len(coin_mask), dtype=np.int32)
    count = 0
    for position in range(len(coin_mask)):
        row = position // num_reels
        if coin_mask[position] and row > locked_row_idx:
            positions[count] = position
            count += 1
    return positions[:count]


@njit
def _select_unawarded_jackpot_type(type_probabilities, awarded_jackpots):
    """Select a weighted jackpot type, excluding already-awarded types."""
    available_weight = 0.0
    for jackpot_type in range(len(type_probabilities)):
        if not awarded_jackpots[jackpot_type]:
            available_weight += type_probabilities[jackpot_type]

    if available_weight <= 0.0:
        return -1

    choice = np.random.uniform(0.0, available_weight)
    cumulative_weight = 0.0
    for jackpot_type in range(len(type_probabilities)):
        if awarded_jackpots[jackpot_type]:
            continue
        cumulative_weight += type_probabilities[jackpot_type]
        if choice < cumulative_weight:
            return jackpot_type

    return -1


@njit
def collect_free_game_jackpot_tokens(
    pay_window,
    landed_coin_positions,
    locked_row_idx,
    rules,
    jackpot_meters,
    awarded_jackpots,
):
    """Generate and collect ephemeral tokens on newly landed visible QHs."""
    num_reels = pay_window.shape[1]
    overlay_window = np.full(pay_window.shape, -1, dtype=np.int8)
    meters_before = jackpot_meters.copy()
    newly_awarded = np.zeros(len(jackpot_meters), dtype=np.bool_)

    eligible_positions = np.empty(len(landed_coin_positions), dtype=np.int32)
    eligible_count = 0
    for position in landed_coin_positions:
        if position // num_reels > locked_row_idx:
            eligible_positions[eligible_count] = position
            eligible_count += 1

    token_count = 0
    for candidate_index in range(eligible_count):
        swap_index = np.random.randint(candidate_index, eligible_count)
        position = eligible_positions[swap_index]
        eligible_positions[swap_index] = eligible_positions[candidate_index]
        eligible_positions[candidate_index] = position

        if token_count >= rules.max_jackpot_tokens_per_respin:
            break
        if np.random.uniform(0.0, 1.0) >= rules.jackpot_token_probability:
            continue

        jackpot_type = _select_unawarded_jackpot_type(
            rules.jackpot_type_probabilities,
            awarded_jackpots,
        )
        if jackpot_type < 0:
            break

        row = position // num_reels
        col = position % num_reels
        overlay_window[row, col] = jackpot_type
        jackpot_meters[jackpot_type] += 1
        token_count += 1

        if (
            jackpot_meters[jackpot_type]
            >= rules.jackpot_collection_targets[jackpot_type]
        ):
            jackpot_meters[jackpot_type] = (
                rules.jackpot_collection_targets[jackpot_type]
            )
            awarded_jackpots[jackpot_type] = True
            newly_awarded[jackpot_type] = True

    return overlay_window, meters_before, newly_awarded


@njit
def collect_bag_positions(pay_window, coin_mask, bag_symbols):
    """Group all bag-symbol positions by bag index in one board walk."""
    num_rows, num_reels = pay_window.shape
    max_positions = num_rows * num_reels
    positions = np.full(
        (len(bag_symbols), max_positions),
        -1,
        dtype=np.int32,
    )
    counts = np.zeros(len(bag_symbols), dtype=np.int32)

    for row in range(num_rows):
        for col in range(num_reels):
            position = row * num_reels + col
            if coin_mask[position]:
                continue
            symbol = pay_window[row, col]
            for bag_index in range(len(bag_symbols)):
                if symbol == bag_symbols[bag_index]:
                    positions[bag_index, counts[bag_index]] = position
                    counts[bag_index] += 1
                    break

    return positions, counts


@njit
def apply_bag_symbol_action(
    pay_window,
    coin_mask,
    position,
    action,
    coin_value_probabilities,
    coin_values,
):
    """Keep, clear, or convert a resolved bag symbol."""
    num_reels = pay_window.shape[1]
    row = position // num_reels
    col = position % num_reels

    if action == 1:
        pay_window[row, col] = 0
        coin_mask[position] = False
    elif action == 2:
        pay_window[row, col] = probChoice(
            coin_value_probabilities,
            coin_values,
        )
        coin_mask[position] = True


class HoldAndSpinResult(NamedTuple):
    pay_window: np.ndarray
    coin_mask: np.ndarray
    collector_meter: int
    locked_row_idx: int
    total_spins: int
    jackpot_meters: np.ndarray
    awarded_jackpots: np.ndarray


@njit
def hold_and_spin(
    starting_bag_symbols,
    rules,
    storage,
):
    """Run a Hold-and-Spin round and resolve all active Bags in order."""
    num_rows = rules.num_rows
    num_reels = rules.num_reels
    max_positions = num_rows * num_reels
    locked_row_idx = num_rows - rules.starting_rows - 1
    pay_window = np.zeros((num_rows, num_reels), dtype=np.int16)
    coin_mask = np.zeros(max_positions, dtype=np.bool_)
    initial_num_coins = int(
        probChoice(
            rules.starting_coin_count_probabilities,
            rules.starting_coin_counts,
        )
    )
    effective_num_rows = num_rows - (locked_row_idx + 1)
    offset = (locked_row_idx + 1) * num_reels
    num_initial_symbols = min(
        initial_num_coins + len(starting_bag_symbols),
        effective_num_rows * num_reels,
    )
    initial_idx = np.random.choice(
        effective_num_rows * num_reels,
        num_initial_symbols,
        replace=False,
    )
    scat_count = 0
    for idx in initial_idx:
        pos_x = (offset + idx) // num_reels
        pos_y = (offset + idx) % num_reels
        position = pos_x * num_reels + pos_y
        if scat_count < len(starting_bag_symbols):
            pay_window[pos_x, pos_y] = starting_bag_symbols[scat_count]
            scat_count += 1
        else:
            coin_val = probChoice(
                rules.coin_value_probabilities,
                rules.coin_values,
            )
            pay_window[pos_x, pos_y] = coin_val
            coin_mask[position] = True

    remaining_spins = rules.respin_reset_count
    collector_meter = 0
    collector_events_used = 0
    total_spins = 0
    jackpot_meters = np.zeros(
        len(rules.jackpot_collection_targets),
        dtype=np.int16,
    )
    awarded_jackpots = np.zeros(
        len(rules.jackpot_collection_targets),
        dtype=np.bool_,
    )
    storage.begin_session(starting_bag_symbols)

    while remaining_spins > 0:
        storage.begin_respin(remaining_spins)
        if total_spins == 0:
            storage.save_step(
                pay_window,
                coin_mask,
                -2,
                -1,
                remaining_spins,
                locked_row_idx,
                collector_meter,
            )

        # This is an event list, separate from the persistent coin mask.  It
        # is used only to decide whether this respin resets after Expansion.
        landed_coin_positions = np.empty(max_positions, dtype=np.int32)
        num_landed_coins = 0

        for row in range(num_rows):
            for col in range(num_reels):
                if pay_window[row, col] == 0:
                    coin_probability = rules.p_coin_locked
                    if row > locked_row_idx:
                        coin_probability = rules.p_coin_unlocked

                    if np.random.uniform(0.0, 1.0) < coin_probability:
                        coin_type = probChoice(
                            rules.coin_type_probabilities,
                            rules.coin_types,
                        )
                        position = row * num_reels + col
                        if coin_type == rules.coin_types[0]:
                            coin_val = probChoice(
                                rules.coin_value_probabilities,
                                rules.coin_values,
                            )
                            pay_window[row, col] = coin_val
                            coin_mask[position] = True
                            landed_coin_positions[num_landed_coins] = position
                            num_landed_coins += 1
                        else:
                            pay_window[row, col] = coin_type
                            coin_mask[position] = False

        (
            jackpot_overlay_window,
            jackpot_meters_before,
            newly_awarded_jackpots,
        ) = collect_free_game_jackpot_tokens(
            pay_window,
            landed_coin_positions[:num_landed_coins],
            locked_row_idx,
            rules,
            jackpot_meters,
            awarded_jackpots,
        )
        storage.save_jackpot_respin(
            jackpot_overlay_window,
            jackpot_meters_before,
            jackpot_meters,
            newly_awarded_jackpots,
        )

        storage.save_step(
            pay_window,
            coin_mask,
            -1,
            -1,
            remaining_spins,
            locked_row_idx,
            collector_meter,
        )

        bag_positions, bag_counts = collect_bag_positions(
            pay_window,
            coin_mask,
            rules.bag_symbols,
        )

        for order_index in range(len(rules.bag_resolution_order)):
            bag_index = int(rules.bag_resolution_order[order_index])
            action = int(rules.bag_symbol_actions[bag_index])

            # Expansion is processed bottom-up so an unlock can activate an
            # Expansion symbol stored in the newly opened row.
            if bag_index == 5:
                for feature_index in range(bag_counts[bag_index] - 1, -1, -1):
                    position = bag_positions[bag_index, feature_index]
                    row = position // num_reels
                    if row <= locked_row_idx:
                        continue
                    locked_row_idx = expansion_bag(
                        locked_row_idx,
                        rules.rows_unlocked_per_expansion,
                    )
                    apply_bag_symbol_action(
                        pay_window,
                        coin_mask,
                        position,
                        action,
                        rules.coin_value_probabilities,
                        rules.coin_values,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

            elif bag_index == 0:
                # One pre-Splitter snapshot guarantees that generated coins
                # cannot split again during this resolution step.
                active_coin_positions = collect_active_coin_positions(
                    coin_mask,
                    locked_row_idx,
                    num_reels,
                )
                for feature_index in range(bag_counts[bag_index]):
                    position = bag_positions[bag_index, feature_index]
                    if position // num_reels <= locked_row_idx:
                        continue
                    splitter_bag(
                        pay_window,
                        coin_mask,
                        active_coin_positions,
                        rules.splitter_source_count_probabilities,
                        rules.splitter_source_counts,
                        rules.splitter_copy_count_probabilities,
                        rules.splitter_copy_counts,
                        locked_row_idx,
                    )
                    apply_bag_symbol_action(
                        pay_window,
                        coin_mask,
                        position,
                        action,
                        rules.coin_value_probabilities,
                        rules.coin_values,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

            elif bag_index == 2:
                active_coin_positions = collect_active_coin_positions(
                    coin_mask,
                    locked_row_idx,
                    num_reels,
                )
                for feature_index in range(bag_counts[bag_index]):
                    position = bag_positions[bag_index, feature_index]
                    if position // num_reels <= locked_row_idx:
                        continue
                    booster_bag(
                        pay_window,
                        active_coin_positions,
                        rules.booster_increment_probabilities,
                        rules.booster_increment_values,
                        rules.max_coin_value,
                    )
                    apply_bag_symbol_action(
                        pay_window,
                        coin_mask,
                        position,
                        action,
                        rules.coin_value_probabilities,
                        rules.coin_values,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

            elif bag_index == 1:
                active_coin_positions = collect_active_coin_positions(
                    coin_mask,
                    locked_row_idx,
                    num_reels,
                )
                for feature_index in range(bag_counts[bag_index]):
                    position = bag_positions[bag_index, feature_index]
                    if position // num_reels <= locked_row_idx:
                        continue
                    grower_bag(
                        pay_window,
                        active_coin_positions,
                        locked_row_idx,
                        rules.grower_coin_count_probabilities,
                        rules.grower_coin_counts,
                        rules.grower_increment_probabilities,
                        rules.grower_increment_values,
                        rules.max_coin_value,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

            elif bag_index == 3:
                active_coin_positions = collect_active_coin_positions(
                    coin_mask,
                    locked_row_idx,
                    num_reels,
                )
                for feature_index in range(bag_counts[bag_index]):
                    position = bag_positions[bag_index, feature_index]
                    if position // num_reels <= locked_row_idx:
                        continue
                    multiplier_bag(
                        pay_window,
                        active_coin_positions,
                        rules.multiplier_probabilities,
                        rules.multiplier_values,
                        rules.max_coin_value,
                    )
                    apply_bag_symbol_action(
                        pay_window,
                        coin_mask,
                        position,
                        action,
                        rules.coin_value_probabilities,
                        rules.coin_values,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

            elif bag_index == 4:
                active_coin_positions = collect_active_coin_positions(
                    coin_mask,
                    locked_row_idx,
                    num_reels,
                )
                for feature_index in range(bag_counts[bag_index]):
                    position = bag_positions[bag_index, feature_index]
                    if position // num_reels <= locked_row_idx:
                        continue
                    if collector_events_used < rules.max_collector_events:
                        collector_meter = collector_bag(
                            pay_window,
                            active_coin_positions,
                            collector_meter,
                        )
                        collector_events_used += 1
                    apply_bag_symbol_action(
                        pay_window,
                        coin_mask,
                        position,
                        action,
                        rules.coin_value_probabilities,
                        rules.coin_values,
                    )
                    storage.save_step(
                        pay_window,
                        coin_mask,
                        bag_index,
                        position,
                        remaining_spins,
                        locked_row_idx,
                        collector_meter,
                    )

        natural_coin_landed = False
        for idx in range(num_landed_coins):
            if landed_coin_positions[idx] // num_reels > locked_row_idx:
                natural_coin_landed = True
                break

        if natural_coin_landed:
            remaining_spins = rules.respin_reset_count
        else:
            remaining_spins -= 1
        total_spins += 1
        storage.finish_respin(remaining_spins, natural_coin_landed)

        has_empty_unlocked_position = False
        for row in range(locked_row_idx + 1, num_rows):
            for col in range(num_reels):
                if pay_window[row, col] == 0:
                    has_empty_unlocked_position = True
                    break
            if has_empty_unlocked_position:
                break

        if not has_empty_unlocked_position:
            break

    coin_win = 0
    flattened_window = pay_window.ravel()
    for position in range(len(coin_mask)):
        if coin_mask[position]:
            coin_win += flattened_window[position]
    storage.finish_session(
        float(coin_win + collector_meter),
        float(coin_win),
        float(collector_meter),
        jackpot_meters,
        awarded_jackpots,
    )

    return HoldAndSpinResult(
        pay_window,
        coin_mask,
        collector_meter,
        locked_row_idx,
        total_spins,
        jackpot_meters,
        awarded_jackpots,
    )


# Backwards-compatible alias for the original public spelling.
HoldAndSpin = hold_and_spin
