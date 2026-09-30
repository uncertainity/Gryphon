"""Numba kernels for the seven configured Gryphon feature routes."""

import numpy as np
from numba import njit

from .kernels import probChoice


EMPTY_SYMBOL = -1


@njit
def _coin_value(rules):
    return float(probChoice(rules.coin_value_probabilities, rules.coin_values))


@njit
def _jackpot_index(symbol, jackpot_symbols):
    for index in range(len(jackpot_symbols)):
        if symbol == jackpot_symbols[index]:
            return index
    return -1


@njit
def _collect_jackpot(symbol, meters, awards, rules):
    index = _jackpot_index(symbol, rules.jackpot_symbols)
    if index < 0:
        return 0.0
    meters[index] += 1
    if meters[index] >= rules.jackpot_collection_targets[index]:
        meters[index] = 0
        awards[index] += 1
        return float(rules.jackpot_awards[index])
    return 0.0


@njit
def _grid_is_full(board):
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] == EMPTY_SYMBOL:
                return False
    return True


@njit
def _empty_positions(board, row_min, row_max):
    positions = np.empty(board.size, dtype=np.int32)
    count = 0
    for row in range(row_min, row_max):
        for col in range(board.shape[1]):
            if board[row, col] == EMPTY_SYMBOL:
                positions[count] = row * board.shape[1] + col
                count += 1
    for index in range(count - 1, 0, -1):
        swap = np.random.randint(0, index + 1)
        value = positions[index]
        positions[index] = positions[swap]
        positions[swap] = value
    return positions, count


@njit
def _seed_start(board, values, rules, trigger_has_value, row_min):
    row = board.shape[0] - 1
    col = board.shape[1] // 2
    board[row, col] = rules.trigger_symbol
    if trigger_has_value:
        values[row, col] = _coin_value(rules)
    positions, available = _empty_positions(board, row_min, board.shape[0])
    count = min(rules.starting_extra_coin_count, available)
    for index in range(count):
        position = positions[index]
        row = position // board.shape[1]
        col = position % board.shape[1]
        board[row, col] = rules.coin_symbol
        values[row, col] = _coin_value(rules)


@njit
def _normal_coin_positions(board, coin_symbol, row_min=0):
    positions = np.empty(board.size, dtype=np.int32)
    count = 0
    for row in range(row_min, board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] == coin_symbol:
                positions[count] = row * board.shape[1] + col
                count += 1
    for index in range(count - 1, 0, -1):
        swap = np.random.randint(0, index + 1)
        value = positions[index]
        positions[index] = positions[swap]
        positions[swap] = value
    return positions, count


@njit
def run_expansion_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    locked_until = rules.num_rows - rules.starting_unlocked_rows - 1
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    go_landed = 1
    locked_go_landed = 0
    _seed_start(board, values, rules, True, locked_until + 1)

    while spins_left > 0:
        total_spins += 1
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if expire[row, col]:
                    board[row, col] = EMPTY_SYMBOL
                    values[row, col] = 0.0
                    expire[row, col] = False
        landed = False
        for row in range(rules.num_rows):
            is_locked = row <= locked_until
            symbols = rules.locked_landing_symbols if is_locked else rules.unlocked_landing_symbols
            probabilities = rules.locked_landing_probabilities if is_locked else rules.unlocked_landing_probabilities
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(probabilities, symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.go_symbol:
                    if go_landed >= rules.max_go_symbols:
                        continue
                    if is_locked and locked_go_landed >= rules.max_locked_go_symbols:
                        continue
                    go_landed += 1
                    if is_locked:
                        locked_go_landed += 1
                board[row, col] = symbol
                if symbol == rules.coin_symbol:
                    values[row, col] = _coin_value(rules)
                elif not is_locked and _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _collect_jackpot(symbol, meters, awards, rules)
                    expire[row, col] = True
                if not is_locked:
                    landed = True
        unlock_again = True
        while unlock_again:
            unlock_again = False
            for row in range(max(0, locked_until + 1), rules.num_rows):
                for col in range(rules.num_reels):
                    symbol = board[row, col]
                    if symbol == rules.go_symbol and not expire[row, col]:
                        if locked_until >= 0:
                            locked_until -= 1
                        expire[row, col] = True
                        unlock_again = True
                    elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0 and not expire[row, col]:
                        jackpot_win += _collect_jackpot(symbol, meters, awards, rules)
                        expire[row, col] = True
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def _place_multipliers(cells, rules, row_min=0):
    if np.random.random() >= rules.multiplier_trigger_probability:
        return
    count = int(probChoice(rules.multiplier_count_probabilities, rules.multiplier_counts))
    positions = np.empty(cells.size, np.int32)
    available = 0
    for row in range(row_min, cells.shape[0]):
        for col in range(cells.shape[1]):
            if cells[row, col] <= 1:
                positions[available] = row * cells.shape[1] + col
                available += 1
    for index in range(available - 1, 0, -1):
        swap = np.random.randint(0, index + 1)
        positions[index], positions[swap] = positions[swap], positions[index]
    for index in range(min(count, available)):
        position = positions[index]
        cells[position // cells.shape[1], position % cells.shape[1]] = int(
            probChoice(rules.multiplier_value_probabilities, rules.multiplier_values)
        )


@njit
def run_multiplier_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    expire = np.zeros(board.shape, np.bool_)
    cells = np.ones(board.shape, np.int16)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    _seed_start(board, values, rules, True, 0)
    while spins_left > 0:
        total_spins += 1
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if expire[row, col]:
                    board[row, col] = EMPTY_SYMBOL
                    values[row, col] = 0.0
                    expire[row, col] = False
        _place_multipliers(cells, rules)
        landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(rules.landing_probabilities, rules.landing_symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                board[row, col] = symbol
                if symbol == rules.coin_symbol:
                    value = _coin_value(rules)
                    if cells[row, col] > 1:
                        value *= cells[row, col]
                        cells[row, col] = 1
                    values[row, col] = value
                    landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _collect_jackpot(symbol, meters, awards, rules)
                    expire[row, col] = True
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def _expire_cells(board, values, expire):
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if expire[row, col]:
                board[row, col] = EMPTY_SYMBOL
                values[row, col] = 0.0
                expire[row, col] = False


@njit
def _apply_jackpot(board, expire, row, col, symbol, rules, meters, awards):
    board[row, col] = symbol
    expire[row, col] = True
    return _collect_jackpot(symbol, meters, awards, rules)


@njit
def run_grow_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    grower_landed = 1
    _seed_start(board, values, rules, True, 0)
    while spins_left > 0:
        total_spins += 1
        _expire_cells(board, values, expire)
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != rules.grower_symbol or np.random.random() >= rules.grow_trigger_probability:
                    continue
                positions, count = _normal_coin_positions(board, rules.coin_symbol)
                target_count = int(probChoice(rules.grow_coin_count_probabilities, rules.grow_coin_counts))
                amount = float(probChoice(rules.grow_value_probabilities, rules.grow_values))
                used = 0
                for index in range(count):
                    position = positions[index]
                    target_row = position // rules.num_reels
                    target_col = position % rules.num_reels
                    if target_row == row and target_col == col:
                        continue
                    values[target_row, target_col] += amount
                    used += 1
                    if used == target_count:
                        break
        landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(rules.landing_probabilities, rules.landing_symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.grower_symbol:
                    if grower_landed >= rules.max_grower_symbols:
                        continue
                    grower_landed += 1
                    board[row, col] = symbol
                    values[row, col] = _coin_value(rules)
                    landed = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    values[row, col] = _coin_value(rules)
                    landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot(board, expire, row, col, symbol, rules, meters, awards)
                    landed = True
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def run_boost_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    booster_active = True
    booster_row = rules.num_rows - 1
    booster_col = rules.num_reels // 2
    _seed_start(board, values, rules, False, 0)
    while spins_left > 0:
        total_spins += 1
        _expire_cells(board, values, expire)
        if booster_active and np.random.random() < rules.boost_trigger_probability:
            amount = float(probChoice(rules.boost_value_probabilities, rules.boost_values))
            for row in range(rules.num_rows):
                for col in range(rules.num_reels):
                    if board[row, col] == rules.coin_symbol and not (row == booster_row and col == booster_col):
                        values[row, col] += amount
            board[booster_row, booster_col] = rules.coin_symbol
            values[booster_row, booster_col] = amount
            booster_active = False
        landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(rules.landing_probabilities, rules.landing_symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    values[row, col] = _coin_value(rules)
                    landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot(board, expire, row, col, symbol, rules, meters, awards)
                    landed = True
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def _collect_at(board, values, row, col, coin_symbol, row_min=0):
    total = 0.0
    for coin_row in range(row_min, board.shape[0]):
        for coin_col in range(board.shape[1]):
            if board[coin_row, coin_col] == coin_symbol and not (coin_row == row and coin_col == col):
                total += values[coin_row, coin_col]
    board[row, col] = coin_symbol
    values[row, col] = total
    return total


@njit
def run_collect_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    collector_landed = 1
    _seed_start(board, values, rules, False, 0)
    _collect_at(board, values, rules.num_rows - 1, rules.num_reels // 2, rules.coin_symbol)
    while spins_left > 0:
        total_spins += 1
        _expire_cells(board, values, expire)
        landed_collectors = np.zeros(board.shape, np.bool_)
        landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(rules.landing_probabilities, rules.landing_symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.collector_symbol:
                    if collector_landed >= rules.max_collector_symbols:
                        continue
                    collector_landed += 1
                    board[row, col] = symbol
                    landed_collectors[row, col] = True
                    landed = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    values[row, col] = _coin_value(rules)
                    landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot(board, expire, row, col, symbol, rules, meters, awards)
                    landed = True
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if landed_collectors[row, col]:
                    _collect_at(board, values, row, col, rules.coin_symbol)
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def _place_splitter(board, values, counts, row, col, rules):
    count = int(probChoice(rules.splitter_coin_count_probabilities, rules.splitter_coin_counts))
    value = 0.0
    for _ in range(count):
        value += float(probChoice(rules.splitter_coin_value_probabilities, rules.splitter_coin_values))
    board[row, col] = rules.splitter_symbol
    values[row, col] = value
    counts[row, col] = count


@njit
def run_splitter_kernel(rules, payout_multiplier):
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    counts = np.zeros(board.shape, np.int16)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    splitter_landed = 1
    _seed_start(board, values, rules, False, 0)
    _place_splitter(board, values, counts, rules.num_rows - 1, rules.num_reels // 2, rules)
    while spins_left > 0:
        total_spins += 1
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if expire[row, col]:
                    board[row, col] = EMPTY_SYMBOL
                    values[row, col] = 0.0
                    counts[row, col] = 0
                    expire[row, col] = False
        landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                empty_count = np.count_nonzero(board == EMPTY_SYMBOL)
                if splitter_landed < rules.min_splitter_symbols and empty_count < 4:
                    symbol = rules.splitter_symbol
                else:
                    symbol = int(probChoice(rules.landing_probabilities, rules.landing_symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.splitter_symbol:
                    splitter_landed += 1
                    _place_splitter(board, values, counts, row, col, rules)
                    landed = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    values[row, col] = _coin_value(rules)
                    counts[row, col] = 1
                    landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot(board, expire, row, col, symbol, rules, meters, awards)
                    landed = True
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win


@njit
def run_mega_combo_kernel(base_window, rules, payout_multiplier):
    """Compact combined kernel preserving positions and all six powers."""
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, np.int16)
    values = np.zeros(board.shape, np.float64)
    counts = np.zeros(board.shape, np.int16)
    cells = np.ones(board.shape, np.int16)
    expire = np.zeros(board.shape, np.bool_)
    meters = np.zeros(len(rules.jackpot_symbols), np.int16)
    awards = np.zeros(len(rules.jackpot_symbols), np.int16)
    locked_until = rules.num_rows - rules.starting_unlocked_rows - 1
    active_start = locked_until + 1
    special_counts = np.zeros(5, np.int16)  # go, locked-go, grower, collector, splitter
    found = np.zeros(len(rules.trigger_symbols), np.bool_)
    for base_row in range(base_window.shape[0]):
        for col in range(base_window.shape[1]):
            symbol = base_window[base_row, col]
            for index in range(len(rules.trigger_symbols)):
                if symbol != rules.trigger_symbols[index]:
                    continue
                found[index] = True
                row = active_start + base_row
                board[row, col] = symbol
                if symbol != rules.collector_symbol:
                    values[row, col] = _coin_value(rules)
                if symbol == rules.go_symbol:
                    special_counts[0] += 1
                elif symbol == rules.grower_symbol:
                    special_counts[2] += 1
                elif symbol == rules.collector_symbol:
                    special_counts[3] += 1
                elif symbol == rules.splitter_symbol:
                    special_counts[4] += 1
    for index in range(len(found)):
        if not found[index]:
            raise ValueError("Mega combo requires SC1-SC6")
    extra = int(probChoice(rules.additional_coin_count_probabilities, rules.additional_coin_counts))
    positions, available = _empty_positions(board, active_start, rules.num_rows)
    for index in range(min(extra, available)):
        position = positions[index]
        row, col = position // rules.num_reels, position % rules.num_reels
        board[row, col] = rules.coin_symbol
        values[row, col] = _coin_value(rules)
        counts[row, col] = 1

    # Guaranteed opening actions.
    for row in range(active_start, rules.num_rows):
        for col in range(rules.num_reels):
            symbol = board[row, col]
            if symbol == rules.splitter_symbol:
                source_positions, source_count = _normal_coin_positions(board, rules.coin_symbol, active_start)
                for source_index in range(source_count):
                    empties, empty_count = _empty_positions(board, active_start, rules.num_rows)
                    if empty_count == 0:
                        break
                    source = source_positions[source_index]
                    target = empties[0]
                    target_row, target_col = target // rules.num_reels, target % rules.num_reels
                    source_row, source_col = source // rules.num_reels, source % rules.num_reels
                    board[target_row, target_col] = rules.coin_symbol
                    values[target_row, target_col] = values[source_row, source_col]
                    counts[target_row, target_col] = max(1, counts[source_row, source_col])
                board[row, col] = rules.coin_symbol
                counts[row, col] = 1
            elif symbol == rules.grower_symbol:
                coin_positions, coin_count = _normal_coin_positions(board, rules.coin_symbol, active_start)
                target_count = int(probChoice(rules.grow_coin_count_probabilities, rules.grow_coin_counts))
                amount = float(probChoice(rules.grow_value_probabilities, rules.grow_values))
                for index in range(min(target_count, coin_count)):
                    target = coin_positions[index]
                    values[target // rules.num_reels, target % rules.num_reels] += amount
            elif symbol == rules.booster_symbol:
                amount = float(probChoice(rules.boost_value_probabilities, rules.boost_values))
                for coin_row in range(active_start, rules.num_rows):
                    for coin_col in range(rules.num_reels):
                        if board[coin_row, coin_col] == rules.coin_symbol:
                            values[coin_row, coin_col] += amount
                board[row, col] = rules.coin_symbol
                values[row, col] = amount
                counts[row, col] = 1
            elif symbol == rules.multiplier_symbol:
                _place_multipliers(cells, rules, active_start)
                board[row, col] = rules.coin_symbol
                counts[row, col] = 1
            elif symbol == rules.collector_symbol:
                _collect_at(board, values, row, col, rules.coin_symbol, active_start)
                counts[row, col] = 1
            elif symbol == rules.go_symbol:
                board[row, col] = rules.coin_symbol
                counts[row, col] = 1

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    while spins_left > 0:
        total_spins += 1
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if expire[row, col]:
                    board[row, col] = EMPTY_SYMBOL
                    values[row, col] = 0.0
                    counts[row, col] = 0
                    expire[row, col] = False
        if np.random.random() < rules.multiplier_trigger_probability:
            _place_multipliers(cells, rules, max(0, locked_until + 1))
        if np.random.random() < rules.boost_trigger_probability:
            amount = float(probChoice(rules.boost_value_probabilities, rules.boost_values))
            for row in range(max(0, locked_until + 1), rules.num_rows):
                for col in range(rules.num_reels):
                    if board[row, col] == rules.coin_symbol:
                        values[row, col] += amount
        landed = False
        pending_collectors = np.zeros(board.shape, np.bool_)
        for row in range(rules.num_rows):
            is_locked = row <= locked_until
            symbols = rules.locked_landing_symbols if is_locked else rules.unlocked_landing_symbols
            probabilities = rules.locked_landing_probabilities if is_locked else rules.unlocked_landing_probabilities
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(probChoice(probabilities, symbols))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.go_symbol:
                    if special_counts[0] >= rules.max_go_symbols or (is_locked and special_counts[1] >= rules.max_locked_go_symbols):
                        continue
                    special_counts[0] += 1
                    if is_locked:
                        special_counts[1] += 1
                elif symbol == rules.grower_symbol:
                    if special_counts[2] >= rules.max_grower_symbols:
                        continue
                    special_counts[2] += 1
                elif symbol == rules.collector_symbol:
                    if special_counts[3] >= rules.max_collector_symbols:
                        continue
                    special_counts[3] += 1
                elif symbol == rules.splitter_symbol:
                    if special_counts[4] >= rules.max_splitter_symbols:
                        continue
                    special_counts[4] += 1
                board[row, col] = symbol
                if symbol == rules.coin_symbol:
                    value = _coin_value(rules)
                    if not is_locked and cells[row, col] > 1:
                        value *= cells[row, col]
                        cells[row, col] = 1
                    values[row, col] = value
                    counts[row, col] = 1
                elif not is_locked and _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _collect_jackpot(symbol, meters, awards, rules)
                    expire[row, col] = True
                elif not is_locked and symbol == rules.collector_symbol:
                    pending_collectors[row, col] = True
                elif not is_locked and symbol == rules.splitter_symbol:
                    _place_splitter(board, values, counts, row, col, rules)
                elif not is_locked and symbol == rules.go_symbol:
                    if locked_until >= 0:
                        locked_until -= 1
                    expire[row, col] = True
                elif not is_locked and symbol == rules.grower_symbol:
                    values[row, col] = _coin_value(rules)
                if not is_locked:
                    landed = True
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if pending_collectors[row, col]:
                    _collect_at(board, values, row, col, rules.coin_symbol, max(0, locked_until + 1))
                    counts[row, col] = 1
        spins_left = rules.respin_reset_count if landed else spins_left - 1
        if _grid_is_full(board):
            break
    return (np.sum(values) + jackpot_win) * payout_multiplier, total_spins, awards, jackpot_win
