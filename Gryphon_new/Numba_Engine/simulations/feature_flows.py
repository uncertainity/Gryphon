from dataclasses import dataclass

import numpy as np

from ..core.config import (
    BOOST_FEATURE_CONFIG,
    COLLECT_FEATURE_CONFIG,
    EXPANSION_FEATURE_CONFIG,
    FEATURE_RTP_CONFIG,
    GROW_FEATURE_CONFIG,
    MEGA_COMBO_FEATURE_CONFIG,
    MULTIPLIER_FEATURE_CONFIG,
    SPLITTER_FEATURE_CONFIG,
)


EMPTY_SYMBOL = -1


@dataclass
class FeatureResult:
    board: np.ndarray
    coin_values: np.ndarray
    total_win: float
    total_spins: int
    jackpot_meters: np.ndarray
    awarded_jackpots: np.ndarray
    jackpot_win: float
    unlocked_rows: int
    go_landed: int = 0
    locked_go_landed: int = 0
    multiplier_cells: np.ndarray | None = None
    special_landed: int = 0
    collected_value: float = 0.0
    splitter_coin_counts: np.ndarray | None = None
    special_counts: dict | None = None


def _choice(values, probabilities, rng):
    return values[int(rng.choice(len(values), p=probabilities))]


def _random_empty_positions(board, rng, count, row_min=0, row_max=None):
    if row_max is None:
        row_max = board.shape[0]
    positions = [
        (row, col)
        for row in range(row_min, row_max)
        for col in range(board.shape[1])
        if board[row, col] == EMPTY_SYMBOL
    ]
    rng.shuffle(positions)
    return positions[: min(count, len(positions))]


def _coin_value(rules, rng):
    return float(_choice(rules.coin_values, rules.coin_value_probabilities, rng))


def _feature_total(coin_win, jackpot_win):
    return (coin_win + jackpot_win) * FEATURE_RTP_CONFIG.feature_payout_multiplier


def _jackpot_index(symbol, jackpot_symbols):
    matches = np.where(jackpot_symbols == symbol)[0]
    if len(matches) == 0:
        return -1
    return int(matches[0])


def _collect_jackpot(symbol, jackpot_meters, awarded_jackpots, rules):
    jackpot_index = _jackpot_index(symbol, rules.jackpot_symbols)
    if jackpot_index < 0:
        return 0.0

    jackpot_meters[jackpot_index] += 1
    if jackpot_meters[jackpot_index] >= rules.jackpot_collection_targets[jackpot_index]:
        jackpot_meters[jackpot_index] = 0
        awarded_jackpots[jackpot_index] += 1
        return float(rules.jackpot_awards[jackpot_index])
    return 0.0


def _grid_is_full(board):
    return not np.any(board == EMPTY_SYMBOL)


def _normal_coin_positions(board, rules):
    return [
        (row, col)
        for row in range(board.shape[0])
        for col in range(board.shape[1])
        if board[row, col] == rules.coin_symbol
    ]


def _choose_positions_without_replacement(positions, rng, count):
    shuffled = list(positions)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def _seed_common_start(board, coin_values, trigger_symbol, rules, rng):
    start_row = board.shape[0] - 1
    start_col = board.shape[1] // 2
    board[start_row, start_col] = trigger_symbol
    if trigger_symbol != getattr(rules, "collector_symbol", None):
        coin_values[start_row, start_col] = _coin_value(rules, rng)

    for row, col in _random_empty_positions(
        board,
        rng,
        rules.starting_extra_coin_count,
        row_min=max(0, board.shape[0] - getattr(rules, "starting_unlocked_rows", board.shape[0])),
    ):
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = _coin_value(rules, rng)


def run_expansion_feature(rules=EXPANSION_FEATURE_CONFIG, seed=None):
    """Run the requested Expansion feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)

    locked_until = rules.num_rows - rules.starting_unlocked_rows - 1
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    go_landed = 1
    locked_go_landed = 0

    _seed_common_start(board, coin_values, rules.trigger_symbol, rules, rng)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        expire_next[:, :] = False

        new_unlocked_symbol = False

        for row in range(rules.num_rows):
            is_locked = row <= locked_until
            symbols = rules.locked_landing_symbols if is_locked else rules.unlocked_landing_symbols
            probabilities = (
                rules.locked_landing_probabilities
                if is_locked
                else rules.unlocked_landing_probabilities
            )
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(_choice(symbols, probabilities, rng))
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
                    coin_values[row, col] = _coin_value(rules, rng)
                elif not is_locked and _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _collect_jackpot(
                        symbol,
                        jackpot_meters,
                        awarded_jackpots,
                        rules,
                    )
                    expire_next[row, col] = True
                if not is_locked:
                    new_unlocked_symbol = True

        unlock_again = True
        while unlock_again:
            unlock_again = False
            for row in range(locked_until + 1, rules.num_rows):
                for col in range(rules.num_reels):
                    symbol = int(board[row, col])
                    if symbol == rules.go_symbol and not expire_next[row, col]:
                        if locked_until >= 0:
                            locked_until -= 1
                        expire_next[row, col] = True
                        unlock_again = True
                    elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0 and not expire_next[row, col]:
                        jackpot_win += _collect_jackpot(
                            symbol,
                            jackpot_meters,
                            awarded_jackpots,
                            rules,
                        )
                        expire_next[row, col] = True

        if new_unlocked_symbol:
            spins_left = rules.respin_reset_count
        else:
            spins_left -= 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows - locked_until - 1,
        go_landed=go_landed,
        locked_go_landed=locked_go_landed,
    )


def _place_multipliers(multiplier_cells, rules, rng):
    if rng.random() >= rules.multiplier_trigger_probability:
        return
    count = int(_choice(rules.multiplier_counts, rules.multiplier_count_probabilities, rng))
    empty_positions = [
        (row, col)
        for row in range(multiplier_cells.shape[0])
        for col in range(multiplier_cells.shape[1])
        if multiplier_cells[row, col] <= 1
    ]
    rng.shuffle(empty_positions)
    for row, col in empty_positions[: min(count, len(empty_positions))]:
        multiplier_cells[row, col] = int(
            _choice(
                rules.multiplier_values,
                rules.multiplier_value_probabilities,
                rng,
            )
        )


def run_multiplier_feature(rules=MULTIPLIER_FEATURE_CONFIG, seed=None):
    """Run the requested Multiplier feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    multiplier_cells = np.ones((rules.num_rows, rules.num_reels), dtype=np.int16)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0

    _seed_common_start(board, coin_values, rules.trigger_symbol, rules, rng)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        expire_next[:, :] = False

        _place_multipliers(multiplier_cells, rules, rng)

        new_coin_landed = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(
                    _choice(
                        rules.landing_symbols,
                        rules.landing_probabilities,
                        rng,
                    )
                )
                if symbol == EMPTY_SYMBOL:
                    continue

                board[row, col] = symbol
                if symbol == rules.coin_symbol:
                    value = _coin_value(rules, rng)
                    if multiplier_cells[row, col] > 1:
                        value *= float(multiplier_cells[row, col])
                        multiplier_cells[row, col] = 1
                    coin_values[row, col] = value
                    new_coin_landed = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _collect_jackpot(
                        symbol,
                        jackpot_meters,
                        awarded_jackpots,
                        rules,
                    )
                    expire_next[row, col] = True

        if new_coin_landed:
            spins_left = rules.respin_reset_count
        else:
            spins_left -= 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows,
        multiplier_cells=multiplier_cells,
    )


def _seed_plain_3x5_start(board, coin_values, rules, rng, trigger_has_coin_value=True):
    start_row = board.shape[0] - 1
    start_col = board.shape[1] // 2
    board[start_row, start_col] = rules.trigger_symbol
    if trigger_has_coin_value:
        coin_values[start_row, start_col] = _coin_value(rules, rng)
    for row, col in _random_empty_positions(
        board,
        rng,
        rules.starting_extra_coin_count,
    ):
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = _coin_value(rules, rng)


def _apply_jackpot_landing(board, expire_next, row, col, symbol, rules, meters, awarded):
    board[row, col] = symbol
    expire_next[row, col] = True
    return _collect_jackpot(symbol, meters, awarded, rules)


def run_grow_feature(rules=GROW_FEATURE_CONFIG, seed=None):
    """Run the requested Grow feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    grower_positions = []

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    grower_landed = 1

    _seed_plain_3x5_start(board, coin_values, rules, rng)
    grower_positions.append((rules.num_rows - 1, rules.num_reels // 2))

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        expire_next[:, :] = False

        for grower_row, grower_col in list(grower_positions):
            if board[grower_row, grower_col] != rules.grower_symbol:
                continue
            if rng.random() >= rules.grow_trigger_probability:
                continue
            normal_coins = [
                position
                for position in _normal_coin_positions(board, rules)
                if position != (grower_row, grower_col)
            ]
            count = int(_choice(rules.grow_coin_counts, rules.grow_coin_count_probabilities, rng))
            grow_value = float(_choice(rules.grow_values, rules.grow_value_probabilities, rng))
            for row, col in _choose_positions_without_replacement(normal_coins, rng, count):
                coin_values[row, col] += grow_value

        landed_any = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(_choice(rules.landing_symbols, rules.landing_probabilities, rng))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.grower_symbol:
                    if grower_landed >= rules.max_grower_symbols:
                        continue
                    grower_landed += 1
                    grower_positions.append((row, col))
                    board[row, col] = symbol
                    coin_values[row, col] = _coin_value(rules, rng)
                    landed_any = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    coin_values[row, col] = _coin_value(rules, rng)
                    landed_any = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot_landing(
                        board,
                        expire_next,
                        row,
                        col,
                        symbol,
                        rules,
                        jackpot_meters,
                        awarded_jackpots,
                    )
                    landed_any = True

        spins_left = rules.respin_reset_count if landed_any else spins_left - 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows,
        special_landed=grower_landed,
    )


def run_boost_feature(rules=BOOST_FEATURE_CONFIG, seed=None):
    """Run the requested Boost feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    booster_position = (rules.num_rows - 1, rules.num_reels // 2)
    booster_active = True

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0

    _seed_plain_3x5_start(board, coin_values, rules, rng, trigger_has_coin_value=False)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        expire_next[:, :] = False

        if booster_active and rng.random() < rules.boost_trigger_probability:
            boost_value = float(_choice(rules.boost_values, rules.boost_value_probabilities, rng))
            for row, col in _normal_coin_positions(board, rules):
                if (row, col) != booster_position:
                    coin_values[row, col] += boost_value
            row, col = booster_position
            board[row, col] = rules.coin_symbol
            coin_values[row, col] = boost_value
            booster_active = False

        landed_any = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(_choice(rules.landing_symbols, rules.landing_probabilities, rng))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    coin_values[row, col] = _coin_value(rules, rng)
                    landed_any = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot_landing(
                        board,
                        expire_next,
                        row,
                        col,
                        symbol,
                        rules,
                        jackpot_meters,
                        awarded_jackpots,
                    )
                    landed_any = True

        spins_left = rules.respin_reset_count if landed_any else spins_left - 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows,
        special_landed=0 if booster_active else 1,
    )


def run_collect_feature(rules=COLLECT_FEATURE_CONFIG, seed=None):
    """Run the requested Collect feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    collected_value = 0.0
    collector_landed = 1

    _seed_plain_3x5_start(board, coin_values, rules, rng, trigger_has_coin_value=False)

    def collect_at(row, col):
        total = 0.0
        for coin_row, coin_col in _normal_coin_positions(board, rules):
            if (coin_row, coin_col) != (row, col):
                total += coin_values[coin_row, coin_col]
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = total
        return total

    collected_value += collect_at(rules.num_rows - 1, rules.num_reels // 2)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        expire_next[:, :] = False

        landed_any = False
        landed_collectors = []
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                symbol = int(_choice(rules.landing_symbols, rules.landing_probabilities, rng))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.collector_symbol:
                    if collector_landed >= rules.max_collector_symbols:
                        continue
                    collector_landed += 1
                    board[row, col] = symbol
                    landed_collectors.append((row, col))
                    landed_any = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    coin_values[row, col] = _coin_value(rules, rng)
                    landed_any = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot_landing(
                        board,
                        expire_next,
                        row,
                        col,
                        symbol,
                        rules,
                        jackpot_meters,
                        awarded_jackpots,
                    )
                    landed_any = True

        for row, col in landed_collectors:
            collected_value += collect_at(row, col)

        spins_left = rules.respin_reset_count if landed_any else spins_left - 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows,
        special_landed=collector_landed,
        collected_value=collected_value,
    )


def run_splitter_feature(rules=SPLITTER_FEATURE_CONFIG, seed=None):
    """Run the requested Splitter feature flow."""
    rng = np.random.default_rng(seed)
    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    splitter_coin_counts = np.zeros((rules.num_rows, rules.num_reels), dtype=np.int16)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)

    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    splitter_landed = 1

    _seed_plain_3x5_start(board, coin_values, rules, rng, trigger_has_coin_value=False)

    def place_splitter(row, col):
        count = int(_choice(rules.splitter_coin_counts, rules.splitter_coin_count_probabilities, rng))
        value = 0.0
        for _ in range(count):
            value += float(
                _choice(
                    rules.splitter_coin_values,
                    rules.splitter_coin_value_probabilities,
                    rng,
                )
            )
        board[row, col] = rules.splitter_symbol
        coin_values[row, col] = value
        splitter_coin_counts[row, col] = count

    place_splitter(rules.num_rows - 1, rules.num_reels // 2)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        splitter_coin_counts[expire_next] = 0
        expire_next[:, :] = False

        landed_any = False
        for row in range(rules.num_rows):
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                empty_count = int(np.count_nonzero(board == EMPTY_SYMBOL))
                force_splitter = (
                    splitter_landed < rules.min_splitter_symbols
                    and empty_count < 4
                )
                if force_splitter:
                    symbol = rules.splitter_symbol
                else:
                    symbol = int(_choice(rules.landing_symbols, rules.landing_probabilities, rng))
                if symbol == EMPTY_SYMBOL:
                    continue
                if symbol == rules.splitter_symbol:
                    splitter_landed += 1
                    place_splitter(row, col)
                    landed_any = True
                elif symbol == rules.coin_symbol:
                    board[row, col] = symbol
                    coin_values[row, col] = _coin_value(rules, rng)
                    splitter_coin_counts[row, col] = 1
                    landed_any = True
                elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    jackpot_win += _apply_jackpot_landing(
                        board,
                        expire_next,
                        row,
                        col,
                        symbol,
                        rules,
                        jackpot_meters,
                        awarded_jackpots,
                    )
                    landed_any = True

        spins_left = rules.respin_reset_count if landed_any else spins_left - 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows,
        special_landed=splitter_landed,
        splitter_coin_counts=splitter_coin_counts,
    )


def _default_combo_base_window(rules):
    base = np.full((3, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    positions = [(0, 0), (0, 2), (1, 1), (1, 3), (2, 2), (2, 4)]
    for symbol, (row, col) in zip(rules.trigger_symbols, positions):
        base[row, col] = symbol
    return base


def _combo_trigger_positions(base_window, rules):
    positions = []
    missing = []
    for symbol in rules.trigger_symbols:
        found = np.argwhere(base_window == symbol)
        if len(found) == 0:
            missing.append(int(symbol))
            continue
        for row, col in found:
            positions.append((int(symbol), int(row), int(col)))
    if missing:
        raise ValueError(
            "Mega combo requires all SC1-SC6 symbols on the base window"
        )
    return positions


def _place_combo_multiplier_cells(multiplier_cells, rules, rng):
    active_start = rules.num_rows - rules.starting_unlocked_rows
    count = int(_choice(rules.multiplier_counts, rules.multiplier_count_probabilities, rng))
    positions = [
        (row, col)
        for row in range(active_start, rules.num_rows)
        for col in range(rules.num_reels)
    ]
    rng.shuffle(positions)
    for row, col in positions[: min(count, len(positions))]:
        multiplier_cells[row, col] = int(
            _choice(
                rules.multiplier_values,
                rules.multiplier_value_probabilities,
                rng,
            )
        )


def run_mega_combo_feature(
    base_window=None,
    rules=MEGA_COMBO_FEATURE_CONFIG,
    seed=None,
):
    """Run the Mega/Combo feature, combining all six single-feature powers."""
    rng = np.random.default_rng(seed)
    if base_window is None:
        base_window = _default_combo_base_window(rules)
    base_window = np.asarray(base_window, dtype=np.int16)
    if base_window.shape != (3, rules.num_reels):
        raise ValueError("base_window must be a 3x5 converted base window")

    board = np.full((rules.num_rows, rules.num_reels), EMPTY_SYMBOL, dtype=np.int16)
    coin_values = np.zeros((rules.num_rows, rules.num_reels), dtype=np.float64)
    splitter_coin_counts = np.zeros((rules.num_rows, rules.num_reels), dtype=np.int16)
    multiplier_cells = np.ones((rules.num_rows, rules.num_reels), dtype=np.int16)
    expire_next = np.zeros((rules.num_rows, rules.num_reels), dtype=bool)
    jackpot_meters = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)
    awarded_jackpots = np.zeros(len(rules.jackpot_symbols), dtype=np.int16)

    locked_until = rules.num_rows - rules.starting_unlocked_rows - 1
    active_start = locked_until + 1
    spins_left = rules.respin_reset_count
    total_spins = 0
    jackpot_win = 0.0
    collected_value = 0.0
    special_counts = {
        "go": 0,
        "locked_go": 0,
        "grower": 0,
        "collector": 0,
        "splitter": 0,
    }
    grower_positions = []

    for symbol, base_row, base_col in _combo_trigger_positions(base_window, rules):
        row = active_start + base_row
        col = base_col
        board[row, col] = symbol
        if symbol != rules.collector_symbol:
            coin_values[row, col] = _coin_value(rules, rng)
        if symbol == rules.go_symbol:
            special_counts["go"] += 1
        elif symbol == rules.grower_symbol:
            special_counts["grower"] += 1
            grower_positions.append((row, col))
        elif symbol == rules.collector_symbol:
            special_counts["collector"] += 1
        elif symbol == rules.splitter_symbol:
            special_counts["splitter"] += 1

    additional_coin_count = int(
        _choice(
            rules.additional_coin_counts,
            rules.additional_coin_count_probabilities,
            rng,
        )
    )
    for row, col in _random_empty_positions(
        board,
        rng,
        additional_coin_count,
        row_min=active_start,
    ):
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = _coin_value(rules, rng)
        splitter_coin_counts[row, col] = 1

    def place_splitter_cell(row, col):
        count = int(
            _choice(
                rules.splitter_coin_counts,
                rules.splitter_coin_count_probabilities,
                rng,
            )
        )
        value = 0.0
        for _ in range(count):
            value += float(
                _choice(
                    rules.splitter_coin_values,
                    rules.splitter_coin_value_probabilities,
                    rng,
                )
            )
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = value
        splitter_coin_counts[row, col] = count

    def normal_positions():
        return _normal_coin_positions(board, rules)

    def grow_from_position(row, col):
        targets = [
            position
            for position in normal_positions()
            if position != (row, col) and position[0] > locked_until
        ]
        count = int(_choice(rules.grow_coin_counts, rules.grow_coin_count_probabilities, rng))
        grow_value = float(_choice(rules.grow_values, rules.grow_value_probabilities, rng))
        for target_row, target_col in _choose_positions_without_replacement(targets, rng, count):
            coin_values[target_row, target_col] += grow_value

    def boost_active_coins(exclude=None):
        if exclude is None:
            exclude = set()
        boost_value = float(_choice(rules.boost_values, rules.boost_value_probabilities, rng))
        for row, col in normal_positions():
            if (row, col) not in exclude and row > locked_until:
                coin_values[row, col] += boost_value
        return boost_value

    def collect_at(row, col):
        total = 0.0
        for coin_row, coin_col in normal_positions():
            if (coin_row, coin_col) != (row, col) and coin_row > locked_until:
                total += coin_values[coin_row, coin_col]
        board[row, col] = rules.coin_symbol
        coin_values[row, col] = total
        splitter_coin_counts[row, col] = 1
        return total

    def act_special_at(row, col):
        nonlocal jackpot_win, collected_value, locked_until
        symbol = int(board[row, col])
        if symbol == rules.go_symbol:
            if locked_until >= 0:
                locked_until -= 1
            expire_next[row, col] = True
            return True
        if symbol == rules.grower_symbol:
            if (row, col) not in grower_positions:
                grower_positions.append((row, col))
            grow_from_position(row, col)
            return False
        if symbol == rules.collector_symbol:
            collected_value += collect_at(row, col)
            return False
        if symbol == rules.splitter_symbol:
            source_positions = [
                position
                for position in normal_positions()
                if position[0] > locked_until
            ]
            for source_row, source_col in list(source_positions):
                empties = _random_empty_positions(
                    board,
                    rng,
                    1,
                    row_min=locked_until + 1,
                )
                if not empties:
                    break
                target_row, target_col = empties[0]
                board[target_row, target_col] = rules.coin_symbol
                coin_values[target_row, target_col] = coin_values[source_row, source_col]
                splitter_coin_counts[target_row, target_col] = max(
                    1,
                    splitter_coin_counts[source_row, source_col],
                )
            board[row, col] = rules.coin_symbol
            if coin_values[row, col] <= 0.0:
                coin_values[row, col] = _coin_value(rules, rng)
            splitter_coin_counts[row, col] = 1
            return False
        if _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
            jackpot_win += _collect_jackpot(
                symbol,
                jackpot_meters,
                awarded_jackpots,
                rules,
            )
            expire_next[row, col] = True
            return True
        return False

    # Guaranteed opening actions in the order supplied by the rules answer.
    for symbol in (
        rules.splitter_symbol,
        rules.grower_symbol,
        rules.booster_symbol,
        rules.multiplier_symbol,
        rules.collector_symbol,
        rules.go_symbol,
    ):
        positions = [
            (row, col)
            for row in range(active_start, rules.num_rows)
            for col in range(rules.num_reels)
            if board[row, col] == symbol
        ]
        for row, col in positions:
            if symbol == rules.booster_symbol:
                boost_value = boost_active_coins(exclude={(row, col)})
                board[row, col] = rules.coin_symbol
                coin_values[row, col] = boost_value
                splitter_coin_counts[row, col] = 1
            elif symbol == rules.multiplier_symbol:
                _place_combo_multiplier_cells(multiplier_cells, rules, rng)
                board[row, col] = rules.coin_symbol
                if coin_values[row, col] <= 0.0:
                    coin_values[row, col] = _coin_value(rules, rng)
                splitter_coin_counts[row, col] = 1
            elif symbol == rules.go_symbol:
                board[row, col] = rules.coin_symbol
                if coin_values[row, col] <= 0.0:
                    coin_values[row, col] = _coin_value(rules, rng)
                splitter_coin_counts[row, col] = 1
            else:
                act_special_at(row, col)

    while spins_left > 0:
        total_spins += 1
        board[expire_next] = EMPTY_SYMBOL
        coin_values[expire_next] = 0.0
        splitter_coin_counts[expire_next] = 0
        expire_next[:, :] = False

        for grower_row, grower_col in list(grower_positions):
            if (
                grower_row > locked_until
                and board[grower_row, grower_col] == rules.grower_symbol
                and rng.random() < rules.grow_trigger_probability
            ):
                grow_from_position(grower_row, grower_col)

        if rng.random() < rules.multiplier_trigger_probability:
            _place_combo_multiplier_cells(multiplier_cells, rules, rng)

        if rng.random() < rules.boost_trigger_probability:
            boost_active_coins()

        landed_unlocked_symbol = False
        pending_active_specials = []
        for row in range(rules.num_rows):
            is_locked = row <= locked_until
            symbols = rules.locked_landing_symbols if is_locked else rules.unlocked_landing_symbols
            probabilities = (
                rules.locked_landing_probabilities
                if is_locked
                else rules.unlocked_landing_probabilities
            )
            for col in range(rules.num_reels):
                if board[row, col] != EMPTY_SYMBOL:
                    continue
                active_empty_count = int(
                    np.count_nonzero(board[locked_until + 1 :, :] == EMPTY_SYMBOL)
                )
                force_splitter = (
                    not is_locked
                    and special_counts["splitter"] < rules.min_splitter_symbols
                )
                if force_splitter:
                    symbol = rules.splitter_symbol
                else:
                    symbol = int(_choice(symbols, probabilities, rng))
                if symbol == EMPTY_SYMBOL:
                    continue

                if symbol == rules.go_symbol:
                    if special_counts["go"] >= rules.max_go_symbols:
                        continue
                    if is_locked and special_counts["locked_go"] >= rules.max_locked_go_symbols:
                        continue
                    special_counts["go"] += 1
                    if is_locked:
                        special_counts["locked_go"] += 1
                elif symbol == rules.grower_symbol:
                    if special_counts["grower"] >= rules.max_grower_symbols:
                        continue
                    special_counts["grower"] += 1
                elif symbol == rules.collector_symbol:
                    if special_counts["collector"] >= rules.max_collector_symbols:
                        continue
                    special_counts["collector"] += 1
                elif symbol == rules.splitter_symbol:
                    if special_counts["splitter"] >= rules.max_splitter_symbols:
                        continue
                    special_counts["splitter"] += 1

                board[row, col] = symbol
                if symbol == rules.coin_symbol:
                    value = _coin_value(rules, rng)
                    if not is_locked and multiplier_cells[row, col] > 1:
                        value *= float(multiplier_cells[row, col])
                        multiplier_cells[row, col] = 1
                    coin_values[row, col] = value
                    splitter_coin_counts[row, col] = 1
                elif not is_locked and _jackpot_index(symbol, rules.jackpot_symbols) >= 0:
                    pending_active_specials.append((row, col))
                elif not is_locked and symbol in (
                    rules.go_symbol,
                    rules.grower_symbol,
                    rules.collector_symbol,
                    rules.splitter_symbol,
                ):
                    if symbol in (rules.grower_symbol, rules.splitter_symbol):
                        coin_values[row, col] = _coin_value(rules, rng)
                    pending_active_specials.append((row, col))

                if not is_locked:
                    landed_unlocked_symbol = True

        for row, col in pending_active_specials:
            act_special_at(row, col)

        unlock_again = True
        while unlock_again:
            unlock_again = False
            for row in range(locked_until + 1, rules.num_rows):
                for col in range(rules.num_reels):
                    symbol = int(board[row, col])
                    if symbol == rules.go_symbol and not expire_next[row, col]:
                        act_special_at(row, col)
                        unlock_again = True
                    elif _jackpot_index(symbol, rules.jackpot_symbols) >= 0 and not expire_next[row, col]:
                        act_special_at(row, col)
                    elif symbol in (
                        rules.grower_symbol,
                        rules.collector_symbol,
                        rules.splitter_symbol,
                    ) and not expire_next[row, col]:
                        if symbol in (rules.grower_symbol, rules.splitter_symbol) and coin_values[row, col] <= 0.0:
                            coin_values[row, col] = _coin_value(rules, rng)
                        act_special_at(row, col)

        spins_left = rules.respin_reset_count if landed_unlocked_symbol else spins_left - 1
        if _grid_is_full(board):
            break

    coin_win = float(np.sum(coin_values))
    return FeatureResult(
        board=board,
        coin_values=coin_values,
        total_win=_feature_total(coin_win, jackpot_win),
        total_spins=total_spins,
        jackpot_meters=jackpot_meters,
        awarded_jackpots=awarded_jackpots,
        jackpot_win=jackpot_win,
        unlocked_rows=rules.num_rows - locked_until - 1,
        go_landed=special_counts["go"],
        locked_go_landed=special_counts["locked_go"],
        multiplier_cells=multiplier_cells,
        special_landed=special_counts["splitter"],
        collected_value=collected_value,
        splitter_coin_counts=splitter_coin_counts,
        special_counts=special_counts,
    )
