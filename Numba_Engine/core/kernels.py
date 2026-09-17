import numpy as np
from numba import njit


@njit
def probChoice(total_wts, mult_numbers):
    cum_sum = np.zeros(total_wts.shape[0])
    for idx in range(total_wts.shape[0]):
        cum_sum[idx] = np.sum(total_wts[:idx + 1])
    choice = np.random.uniform(0, 1)
    matching_indices = np.where(choice < cum_sum)[0]
    if len(matching_indices) == 0:
        return mult_numbers[-1]
    return mult_numbers[int(min(matching_indices))]


@njit
def select_reelset_index(weights):
    """Select one reelset index from a normalized weight array."""
    choice = np.random.uniform(0.0, 1.0)
    cumulative_weight = 0.0

    for reelset_idx in range(weights.shape[0]):
        cumulative_weight += weights[reelset_idx]
        if choice < cumulative_weight:
            return reelset_idx

    # Protect against the normalized weights summing to fractionally less than
    # one due to floating-point rounding.
    return weights.shape[0] - 1


@njit
def count_matching_symbols(pay_window, symbols):
    """Count cells matching any symbol in ``symbols``."""
    count = 0
    for row in range(pay_window.shape[0]):
        for col in range(pay_window.shape[1]):
            symbol = pay_window[row, col]
            for target in symbols:
                if symbol == target:
                    count += 1
                    break
    return count


@njit
def has_free_game_trigger(pay_window, rules):
    """Return true when the configured number of any SC symbols is visible."""
    return count_matching_symbols(
        pay_window,
        rules.free_game_symbols,
    ) >= rules.free_game_trigger_count


@njit
def make_board(reel_lens, reelset_numba, rules):
    reel_stops = np.zeros(rules.num_reels, dtype=np.int16)
    pay_window = np.full(
        (rules.num_rows, rules.num_reels),
        rules.reel_padding_symbol,
        dtype=np.int16,
    )

    for reel in range(rules.num_reels):
        reel_stops[reel] = np.random.choice(reel_lens[reel])

    for reel in range(rules.num_reels):
        for row in range(rules.num_rows):
            pay_window[row, reel] = reelset_numba[
                (reel_stops[reel] + row) % reel_lens[reel],
                reel,
            ]

    return pay_window


@njit
def line_win_eval(pay_window, pay_table, pay_lines, wild_symbol):
    """Evaluate left-to-right line wins.

    The pay-table column and returned match count use the convention from the
    original game rule: zero is the first reel, one is a two-reel match, and
    four is a five-reel match.

    Symbols without a row in ``pay_table`` are treated as non-paying feature
    symbols. This keeps Coin, Collect, Scatter, and Jackpot tokens out of line
    evaluation without relying on a single scatter-symbol ID.
    """
    num_reels = pay_window.shape[1]
    num_lines = pay_lines.shape[0]
    num_pay_symbols = pay_table.shape[0]

    hit_counts = np.zeros(
        (num_pay_symbols, num_reels),
        dtype=np.int64,
    )
    win_amounts = np.zeros(
        (num_pay_symbols, num_reels),
        dtype=np.float64,
    )
    winning_window = pay_window.copy()
    winning_symbols = np.full(num_lines, -1, dtype=np.int16)
    match_counts = np.full(num_lines, -1, dtype=np.int8)
    line_win_amounts = np.zeros(num_lines, dtype=np.float64)
    line_symbols = np.empty(num_reels, dtype=np.int16)
    total_win = 0.0

    for line_idx in range(num_lines):
        for reel_idx in range(num_reels):
            position = pay_lines[line_idx, reel_idx]
            row = position // num_reels
            reel = position % num_reels
            line_symbols[reel_idx] = pay_window[row, reel]

        first_symbol = int(line_symbols[0])
        if first_symbol != wild_symbol and (
            first_symbol < 0 or first_symbol >= num_pay_symbols
        ):
            continue

        paying_symbol = first_symbol
        extra_matches = 0
        leading_wild_match_count = 0
        found_non_wild = False

        for reel_idx in range(num_reels - 1):
            current_symbol = int(line_symbols[reel_idx])
            next_symbol = int(line_symbols[reel_idx + 1])

            if current_symbol == next_symbol or next_symbol == wild_symbol:
                extra_matches += 1
            elif (
                current_symbol == wild_symbol
                and (paying_symbol == next_symbol or paying_symbol == wild_symbol)
                and 0 <= next_symbol < num_pay_symbols
            ):
                extra_matches += 1
                if not found_non_wild:
                    leading_wild_match_count = reel_idx
                    paying_symbol = next_symbol
                    found_non_wild = True
            else:
                break

        if extra_matches == 0:
            continue

        winning_symbol = paying_symbol
        winning_match_count = extra_matches
        line_win = pay_table[paying_symbol, extra_matches]

        if first_symbol == wild_symbol and found_non_wild:
            wild_win = pay_table[wild_symbol, leading_wild_match_count]
            if leading_wild_match_count > 0 and wild_win >= line_win:
                winning_symbol = wild_symbol
                winning_match_count = leading_wild_match_count
                line_win = wild_win

        total_win += line_win
        winning_symbols[line_idx] = winning_symbol
        match_counts[line_idx] = winning_match_count
        line_win_amounts[line_idx] = line_win

    for line_idx in range(num_lines):
        symbol = int(winning_symbols[line_idx])
        match_count = int(match_counts[line_idx])

        # Three or more matching symbols constitute a recorded line hit.
        if symbol != -1 and match_count >= 2:
            hit_counts[symbol, match_count] += 1
            win_amounts[symbol, match_count] += pay_table[symbol, match_count]

            for reel_idx in range(match_count + 1):
                position = pay_lines[line_idx, reel_idx]
                row = position // num_reels
                reel = position % num_reels
                winning_window[row, reel] = -1

    return (
        total_win,
        winning_symbols,
        match_counts,
        line_win_amounts,
        winning_window,
        hit_counts,
        win_amounts,
    )


@njit
def cascade(pay_window, cascade_reel_lens, cascade_reelset_numba, rules):
    num_rows, num_cols = pay_window.shape
    num_blanks = np.zeros(num_cols, dtype=np.int8)

    for reel in range(num_cols):
        num_blanks[reel] = np.count_nonzero(
            pay_window[:, reel] == rules.empty_position
        )

    cascade_pay_window = np.full(
        pay_window.shape,
        rules.empty_position,
        dtype=np.int16,
    )
    for reel in range(num_cols):
        cascade_idx = num_blanks[reel]
        for row in range(num_rows):
            if pay_window[row, reel] != rules.empty_position:
                cascade_pay_window[cascade_idx, reel] = pay_window[row, reel]
                cascade_idx += 1

    cascade_stops = np.zeros(num_cols, dtype=np.int16)
    for reel in range(num_cols):
        cascade_stops[reel] = np.random.choice(cascade_reel_lens[reel])

    for reel in range(num_cols):
        for blank in range(num_blanks[reel]):
            cascade_pay_window[blank, reel] = cascade_reelset_numba[
                (cascade_stops[reel] + blank) % cascade_reel_lens[reel],
                reel,
            ]

    return cascade_pay_window


@njit
def DFS(matrix, starting_idx, symbol, rules):
    num_rows, num_cols = matrix.shape
    num_cells = num_rows * num_cols

    used = np.zeros(num_cells, dtype=np.bool_)
    stack = np.empty(num_cells, dtype=np.int32)
    indices = np.empty(num_cells, dtype=np.int32)

    stack_size = 1
    num_indices = 0
    stack[0] = starting_idx
    used[starting_idx] = True

    while stack_size > 0:
        stack_size -= 1
        cur_idx = stack[stack_size]
        indices[num_indices] = cur_idx
        num_indices += 1

        row = cur_idx // num_cols
        col = cur_idx % num_cols

        if row > 0:
            next_idx = cur_idx - num_cols
            next_symbol = matrix[row - 1, col]
            if not used[next_idx] and (
                next_symbol == symbol or next_symbol == rules.wild_symbol
            ):
                used[next_idx] = True
                stack[stack_size] = next_idx
                stack_size += 1

        if row + 1 < num_rows:
            next_idx = cur_idx + num_cols
            next_symbol = matrix[row + 1, col]
            if not used[next_idx] and (
                next_symbol == symbol or next_symbol == rules.wild_symbol
            ):
                used[next_idx] = True
                stack[stack_size] = next_idx
                stack_size += 1

        if col > 0:
            next_idx = cur_idx - 1
            next_symbol = matrix[row, col - 1]
            if not used[next_idx] and (
                next_symbol == symbol or next_symbol == rules.wild_symbol
            ):
                used[next_idx] = True
                stack[stack_size] = next_idx
                stack_size += 1

        if col + 1 < num_cols:
            next_idx = cur_idx + 1
            next_symbol = matrix[row, col + 1]
            if not used[next_idx] and (
                next_symbol == symbol or next_symbol == rules.wild_symbol
            ):
                used[next_idx] = True
                stack[stack_size] = next_idx
                stack_size += 1

    return indices[:num_indices]


@njit
def cluster_evaluation_rules(
    pay_window,
    multiplier_window,
    pay_table,
    rules,
    payout_multiplier,
):
    num_rows, num_cols = pay_window.shape
    num_cells = num_rows * num_cols

    used = np.zeros(num_cells, dtype=np.bool_)
    winning_mask = np.zeros(num_cells, dtype=np.bool_)
    win = 0.0
    symbol_wins = np.zeros(
        rules.num_paying_symbols,
        dtype=np.float64,
    )

    for symbol in range(rules.num_paying_symbols):
        for starting_idx in range(num_cells):
            row = starting_idx // num_cols
            col = starting_idx % num_cols

            if not used[starting_idx] and pay_window[row, col] == symbol:
                indices = DFS(pay_window, starting_idx, symbol, rules)

                for idx in indices:
                    used[idx] = True

                if len(indices) >= rules.min_cluster_size:
                    multiplier_sum = 0
                    for idx in indices:
                        winning_mask[idx] = True
                        multiplier_sum += multiplier_window[
                            idx // num_cols,
                            idx % num_cols,
                        ]

                    cluster_size = min(len(indices), rules.max_cluster_size)
                    cluster_multiplier = max(
                        rules.default_cluster_multiplier,
                        multiplier_sum,
                    )
                    cluster_win = (
                        pay_table[
                            symbol,
                            cluster_size - rules.min_cluster_size,
                        ]
                        * cluster_multiplier
                        * payout_multiplier
                    )
                    symbol_wins[symbol] += cluster_win
                    win += cluster_win

    for idx in range(num_cells):
        if winning_mask[idx]:
            row = idx // num_cols
            col = idx % num_cols
            pay_window[row, col] = rules.empty_position

            if (
                multiplier_window[row, col]
                == rules.unmarked_position_multiplier
            ):
                multiplier_window[row, col] = rules.first_position_multiplier
            else:
                multiplier_window[row, col] = min(
                    rules.max_position_multiplier,
                    multiplier_window[row, col]
                    * rules.position_multiplier_growth_factor,
                )

    return pay_window, multiplier_window, winning_mask, win, symbol_wins
