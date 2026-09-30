import numpy as np

from Numba_Engine import splitter_bag


def _certain(value):
    return np.array([value], dtype=np.int8), np.array([1.0])


def test_splitter_copies_each_selected_coin_value_to_unlocked_cells():
    board = np.zeros((6, 5), dtype=np.int16)
    board[3, 0] = 10
    board[4, 1] = 20
    coin_positions = np.array([15, 21], dtype=np.int32)
    coin_mask = np.zeros(board.size, dtype=np.bool_)
    coin_mask[coin_positions] = True
    source_counts, source_probabilities = _certain(2)
    split_counts, split_probabilities = _certain(2)

    splitter_bag(
        board,
        coin_mask,
        coin_positions,
        source_probabilities,
        source_counts,
        split_probabilities,
        split_counts,
        locked_row_idx=2,
    )

    assert np.count_nonzero(board == 10) == 3
    assert np.count_nonzero(board == 20) == 3
    assert np.count_nonzero(board[:3]) == 0
    assert np.count_nonzero(coin_mask) == 6


def test_splitter_caps_generated_coins_at_available_positions():
    board = np.full((6, 5), 99, dtype=np.int16)
    board[:3] = 0
    board[5, 4] = 0
    board[3, 0] = 7
    coin_mask = np.zeros(board.size, dtype=np.bool_)
    coin_mask[15] = True
    source_counts, source_probabilities = _certain(1)
    split_counts, split_probabilities = _certain(3)

    splitter_bag(
        board,
        coin_mask,
        np.array([15], dtype=np.int32),
        source_probabilities,
        source_counts,
        split_probabilities,
        split_counts,
        locked_row_idx=2,
    )

    assert board[5, 4] == 7
    assert np.count_nonzero(board[:3]) == 0
    assert coin_mask[29]


def test_splitter_does_nothing_without_current_spin_candidates():
    board = np.zeros((6, 5), dtype=np.int16)
    board[3, 0] = 10
    original = board.copy()
    coin_mask = np.zeros(board.size, dtype=np.bool_)
    coin_mask[15] = True
    source_counts, source_probabilities = _certain(1)
    split_counts, split_probabilities = _certain(1)

    splitter_bag(
        board,
        coin_mask,
        np.empty(0, dtype=np.int32),
        source_probabilities,
        source_counts,
        split_probabilities,
        split_counts,
        locked_row_idx=2,
    )

    np.testing.assert_array_equal(board, original)
