import numpy as np
from pathlib import Path

from numba import boolean, float64, int8, int16, int32, int64
from numba.experimental import jitclass


NPZ_LIBRARY_DIR = (
    Path(__file__).resolve().parents[1] / "output" / "npz_library"
)
NPZ_FORMAT_VERSION = 1


_NPZ_ARRAY_KEYS = (
    "boards",
    "coin_value_boards",
    "jackpot_overlay_boards",
    "jackpot_values_before",
    "jackpot_values_after",
    "jackpot_increment_counts",
    "wins",
    "line_wins",
    "collect_wins",
    "symbol_wins",
    "symbol_hit_counts",
    "winning_line_numbers",
    "line_winning_symbols",
    "line_symbol_counts",
    "line_win_amounts",
    "spin_triggers",
    "collector_counts",
    "round_spin_offsets",
    "round_wins",
    "round_triggers",
)

_SPIN_ARRAY_KEYS = (
    "boards",
    "coin_value_boards",
    "jackpot_overlay_boards",
    "jackpot_values_before",
    "jackpot_values_after",
    "jackpot_increment_counts",
    "wins",
    "line_wins",
    "collect_wins",
    "symbol_wins",
    "symbol_hit_counts",
    "winning_line_numbers",
    "line_winning_symbols",
    "line_symbol_counts",
    "line_win_amounts",
    "spin_triggers",
    "collector_counts",
)

_ROUND_ARRAY_KEYS = (
    "round_wins",
    "round_triggers",
)


storage_spec = [
    # One entry for every paid spin (one Collector-loop iteration).
    ("boards", int16[:, :, :]),
    ("coin_value_boards", float64[:, :, :]),
    ("jackpot_overlay_boards", int8[:, :, :]),
    ("jackpot_values_before", float64[:, :]),
    ("jackpot_values_after", float64[:, :]),
    ("jackpot_increment_counts", int16[:, :]),
    ("wins", float64[:]),
    ("line_wins", float64[:]),
    ("collect_wins", float64[:]),
    ("spin_triggers", boolean[:]),
    ("collector_counts", int16[:]),

    # Statistical line-win information. The final axis is the paytable index:
    # index 2 represents a three-symbol left-to-right win, for example.
    ("symbol_wins", float64[:, :, :]),
    ("symbol_hit_counts", int64[:, :, :]),

    # Compact backend-facing winning-line information. Unused entries are -1
    # for integer arrays and 0.0 for line_win_amounts.
    ("winning_line_numbers", int16[:, :]),
    ("line_winning_symbols", int16[:, :]),
    ("line_symbol_counts", int8[:, :]),
    ("line_win_amounts", float64[:, :]),

    # One outer round is one call to run_one_spin(). A round may contain
    # several paid spins while walking Collectors remain active.
    ("round_spin_offsets", int64[:]),
    ("round_wins", float64[:]),
    ("round_triggers", int64[:]),
    ("spin_count", int64),
    ("round_count", int64),
]


@jitclass(storage_spec)
class Storage:
    def __init__(
        self,
        spin_capacity,
        num_rows,
        num_reels,
        num_symbols,
        num_lines,
        num_jackpots,
    ):
        if spin_capacity < 1:
            raise ValueError("Storage capacity must be positive")
        if num_rows < 1 or num_reels < 1:
            raise ValueError("Board dimensions must be positive")
        if num_symbols < 1 or num_lines < 1 or num_jackpots < 1:
            raise ValueError("Symbol, line and jackpot counts must be positive")

        self.boards = np.empty(
            (spin_capacity, num_rows, num_reels),
            dtype=np.int16,
        )
        self.coin_value_boards = np.empty(
            (spin_capacity, num_rows, num_reels),
            dtype=np.float64,
        )
        self.jackpot_overlay_boards = np.full(
            (spin_capacity, num_rows, num_reels),
            -1,
            dtype=np.int8,
        )
        self.jackpot_values_before = np.empty(
            (spin_capacity, num_jackpots),
            dtype=np.float64,
        )
        self.jackpot_values_after = np.empty(
            (spin_capacity, num_jackpots),
            dtype=np.float64,
        )
        self.jackpot_increment_counts = np.zeros(
            (spin_capacity, num_jackpots),
            dtype=np.int16,
        )
        self.wins = np.empty(spin_capacity, dtype=np.float64)
        self.line_wins = np.empty(spin_capacity, dtype=np.float64)
        self.collect_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_triggers = np.empty(spin_capacity, dtype=np.bool_)
        self.collector_counts = np.empty(spin_capacity, dtype=np.int16)

        self.symbol_wins = np.empty(
            (spin_capacity, num_symbols, num_reels),
            dtype=np.float64,
        )
        self.symbol_hit_counts = np.empty(
            (spin_capacity, num_symbols, num_reels),
            dtype=np.int64,
        )

        self.winning_line_numbers = np.full(
            (spin_capacity, num_lines),
            -1,
            dtype=np.int16,
        )
        self.line_winning_symbols = np.full(
            (spin_capacity, num_lines),
            -1,
            dtype=np.int16,
        )
        self.line_symbol_counts = np.full(
            (spin_capacity, num_lines),
            -1,
            dtype=np.int8,
        )
        self.line_win_amounts = np.zeros(
            (spin_capacity, num_lines),
            dtype=np.float64,
        )

        # A round cannot outnumber its paid spins, so the initial paid-spin
        # capacity is also a safe initial round capacity.
        self.round_spin_offsets = np.empty(
            spin_capacity + 1,
            dtype=np.int64,
        )
        self.round_wins = np.empty(spin_capacity, dtype=np.float64)
        self.round_triggers = np.empty(spin_capacity, dtype=np.int64)

        self.spin_count = 0
        self.round_count = 0
        self.round_spin_offsets[0] = 0

    def _grow_spins(self):
        old_capacity = self.boards.shape[0]
        new_capacity = old_capacity * 2
        num_rows = self.boards.shape[1]
        num_reels = self.boards.shape[2]
        num_symbols = self.symbol_wins.shape[1]
        num_lines = self.winning_line_numbers.shape[1]
        num_jackpots = self.jackpot_values_before.shape[1]
        count = self.spin_count

        boards = np.empty(
            (new_capacity, num_rows, num_reels),
            dtype=np.int16,
        )
        coin_value_boards = np.empty(
            (new_capacity, num_rows, num_reels),
            dtype=np.float64,
        )
        jackpot_overlay_boards = np.full(
            (new_capacity, num_rows, num_reels),
            -1,
            dtype=np.int8,
        )
        jackpot_values_before = np.empty(
            (new_capacity, num_jackpots),
            dtype=np.float64,
        )
        jackpot_values_after = np.empty(
            (new_capacity, num_jackpots),
            dtype=np.float64,
        )
        jackpot_increment_counts = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.int16,
        )
        wins = np.empty(new_capacity, dtype=np.float64)
        line_wins = np.empty(new_capacity, dtype=np.float64)
        collect_wins = np.empty(new_capacity, dtype=np.float64)
        spin_triggers = np.empty(new_capacity, dtype=np.bool_)
        collector_counts = np.empty(new_capacity, dtype=np.int16)
        symbol_wins = np.empty(
            (new_capacity, num_symbols, num_reels),
            dtype=np.float64,
        )
        symbol_hit_counts = np.empty(
            (new_capacity, num_symbols, num_reels),
            dtype=np.int64,
        )
        winning_line_numbers = np.full(
            (new_capacity, num_lines),
            -1,
            dtype=np.int16,
        )
        line_winning_symbols = np.full(
            (new_capacity, num_lines),
            -1,
            dtype=np.int16,
        )
        line_symbol_counts = np.full(
            (new_capacity, num_lines),
            -1,
            dtype=np.int8,
        )
        line_win_amounts = np.zeros(
            (new_capacity, num_lines),
            dtype=np.float64,
        )

        boards[:count] = self.boards[:count]
        coin_value_boards[:count] = self.coin_value_boards[:count]
        jackpot_overlay_boards[:count] = self.jackpot_overlay_boards[:count]
        jackpot_values_before[:count] = self.jackpot_values_before[:count]
        jackpot_values_after[:count] = self.jackpot_values_after[:count]
        jackpot_increment_counts[:count] = self.jackpot_increment_counts[:count]
        wins[:count] = self.wins[:count]
        line_wins[:count] = self.line_wins[:count]
        collect_wins[:count] = self.collect_wins[:count]
        spin_triggers[:count] = self.spin_triggers[:count]
        collector_counts[:count] = self.collector_counts[:count]
        symbol_wins[:count] = self.symbol_wins[:count]
        symbol_hit_counts[:count] = self.symbol_hit_counts[:count]
        winning_line_numbers[:count] = self.winning_line_numbers[:count]
        line_winning_symbols[:count] = self.line_winning_symbols[:count]
        line_symbol_counts[:count] = self.line_symbol_counts[:count]
        line_win_amounts[:count] = self.line_win_amounts[:count]

        self.boards = boards
        self.coin_value_boards = coin_value_boards
        self.jackpot_overlay_boards = jackpot_overlay_boards
        self.jackpot_values_before = jackpot_values_before
        self.jackpot_values_after = jackpot_values_after
        self.jackpot_increment_counts = jackpot_increment_counts
        self.wins = wins
        self.line_wins = line_wins
        self.collect_wins = collect_wins
        self.spin_triggers = spin_triggers
        self.collector_counts = collector_counts
        self.symbol_wins = symbol_wins
        self.symbol_hit_counts = symbol_hit_counts
        self.winning_line_numbers = winning_line_numbers
        self.line_winning_symbols = line_winning_symbols
        self.line_symbol_counts = line_symbol_counts
        self.line_win_amounts = line_win_amounts

    def _grow_rounds(self):
        old_capacity = self.round_wins.shape[0]
        new_capacity = old_capacity * 2
        count = self.round_count

        round_spin_offsets = np.empty(new_capacity + 1, dtype=np.int64)
        round_wins = np.empty(new_capacity, dtype=np.float64)
        round_triggers = np.empty(new_capacity, dtype=np.int64)

        round_spin_offsets[:count + 1] = self.round_spin_offsets[:count + 1]
        round_wins[:count] = self.round_wins[:count]
        round_triggers[:count] = self.round_triggers[:count]

        self.round_spin_offsets = round_spin_offsets
        self.round_wins = round_wins
        self.round_triggers = round_triggers

    def begin_round(self):
        if self.round_count == self.round_wins.shape[0]:
            self._grow_rounds()
        self.round_spin_offsets[self.round_count] = self.spin_count

    def finish_round(self, round_win, round_triggers):
        index = self.round_count
        self.round_wins[index] = round_win
        self.round_triggers[index] = round_triggers
        self.round_count += 1
        self.round_spin_offsets[self.round_count] = self.spin_count

    def save_spin(
        self,
        board,
        coin_value_board,
        jackpot_overlay_board,
        jackpot_values_before,
        jackpot_values_after,
        jackpot_increment_counts,
        line_win,
        collect_win,
        symbol_wins,
        symbol_hit_counts,
        line_winning_symbols,
        line_match_counts,
        line_win_amounts,
        free_game_trigger,
        collector_count,
    ):
        if self.spin_count == self.boards.shape[0]:
            self._grow_spins()

        index = self.spin_count
        self.boards[index] = board
        self.coin_value_boards[index] = coin_value_board
        self.jackpot_overlay_boards[index] = jackpot_overlay_board
        self.jackpot_values_before[index] = jackpot_values_before
        self.jackpot_values_after[index] = jackpot_values_after
        self.jackpot_increment_counts[index] = jackpot_increment_counts
        self.line_wins[index] = line_win
        self.collect_wins[index] = collect_win
        self.wins[index] = line_win + collect_win
        self.spin_triggers[index] = free_game_trigger
        self.collector_counts[index] = collector_count
        self.symbol_wins[index] = symbol_wins
        self.symbol_hit_counts[index] = symbol_hit_counts

        # Compact only actual paying lines for the backend. line_match_counts
        # uses the paytable-index convention, so add one to store the actual
        # number of matching symbols (2 -> three matching symbols).
        self.winning_line_numbers[index] = -1
        self.line_winning_symbols[index] = -1
        self.line_symbol_counts[index] = -1
        self.line_win_amounts[index] = 0.0
        write_index = 0
        for line_index in range(len(line_win_amounts)):
            if line_win_amounts[line_index] > 0.0:
                self.winning_line_numbers[index, write_index] = line_index
                self.line_winning_symbols[index, write_index] = (
                    line_winning_symbols[line_index]
                )
                self.line_symbol_counts[index, write_index] = (
                    line_match_counts[line_index] + 1
                )
                self.line_win_amounts[index, write_index] = (
                    line_win_amounts[line_index]
                )
                write_index += 1

        self.spin_count += 1


compact_base_storage_spec = [
    ("spin_count", int64),
    ("round_count", int64),
]


@jitclass(compact_base_storage_spec)
class CompactBaseStorage:
    """Counter-only base storage for memory-efficient full-game simulations."""

    def __init__(self):
        self.spin_count = 0
        self.round_count = 0

    def begin_round(self):
        return

    def finish_round(self, round_win, round_triggers):
        self.round_count += 1

    def save_spin(
        self,
        board,
        coin_value_board,
        jackpot_overlay_board,
        jackpot_values_before,
        jackpot_values_after,
        jackpot_increment_counts,
        line_win,
        collect_win,
        symbol_wins,
        symbol_hit_counts,
        line_winning_symbols,
        line_match_counts,
        line_win_amounts,
        free_game_trigger,
        collector_count,
    ):
        self.spin_count += 1


full_game_storage_spec = [
    ("spin_feature_session_indices", int64[:]),
    ("spin_base_wins", float64[:]),
    ("spin_feature_wins", float64[:]),
    ("spin_jackpot_wins", float64[:]),
    ("spin_total_wins", float64[:]),
    ("spin_jackpot_values_before_feature", float64[:, :]),
    ("spin_jackpot_awards", boolean[:, :]),
    ("spin_jackpot_award_amounts", float64[:, :]),
    ("spin_jackpot_values_after_feature", float64[:, :]),
    ("round_spin_offsets", int64[:]),
    ("round_base_wins", float64[:]),
    ("round_feature_wins", float64[:]),
    ("round_jackpot_wins", float64[:]),
    ("round_total_wins", float64[:]),
    ("spin_count", int64),
    ("round_count", int64),
]


@jitclass(full_game_storage_spec)
class _LegacyFullGameStorage:
    """Integration storage linking paid base spins to feature sessions."""

    def __init__(self, spin_capacity, round_capacity, num_jackpots):
        if spin_capacity < 1 or round_capacity < 1:
            raise ValueError("Full-game capacities must be positive")
        if num_jackpots < 1:
            raise ValueError("Jackpot count must be positive")

        self.spin_feature_session_indices = np.full(
            spin_capacity,
            -1,
            dtype=np.int64,
        )
        self.spin_base_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_feature_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_jackpot_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_total_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_jackpot_values_before_feature = np.empty(
            (spin_capacity, num_jackpots),
            dtype=np.float64,
        )
        self.spin_jackpot_awards = np.zeros(
            (spin_capacity, num_jackpots),
            dtype=np.bool_,
        )
        self.spin_jackpot_award_amounts = np.zeros(
            (spin_capacity, num_jackpots),
            dtype=np.float64,
        )
        self.spin_jackpot_values_after_feature = np.empty(
            (spin_capacity, num_jackpots),
            dtype=np.float64,
        )

        self.round_spin_offsets = np.empty(round_capacity + 1, dtype=np.int64)
        self.round_base_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_feature_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_jackpot_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_total_wins = np.empty(round_capacity, dtype=np.float64)
        self.spin_count = 0
        self.round_count = 0
        self.round_spin_offsets[0] = 0

    def _grow_spins(self):
        old_capacity = self.spin_base_wins.shape[0]
        new_capacity = old_capacity * 2
        num_jackpots = self.spin_jackpot_awards.shape[1]
        count = self.spin_count

        feature_indices = np.full(new_capacity, -1, dtype=np.int64)
        base_wins = np.empty(new_capacity, dtype=np.float64)
        feature_wins = np.empty(new_capacity, dtype=np.float64)
        jackpot_wins = np.empty(new_capacity, dtype=np.float64)
        total_wins = np.empty(new_capacity, dtype=np.float64)
        values_before = np.empty(
            (new_capacity, num_jackpots),
            dtype=np.float64,
        )
        awards = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.bool_,
        )
        award_amounts = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.float64,
        )
        values_after = np.empty(
            (new_capacity, num_jackpots),
            dtype=np.float64,
        )

        feature_indices[:count] = self.spin_feature_session_indices[:count]
        base_wins[:count] = self.spin_base_wins[:count]
        feature_wins[:count] = self.spin_feature_wins[:count]
        jackpot_wins[:count] = self.spin_jackpot_wins[:count]
        total_wins[:count] = self.spin_total_wins[:count]
        values_before[:count] = self.spin_jackpot_values_before_feature[:count]
        awards[:count] = self.spin_jackpot_awards[:count]
        award_amounts[:count] = self.spin_jackpot_award_amounts[:count]
        values_after[:count] = self.spin_jackpot_values_after_feature[:count]

        self.spin_feature_session_indices = feature_indices
        self.spin_base_wins = base_wins
        self.spin_feature_wins = feature_wins
        self.spin_jackpot_wins = jackpot_wins
        self.spin_total_wins = total_wins
        self.spin_jackpot_values_before_feature = values_before
        self.spin_jackpot_awards = awards
        self.spin_jackpot_award_amounts = award_amounts
        self.spin_jackpot_values_after_feature = values_after

    def _grow_rounds(self):
        old_capacity = self.round_base_wins.shape[0]
        new_capacity = old_capacity * 2
        count = self.round_count
        offsets = np.empty(new_capacity + 1, dtype=np.int64)
        base_wins = np.empty(new_capacity, dtype=np.float64)
        feature_wins = np.empty(new_capacity, dtype=np.float64)
        jackpot_wins = np.empty(new_capacity, dtype=np.float64)
        total_wins = np.empty(new_capacity, dtype=np.float64)
        offsets[:count + 1] = self.round_spin_offsets[:count + 1]
        base_wins[:count] = self.round_base_wins[:count]
        feature_wins[:count] = self.round_feature_wins[:count]
        jackpot_wins[:count] = self.round_jackpot_wins[:count]
        total_wins[:count] = self.round_total_wins[:count]
        self.round_spin_offsets = offsets
        self.round_base_wins = base_wins
        self.round_feature_wins = feature_wins
        self.round_jackpot_wins = jackpot_wins
        self.round_total_wins = total_wins

    def begin_round(self):
        if self.round_count == self.round_base_wins.shape[0]:
            self._grow_rounds()
        self.round_spin_offsets[self.round_count] = self.spin_count

    def save_spin(
        self,
        feature_session_index,
        base_win,
        feature_win,
        jackpot_win,
        jackpot_values_before_feature,
        jackpot_awards,
        jackpot_award_amounts,
        jackpot_values_after_feature,
    ):
        if self.spin_count == self.spin_base_wins.shape[0]:
            self._grow_spins()
        index = self.spin_count
        self.spin_feature_session_indices[index] = feature_session_index
        self.spin_base_wins[index] = base_win
        self.spin_feature_wins[index] = feature_win
        self.spin_jackpot_wins[index] = jackpot_win
        self.spin_total_wins[index] = base_win + feature_win + jackpot_win
        self.spin_jackpot_values_before_feature[index] = (
            jackpot_values_before_feature
        )
        self.spin_jackpot_awards[index] = jackpot_awards
        self.spin_jackpot_award_amounts[index] = jackpot_award_amounts
        self.spin_jackpot_values_after_feature[index] = (
            jackpot_values_after_feature
        )
        self.spin_count += 1

    def finish_round(self, base_win, feature_win, jackpot_win):
        index = self.round_count
        self.round_base_wins[index] = base_win
        self.round_feature_wins[index] = feature_win
        self.round_jackpot_wins[index] = jackpot_win
        self.round_total_wins[index] = base_win + feature_win + jackpot_win
        self.round_count += 1
        self.round_spin_offsets[self.round_count] = self.spin_count


active_full_game_storage_spec = [
    ("spin_feature_session_indices", int64[:]),
    ("spin_feature_masks", int16[:]),
    ("spin_base_wins", float64[:]),
    ("spin_feature_wins", float64[:]),
    ("spin_jackpot_wins", float64[:]),
    ("spin_total_wins", float64[:]),
    ("spin_jackpot_values_before_feature", float64[:, :]),
    ("spin_jackpot_awards", int16[:, :]),
    ("spin_jackpot_award_amounts", float64[:, :]),
    ("spin_jackpot_values_after_feature", float64[:, :]),
    ("feature_trigger_counts", int64[:]),
    ("feature_win_amounts", float64[:]),
    ("feature_spin_counts", int64[:]),
    ("round_spin_offsets", int64[:]),
    ("round_base_wins", float64[:]),
    ("round_feature_wins", float64[:]),
    ("round_jackpot_wins", float64[:]),
    ("round_total_wins", float64[:]),
    ("spin_count", int64),
    ("round_count", int64),
    ("feature_session_count", int64),
]


@jitclass(active_full_game_storage_spec)
class FullGameStorage:
    """Full-game storage for the seven configured feature routes.

    Feature detail is aggregated by route to keep long simulations compact.
    """

    def __init__(
        self,
        spin_capacity,
        round_capacity,
        num_jackpots,
        num_feature_types=7,
    ):
        if spin_capacity < 1 or round_capacity < 1:
            raise ValueError("Full-game capacities must be positive")
        if num_jackpots < 1 or num_feature_types < 1:
            raise ValueError("Jackpot and feature counts must be positive")

        self.spin_feature_session_indices = np.full(
            spin_capacity, -1, dtype=np.int64
        )
        self.spin_feature_masks = np.zeros(spin_capacity, dtype=np.int16)
        self.spin_base_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_feature_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_jackpot_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_total_wins = np.empty(spin_capacity, dtype=np.float64)
        self.spin_jackpot_values_before_feature = np.empty(
            (spin_capacity, num_jackpots), dtype=np.float64
        )
        self.spin_jackpot_awards = np.zeros(
            (spin_capacity, num_jackpots), dtype=np.int16
        )
        self.spin_jackpot_award_amounts = np.zeros(
            (spin_capacity, num_jackpots), dtype=np.float64
        )
        self.spin_jackpot_values_after_feature = np.empty(
            (spin_capacity, num_jackpots), dtype=np.float64
        )

        self.feature_trigger_counts = np.zeros(
            num_feature_types, dtype=np.int64
        )
        self.feature_win_amounts = np.zeros(
            num_feature_types, dtype=np.float64
        )
        self.feature_spin_counts = np.zeros(
            num_feature_types, dtype=np.int64
        )

        self.round_spin_offsets = np.empty(round_capacity + 1, dtype=np.int64)
        self.round_base_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_feature_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_jackpot_wins = np.empty(round_capacity, dtype=np.float64)
        self.round_total_wins = np.empty(round_capacity, dtype=np.float64)
        self.spin_count = 0
        self.round_count = 0
        self.feature_session_count = 0
        self.round_spin_offsets[0] = 0

    def _grow_spins(self):
        old_capacity = self.spin_base_wins.shape[0]
        new_capacity = old_capacity * 2
        count = self.spin_count
        num_jackpots = self.spin_jackpot_awards.shape[1]
        session_indices = np.full(new_capacity, -1, dtype=np.int64)
        masks = np.zeros(new_capacity, dtype=np.int16)
        base_wins = np.empty(new_capacity, dtype=np.float64)
        feature_wins = np.empty(new_capacity, dtype=np.float64)
        jackpot_wins = np.empty(new_capacity, dtype=np.float64)
        total_wins = np.empty(new_capacity, dtype=np.float64)
        values_before = np.empty((new_capacity, num_jackpots), dtype=np.float64)
        awards = np.zeros((new_capacity, num_jackpots), dtype=np.int16)
        award_amounts = np.zeros((new_capacity, num_jackpots), dtype=np.float64)
        values_after = np.empty((new_capacity, num_jackpots), dtype=np.float64)
        session_indices[:count] = self.spin_feature_session_indices[:count]
        masks[:count] = self.spin_feature_masks[:count]
        base_wins[:count] = self.spin_base_wins[:count]
        feature_wins[:count] = self.spin_feature_wins[:count]
        jackpot_wins[:count] = self.spin_jackpot_wins[:count]
        total_wins[:count] = self.spin_total_wins[:count]
        values_before[:count] = self.spin_jackpot_values_before_feature[:count]
        awards[:count] = self.spin_jackpot_awards[:count]
        award_amounts[:count] = self.spin_jackpot_award_amounts[:count]
        values_after[:count] = self.spin_jackpot_values_after_feature[:count]
        self.spin_feature_session_indices = session_indices
        self.spin_feature_masks = masks
        self.spin_base_wins = base_wins
        self.spin_feature_wins = feature_wins
        self.spin_jackpot_wins = jackpot_wins
        self.spin_total_wins = total_wins
        self.spin_jackpot_values_before_feature = values_before
        self.spin_jackpot_awards = awards
        self.spin_jackpot_award_amounts = award_amounts
        self.spin_jackpot_values_after_feature = values_after

    def _grow_rounds(self):
        old_capacity = self.round_base_wins.shape[0]
        new_capacity = old_capacity * 2
        count = self.round_count
        offsets = np.empty(new_capacity + 1, dtype=np.int64)
        base_wins = np.empty(new_capacity, dtype=np.float64)
        feature_wins = np.empty(new_capacity, dtype=np.float64)
        jackpot_wins = np.empty(new_capacity, dtype=np.float64)
        total_wins = np.empty(new_capacity, dtype=np.float64)
        offsets[:count + 1] = self.round_spin_offsets[:count + 1]
        base_wins[:count] = self.round_base_wins[:count]
        feature_wins[:count] = self.round_feature_wins[:count]
        jackpot_wins[:count] = self.round_jackpot_wins[:count]
        total_wins[:count] = self.round_total_wins[:count]
        self.round_spin_offsets = offsets
        self.round_base_wins = base_wins
        self.round_feature_wins = feature_wins
        self.round_jackpot_wins = jackpot_wins
        self.round_total_wins = total_wins

    def begin_round(self):
        if self.round_count == self.round_base_wins.shape[0]:
            self._grow_rounds()
        self.round_spin_offsets[self.round_count] = self.spin_count

    def save_spin(
        self,
        feature_trigger_counts,
        feature_wins_by_type,
        feature_spins_by_type,
        base_win,
        jackpot_win,
        jackpot_values_before_feature,
        jackpot_awards,
        jackpot_award_amounts,
        jackpot_values_after_feature,
    ):
        if self.spin_count == self.spin_base_wins.shape[0]:
            self._grow_spins()

        if feature_trigger_counts.shape != self.feature_trigger_counts.shape:
            raise ValueError("Feature trigger counts have an invalid shape")

        feature_count = int(feature_trigger_counts.sum())
        feature_mask = 0
        for feature_index, count in enumerate(feature_trigger_counts):
            if count:
                feature_mask |= 1 << feature_index

        index = self.spin_count
        if feature_count:
            self.spin_feature_session_indices[index] = (
                self.feature_session_count
            )
        self.spin_feature_masks[index] = feature_mask
        feature_win = float(feature_wins_by_type.sum())
        self.spin_base_wins[index] = base_win
        self.spin_feature_wins[index] = feature_win
        self.spin_jackpot_wins[index] = jackpot_win
        self.spin_total_wins[index] = base_win + feature_win + jackpot_win
        self.spin_jackpot_values_before_feature[index] = (
            jackpot_values_before_feature
        )
        self.spin_jackpot_awards[index] = jackpot_awards
        self.spin_jackpot_award_amounts[index] = jackpot_award_amounts
        self.spin_jackpot_values_after_feature[index] = (
            jackpot_values_after_feature
        )

        self.feature_trigger_counts += feature_trigger_counts
        self.feature_win_amounts += feature_wins_by_type
        self.feature_spin_counts += feature_spins_by_type
        self.feature_session_count += feature_count
        self.spin_count += 1

    def finish_round(self, base_win, feature_win, jackpot_win):
        index = self.round_count
        self.round_base_wins[index] = base_win
        self.round_feature_wins[index] = feature_win
        self.round_jackpot_wins[index] = jackpot_win
        self.round_total_wins[index] = base_win + feature_win + jackpot_win
        self.round_count += 1
        self.round_spin_offsets[self.round_count] = self.spin_count


hold_and_spin_storage_spec = [
    # One entry for every stored state inside a respin.
    ("boards", int16[:, :, :]),
    ("coin_masks", boolean[:, :, :]),
    ("feature_types", int8[:]),
    ("feature_positions", int32[:]),
    ("step_remaining_spins", int16[:]),
    ("step_locked_row_indices", int16[:]),
    ("step_collector_meters", int64[:]),

    # Respin -> step mapping and respin-level state transitions.
    ("respin_step_offsets", int64[:]),
    ("respin_remaining_before", int16[:]),
    ("respin_remaining_after", int16[:]),
    ("respin_reset_flags", boolean[:]),
    ("respin_jackpot_overlay_boards", int8[:, :, :]),
    ("respin_jackpot_meters_before", int16[:, :]),
    ("respin_jackpot_meters_after", int16[:, :]),
    ("respin_jackpot_awards", boolean[:, :]),

    # Session -> respin mapping and final feature outcomes.
    ("session_respin_offsets", int64[:]),
    ("session_wins", float64[:]),
    ("session_coin_wins", float64[:]),
    ("session_collector_wins", float64[:]),
    ("session_total_respins", int64[:]),
    ("session_starting_symbols", int16[:, :]),
    ("session_starting_symbol_counts", int16[:]),
    ("session_jackpot_meters", int16[:, :]),
    ("session_jackpot_awards", boolean[:, :]),

    ("step_count", int64),
    ("respin_count", int64),
    ("session_count", int64),
]


@jitclass(hold_and_spin_storage_spec)
class HoldAndSpinStorage:
    """Growing session/respin/step storage for Hold-and-Spin features.

    ``feature_types`` uses ``-2`` for the initial feature window, ``-1`` for
    the landed-symbol state, and bag indices ``0`` through ``5`` for resolved
    Bag steps. ``feature_positions`` is a flattened board position, or ``-1``
    when a step is not associated with one particular Bag symbol.
    """

    def __init__(
        self,
        step_capacity,
        respin_capacity,
        session_capacity,
        num_rows,
        num_reels,
        num_jackpots,
    ):
        if step_capacity < 1:
            raise ValueError("Step capacity must be positive")
        if respin_capacity < 1:
            raise ValueError("Respin capacity must be positive")
        if session_capacity < 1:
            raise ValueError("Session capacity must be positive")
        if num_rows < 1 or num_reels < 1:
            raise ValueError("Board dimensions must be positive")
        if num_jackpots < 1:
            raise ValueError("Jackpot count must be positive")

        self.boards = np.empty(
            (step_capacity, num_rows, num_reels),
            dtype=np.int16,
        )
        self.coin_masks = np.empty(
            (step_capacity, num_rows, num_reels),
            dtype=np.bool_,
        )
        self.feature_types = np.empty(step_capacity, dtype=np.int8)
        self.feature_positions = np.empty(step_capacity, dtype=np.int32)
        self.step_remaining_spins = np.empty(step_capacity, dtype=np.int16)
        self.step_locked_row_indices = np.empty(step_capacity, dtype=np.int16)
        self.step_collector_meters = np.empty(step_capacity, dtype=np.int64)

        self.respin_step_offsets = np.empty(
            respin_capacity + 1,
            dtype=np.int64,
        )
        self.respin_remaining_before = np.empty(
            respin_capacity,
            dtype=np.int16,
        )
        self.respin_remaining_after = np.empty(
            respin_capacity,
            dtype=np.int16,
        )
        self.respin_reset_flags = np.empty(
            respin_capacity,
            dtype=np.bool_,
        )
        self.respin_jackpot_overlay_boards = np.full(
            (respin_capacity, num_rows, num_reels),
            -1,
            dtype=np.int8,
        )
        self.respin_jackpot_meters_before = np.zeros(
            (respin_capacity, num_jackpots),
            dtype=np.int16,
        )
        self.respin_jackpot_meters_after = np.zeros(
            (respin_capacity, num_jackpots),
            dtype=np.int16,
        )
        self.respin_jackpot_awards = np.zeros(
            (respin_capacity, num_jackpots),
            dtype=np.bool_,
        )

        max_starting_symbols = num_rows * num_reels
        self.session_respin_offsets = np.empty(
            session_capacity + 1,
            dtype=np.int64,
        )
        self.session_wins = np.empty(session_capacity, dtype=np.float64)
        self.session_coin_wins = np.empty(
            session_capacity,
            dtype=np.float64,
        )
        self.session_collector_wins = np.empty(
            session_capacity,
            dtype=np.float64,
        )
        self.session_total_respins = np.empty(
            session_capacity,
            dtype=np.int64,
        )
        self.session_starting_symbols = np.full(
            (session_capacity, max_starting_symbols),
            -1,
            dtype=np.int16,
        )
        self.session_starting_symbol_counts = np.empty(
            session_capacity,
            dtype=np.int16,
        )
        self.session_jackpot_meters = np.zeros(
            (session_capacity, num_jackpots),
            dtype=np.int16,
        )
        self.session_jackpot_awards = np.zeros(
            (session_capacity, num_jackpots),
            dtype=np.bool_,
        )

        self.step_count = 0
        self.respin_count = 0
        self.session_count = 0
        self.respin_step_offsets[0] = 0
        self.session_respin_offsets[0] = 0

    def _grow_steps(self):
        old_capacity = self.boards.shape[0]
        new_capacity = old_capacity * 2
        num_rows = self.boards.shape[1]
        num_reels = self.boards.shape[2]
        count = self.step_count

        boards = np.empty(
            (new_capacity, num_rows, num_reels),
            dtype=np.int16,
        )
        coin_masks = np.empty(
            (new_capacity, num_rows, num_reels),
            dtype=np.bool_,
        )
        feature_types = np.empty(new_capacity, dtype=np.int8)
        feature_positions = np.empty(new_capacity, dtype=np.int32)
        step_remaining_spins = np.empty(new_capacity, dtype=np.int16)
        step_locked_row_indices = np.empty(new_capacity, dtype=np.int16)
        step_collector_meters = np.empty(new_capacity, dtype=np.int64)

        boards[:count] = self.boards[:count]
        coin_masks[:count] = self.coin_masks[:count]
        feature_types[:count] = self.feature_types[:count]
        feature_positions[:count] = self.feature_positions[:count]
        step_remaining_spins[:count] = self.step_remaining_spins[:count]
        step_locked_row_indices[:count] = self.step_locked_row_indices[:count]
        step_collector_meters[:count] = self.step_collector_meters[:count]

        self.boards = boards
        self.coin_masks = coin_masks
        self.feature_types = feature_types
        self.feature_positions = feature_positions
        self.step_remaining_spins = step_remaining_spins
        self.step_locked_row_indices = step_locked_row_indices
        self.step_collector_meters = step_collector_meters

    def _grow_respins(self):
        old_capacity = self.respin_remaining_before.shape[0]
        new_capacity = old_capacity * 2
        count = self.respin_count

        respin_step_offsets = np.empty(new_capacity + 1, dtype=np.int64)
        respin_remaining_before = np.empty(new_capacity, dtype=np.int16)
        respin_remaining_after = np.empty(new_capacity, dtype=np.int16)
        respin_reset_flags = np.empty(new_capacity, dtype=np.bool_)
        num_rows = self.respin_jackpot_overlay_boards.shape[1]
        num_reels = self.respin_jackpot_overlay_boards.shape[2]
        num_jackpots = self.respin_jackpot_meters_before.shape[1]
        respin_jackpot_overlay_boards = np.full(
            (new_capacity, num_rows, num_reels),
            -1,
            dtype=np.int8,
        )
        respin_jackpot_meters_before = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.int16,
        )
        respin_jackpot_meters_after = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.int16,
        )
        respin_jackpot_awards = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.bool_,
        )

        respin_step_offsets[:count + 1] = self.respin_step_offsets[:count + 1]
        respin_remaining_before[:count] = self.respin_remaining_before[:count]
        respin_remaining_after[:count] = self.respin_remaining_after[:count]
        respin_reset_flags[:count] = self.respin_reset_flags[:count]
        respin_jackpot_overlay_boards[:count] = (
            self.respin_jackpot_overlay_boards[:count]
        )
        respin_jackpot_meters_before[:count] = (
            self.respin_jackpot_meters_before[:count]
        )
        respin_jackpot_meters_after[:count] = (
            self.respin_jackpot_meters_after[:count]
        )
        respin_jackpot_awards[:count] = self.respin_jackpot_awards[:count]

        self.respin_step_offsets = respin_step_offsets
        self.respin_remaining_before = respin_remaining_before
        self.respin_remaining_after = respin_remaining_after
        self.respin_reset_flags = respin_reset_flags
        self.respin_jackpot_overlay_boards = respin_jackpot_overlay_boards
        self.respin_jackpot_meters_before = respin_jackpot_meters_before
        self.respin_jackpot_meters_after = respin_jackpot_meters_after
        self.respin_jackpot_awards = respin_jackpot_awards

    def _grow_sessions(self):
        old_capacity = self.session_wins.shape[0]
        new_capacity = old_capacity * 2
        max_starting_symbols = self.session_starting_symbols.shape[1]
        num_jackpots = self.session_jackpot_meters.shape[1]
        count = self.session_count

        session_respin_offsets = np.empty(new_capacity + 1, dtype=np.int64)
        session_wins = np.empty(new_capacity, dtype=np.float64)
        session_coin_wins = np.empty(new_capacity, dtype=np.float64)
        session_collector_wins = np.empty(new_capacity, dtype=np.float64)
        session_total_respins = np.empty(new_capacity, dtype=np.int64)
        session_starting_symbols = np.full(
            (new_capacity, max_starting_symbols),
            -1,
            dtype=np.int16,
        )
        session_starting_symbol_counts = np.empty(
            new_capacity,
            dtype=np.int16,
        )
        session_jackpot_meters = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.int16,
        )
        session_jackpot_awards = np.zeros(
            (new_capacity, num_jackpots),
            dtype=np.bool_,
        )

        session_respin_offsets[:count + 1] = self.session_respin_offsets[
            :count + 1
        ]
        session_wins[:count] = self.session_wins[:count]
        session_coin_wins[:count] = self.session_coin_wins[:count]
        session_collector_wins[:count] = self.session_collector_wins[:count]
        session_total_respins[:count] = self.session_total_respins[:count]
        session_starting_symbols[:count] = self.session_starting_symbols[:count]
        session_starting_symbol_counts[:count] = (
            self.session_starting_symbol_counts[:count]
        )
        session_jackpot_meters[:count] = self.session_jackpot_meters[:count]
        session_jackpot_awards[:count] = self.session_jackpot_awards[:count]

        self.session_respin_offsets = session_respin_offsets
        self.session_wins = session_wins
        self.session_coin_wins = session_coin_wins
        self.session_collector_wins = session_collector_wins
        self.session_total_respins = session_total_respins
        self.session_starting_symbols = session_starting_symbols
        self.session_starting_symbol_counts = session_starting_symbol_counts
        self.session_jackpot_meters = session_jackpot_meters
        self.session_jackpot_awards = session_jackpot_awards

    def begin_session(self, starting_bag_symbols):
        if self.session_count == self.session_wins.shape[0]:
            self._grow_sessions()

        index = self.session_count
        self.session_respin_offsets[index] = self.respin_count
        self.session_starting_symbols[index] = -1
        symbol_count = min(
            len(starting_bag_symbols),
            self.session_starting_symbols.shape[1],
        )
        self.session_starting_symbol_counts[index] = symbol_count
        for symbol_index in range(symbol_count):
            self.session_starting_symbols[index, symbol_index] = (
                starting_bag_symbols[symbol_index]
            )

    def finish_session(
        self,
        session_win,
        coin_win,
        collector_win,
        jackpot_meters,
        jackpot_awards,
    ):
        index = self.session_count
        self.session_wins[index] = session_win
        self.session_coin_wins[index] = coin_win
        self.session_collector_wins[index] = collector_win
        self.session_jackpot_meters[index] = jackpot_meters
        self.session_jackpot_awards[index] = jackpot_awards
        self.session_total_respins[index] = (
            self.respin_count - self.session_respin_offsets[index]
        )
        self.session_count += 1
        self.session_respin_offsets[self.session_count] = self.respin_count

    def begin_respin(self, remaining_spins):
        if self.respin_count == self.respin_remaining_before.shape[0]:
            self._grow_respins()

        index = self.respin_count
        self.respin_step_offsets[index] = self.step_count
        self.respin_remaining_before[index] = remaining_spins

    def finish_respin(self, remaining_spins, reset_flag):
        index = self.respin_count
        self.respin_remaining_after[index] = remaining_spins
        self.respin_reset_flags[index] = reset_flag
        self.respin_count += 1
        self.respin_step_offsets[self.respin_count] = self.step_count

    def save_jackpot_respin(
        self,
        overlay_board,
        meters_before,
        meters_after,
        jackpot_awards,
    ):
        index = self.respin_count
        self.respin_jackpot_overlay_boards[index] = overlay_board
        self.respin_jackpot_meters_before[index] = meters_before
        self.respin_jackpot_meters_after[index] = meters_after
        self.respin_jackpot_awards[index] = jackpot_awards

    def save_step(
        self,
        board,
        coin_mask,
        feature_type,
        feature_position,
        remaining_spins,
        locked_row_idx,
        collector_meter,
    ):
        if self.step_count == self.boards.shape[0]:
            self._grow_steps()

        index = self.step_count
        self.boards[index] = board
        self.coin_masks[index] = coin_mask.reshape(board.shape)
        self.feature_types[index] = feature_type
        self.feature_positions[index] = feature_position
        self.step_remaining_spins[index] = remaining_spins
        self.step_locked_row_indices[index] = locked_row_idx
        self.step_collector_meters[index] = collector_meter
        self.step_count += 1


def _npz_library_path(filename):
    """Resolve a filename inside the package NPZ-library directory."""
    filename = Path(filename)
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("NPZ filename must not contain a directory path")
    if filename.suffix == "":
        filename = filename.with_suffix(".npz")
    elif filename.suffix.lower() != ".npz":
        raise ValueError("NPZ filename must use the .npz extension")

    NPZ_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return NPZ_LIBRARY_DIR / filename


def _write_payload(payload, filename, overwrite):
    output_path = _npz_library_path(filename)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"NPZ output already exists: {output_path}")

    np.savez_compressed(output_path, **payload)
    return output_path


def write_npz(storage, filename, overwrite=False):
    """Write the populated paid spins and outer rounds from one shard."""
    spin_count = int(storage.spin_count)
    round_count = int(storage.round_count)

    payload = {
        "format_version": np.array(NPZ_FORMAT_VERSION, dtype=np.int16),
        "spin_count": np.array(spin_count, dtype=np.int64),
        "round_count": np.array(round_count, dtype=np.int64),
        "boards": storage.boards[:spin_count],
        "coin_value_boards": storage.coin_value_boards[:spin_count],
        "jackpot_overlay_boards": storage.jackpot_overlay_boards[:spin_count],
        "jackpot_values_before": storage.jackpot_values_before[:spin_count],
        "jackpot_values_after": storage.jackpot_values_after[:spin_count],
        "jackpot_increment_counts": storage.jackpot_increment_counts[
            :spin_count
        ],
        "wins": storage.wins[:spin_count],
        "line_wins": storage.line_wins[:spin_count],
        "collect_wins": storage.collect_wins[:spin_count],
        "symbol_wins": storage.symbol_wins[:spin_count],
        "symbol_hit_counts": storage.symbol_hit_counts[:spin_count],
        "winning_line_numbers": storage.winning_line_numbers[:spin_count],
        "line_winning_symbols": storage.line_winning_symbols[:spin_count],
        "line_symbol_counts": storage.line_symbol_counts[:spin_count],
        "line_win_amounts": storage.line_win_amounts[:spin_count],
        "spin_triggers": storage.spin_triggers[:spin_count],
        "collector_counts": storage.collector_counts[:spin_count],
        "round_spin_offsets": storage.round_spin_offsets[:round_count + 1],
        "round_wins": storage.round_wins[:round_count],
        "round_triggers": storage.round_triggers[:round_count],
    }
    return _write_payload(payload, filename, overwrite)


def _load_npz(filename):
    input_path = _npz_library_path(filename)
    if not input_path.is_file():
        raise FileNotFoundError(f"NPZ input does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else 0
        if version != NPZ_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported NPZ format version {version} in {input_path.name}"
            )
        missing = [key for key in _NPZ_ARRAY_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"NPZ input {input_path.name} is missing arrays: {missing}"
            )
        arrays = {
            key: np.array(data[key], copy=True)
            for key in _NPZ_ARRAY_KEYS
        }

    spin_count = arrays["boards"].shape[0]
    round_count = arrays["round_wins"].shape[0]
    for key in _SPIN_ARRAY_KEYS:
        if arrays[key].shape[0] != spin_count:
            raise ValueError(
                f"Spin array {key} has an invalid length in {input_path.name}"
            )
    for key in _ROUND_ARRAY_KEYS:
        if arrays[key].shape[0] != round_count:
            raise ValueError(
                f"Round array {key} has an invalid length in {input_path.name}"
            )

    offsets = arrays["round_spin_offsets"]
    if offsets.shape != (round_count + 1,):
        raise ValueError(f"Invalid round offsets in {input_path.name}")
    if offsets[0] != 0:
        raise ValueError(
            f"Round offsets must start at zero in {input_path.name}"
        )
    if offsets[-1] != spin_count:
        raise ValueError(
            f"Round offsets do not cover all spins in {input_path.name}"
        )
    if np.any(np.diff(offsets) < 1):
        raise ValueError(
            f"Every round must contain at least one spin in {input_path.name}"
        )

    return input_path, arrays, spin_count, round_count


def merge_npz(filenames, output_filename, overwrite=False):
    """Merge compatible base-game shards and rebase round-spin offsets."""
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    else:
        filenames = list(filenames)
    if len(filenames) == 0:
        raise ValueError("At least one NPZ input is required")

    shards = [_load_npz(filename) for filename in filenames]
    reference = shards[0][1]
    shape_keys = (
        "boards",
        "coin_value_boards",
        "jackpot_overlay_boards",
        "jackpot_values_before",
        "jackpot_values_after",
        "jackpot_increment_counts",
        "symbol_wins",
        "symbol_hit_counts",
        "winning_line_numbers",
        "line_winning_symbols",
        "line_symbol_counts",
        "line_win_amounts",
    )
    for input_path, arrays, _, _ in shards[1:]:
        for key in shape_keys:
            if arrays[key].shape[1:] != reference[key].shape[1:]:
                raise ValueError(
                    f"Array shape mismatch for {key} in {input_path.name}"
                )

    payload = {
        key: np.concatenate([arrays[key] for _, arrays, _, _ in shards])
        for key in _SPIN_ARRAY_KEYS + _ROUND_ARRAY_KEYS
    }

    round_offsets = [np.array([0], dtype=np.int64)]
    spin_base = 0
    round_total = 0
    for _, arrays, spin_count, round_count in shards:
        round_offsets.append(arrays["round_spin_offsets"][1:] + spin_base)
        spin_base += spin_count
        round_total += round_count

    payload["round_spin_offsets"] = np.concatenate(round_offsets)
    payload["format_version"] = np.array(NPZ_FORMAT_VERSION, dtype=np.int16)
    payload["spin_count"] = np.array(spin_base, dtype=np.int64)
    payload["round_count"] = np.array(round_total, dtype=np.int64)
    return _write_payload(payload, output_filename, overwrite)


_HOLD_STEP_KEYS = (
    "boards",
    "coin_masks",
    "feature_types",
    "feature_positions",
    "step_remaining_spins",
    "step_locked_row_indices",
    "step_collector_meters",
)

_HOLD_RESPIN_KEYS = (
    "respin_remaining_before",
    "respin_remaining_after",
    "respin_reset_flags",
    "respin_jackpot_overlay_boards",
    "respin_jackpot_meters_before",
    "respin_jackpot_meters_after",
    "respin_jackpot_awards",
)

_HOLD_SESSION_KEYS = (
    "session_wins",
    "session_coin_wins",
    "session_collector_wins",
    "session_total_respins",
    "session_starting_symbols",
    "session_starting_symbol_counts",
    "session_jackpot_meters",
    "session_jackpot_awards",
)

_HOLD_NPZ_ARRAY_KEYS = (
    _HOLD_STEP_KEYS
    + ("respin_step_offsets",)
    + _HOLD_RESPIN_KEYS
    + ("session_respin_offsets",)
    + _HOLD_SESSION_KEYS
)


def write_hold_and_spin_npz(storage, filename, overwrite=False):
    """Write one populated Hold-and-Spin storage shard."""
    step_count = int(storage.step_count)
    respin_count = int(storage.respin_count)
    session_count = int(storage.session_count)

    payload = {
        "format_version": np.array(NPZ_FORMAT_VERSION, dtype=np.int16),
        "storage_kind": np.array("hold_and_spin"),
        "step_count": np.array(step_count, dtype=np.int64),
        "respin_count": np.array(respin_count, dtype=np.int64),
        "session_count": np.array(session_count, dtype=np.int64),
        "boards": storage.boards[:step_count],
        "coin_masks": storage.coin_masks[:step_count],
        "feature_types": storage.feature_types[:step_count],
        "feature_positions": storage.feature_positions[:step_count],
        "step_remaining_spins": storage.step_remaining_spins[:step_count],
        "step_locked_row_indices": storage.step_locked_row_indices[:step_count],
        "step_collector_meters": storage.step_collector_meters[:step_count],
        "respin_step_offsets": storage.respin_step_offsets[:respin_count + 1],
        "respin_remaining_before": storage.respin_remaining_before[:respin_count],
        "respin_remaining_after": storage.respin_remaining_after[:respin_count],
        "respin_reset_flags": storage.respin_reset_flags[:respin_count],
        "respin_jackpot_overlay_boards": (
            storage.respin_jackpot_overlay_boards[:respin_count]
        ),
        "respin_jackpot_meters_before": (
            storage.respin_jackpot_meters_before[:respin_count]
        ),
        "respin_jackpot_meters_after": (
            storage.respin_jackpot_meters_after[:respin_count]
        ),
        "respin_jackpot_awards": storage.respin_jackpot_awards[:respin_count],
        "session_respin_offsets": storage.session_respin_offsets[
            :session_count + 1
        ],
        "session_wins": storage.session_wins[:session_count],
        "session_coin_wins": storage.session_coin_wins[:session_count],
        "session_collector_wins": storage.session_collector_wins[:session_count],
        "session_total_respins": storage.session_total_respins[:session_count],
        "session_starting_symbols": storage.session_starting_symbols[
            :session_count
        ],
        "session_starting_symbol_counts": (
            storage.session_starting_symbol_counts[:session_count]
        ),
        "session_jackpot_meters": storage.session_jackpot_meters[
            :session_count
        ],
        "session_jackpot_awards": storage.session_jackpot_awards[
            :session_count
        ],
    }
    return _write_payload(payload, filename, overwrite)


def write_full_game_npz(storage, filename, overwrite=False):
    """Write the combined base/feature/jackpot results for full-game play."""
    spin_count = int(storage.spin_count)
    round_count = int(storage.round_count)
    payload = {
        "format_version": np.array(NPZ_FORMAT_VERSION, dtype=np.int16),
        "storage_kind": np.array("full_game"),
        "spin_count": np.array(spin_count, dtype=np.int64),
        "round_count": np.array(round_count, dtype=np.int64),
        "spin_feature_session_indices": (
            storage.spin_feature_session_indices[:spin_count]
        ),
        "spin_base_wins": storage.spin_base_wins[:spin_count],
        "spin_feature_wins": storage.spin_feature_wins[:spin_count],
        "spin_jackpot_wins": storage.spin_jackpot_wins[:spin_count],
        "spin_total_wins": storage.spin_total_wins[:spin_count],
        "spin_jackpot_values_before_feature": (
            storage.spin_jackpot_values_before_feature[:spin_count]
        ),
        "spin_jackpot_awards": storage.spin_jackpot_awards[:spin_count],
        "spin_jackpot_award_amounts": (
            storage.spin_jackpot_award_amounts[:spin_count]
        ),
        "spin_jackpot_values_after_feature": (
            storage.spin_jackpot_values_after_feature[:spin_count]
        ),
        "round_spin_offsets": storage.round_spin_offsets[:round_count + 1],
        "round_base_wins": storage.round_base_wins[:round_count],
        "round_feature_wins": storage.round_feature_wins[:round_count],
        "round_jackpot_wins": storage.round_jackpot_wins[:round_count],
        "round_total_wins": storage.round_total_wins[:round_count],
    }
    if hasattr(storage, "spin_feature_masks"):
        payload.update(
            {
                "spin_feature_masks": storage.spin_feature_masks[:spin_count],
                "feature_trigger_counts": storage.feature_trigger_counts,
                "feature_win_amounts": storage.feature_win_amounts,
                "feature_spin_counts": storage.feature_spin_counts,
            }
        )
    return _write_payload(payload, filename, overwrite)


def _load_hold_and_spin_npz(filename):
    input_path = _npz_library_path(filename)
    if not input_path.is_file():
        raise FileNotFoundError(f"NPZ input does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else 0
        storage_kind = str(data["storage_kind"]) if "storage_kind" in data else ""
        if version != NPZ_FORMAT_VERSION or storage_kind != "hold_and_spin":
            raise ValueError(f"Invalid Hold-and-Spin NPZ: {input_path.name}")
        missing = [key for key in _HOLD_NPZ_ARRAY_KEYS if key not in data]
        if missing:
            raise ValueError(
                f"NPZ input {input_path.name} is missing arrays: {missing}"
            )
        arrays = {
            key: np.array(data[key], copy=True)
            for key in _HOLD_NPZ_ARRAY_KEYS
        }

    step_count = arrays["boards"].shape[0]
    respin_count = arrays["respin_remaining_before"].shape[0]
    session_count = arrays["session_wins"].shape[0]

    for key in _HOLD_STEP_KEYS:
        if arrays[key].shape[0] != step_count:
            raise ValueError(f"Invalid step array {key} in {input_path.name}")
    for key in _HOLD_RESPIN_KEYS:
        if arrays[key].shape[0] != respin_count:
            raise ValueError(f"Invalid respin array {key} in {input_path.name}")
    for key in _HOLD_SESSION_KEYS:
        if arrays[key].shape[0] != session_count:
            raise ValueError(f"Invalid session array {key} in {input_path.name}")

    respin_offsets = arrays["respin_step_offsets"]
    session_offsets = arrays["session_respin_offsets"]
    if (
        respin_offsets.shape != (respin_count + 1,)
        or respin_offsets[0] != 0
        or respin_offsets[-1] != step_count
        or np.any(np.diff(respin_offsets) < 1)
    ):
        raise ValueError(f"Invalid respin-step offsets in {input_path.name}")
    if (
        session_offsets.shape != (session_count + 1,)
        or session_offsets[0] != 0
        or session_offsets[-1] != respin_count
        or np.any(np.diff(session_offsets) < 1)
    ):
        raise ValueError(f"Invalid session-respin offsets in {input_path.name}")

    return input_path, arrays, step_count, respin_count, session_count


def merge_hold_and_spin_npz(filenames, output_filename, overwrite=False):
    """Merge feature shards and rebase both Hold-and-Spin offset levels."""
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    else:
        filenames = list(filenames)
    if len(filenames) == 0:
        raise ValueError("At least one NPZ input is required")

    shards = [_load_hold_and_spin_npz(filename) for filename in filenames]
    reference = shards[0][1]
    for input_path, arrays, _, _, _ in shards[1:]:
        for key in (
            "boards",
            "coin_masks",
            "respin_jackpot_overlay_boards",
            "respin_jackpot_meters_before",
            "respin_jackpot_meters_after",
            "respin_jackpot_awards",
            "session_starting_symbols",
            "session_jackpot_meters",
            "session_jackpot_awards",
        ):
            if arrays[key].shape[1:] != reference[key].shape[1:]:
                raise ValueError(
                    f"Array shape mismatch for {key} in {input_path.name}"
                )

    payload = {
        key: np.concatenate([arrays[key] for _, arrays, _, _, _ in shards])
        for key in _HOLD_STEP_KEYS + _HOLD_RESPIN_KEYS + _HOLD_SESSION_KEYS
    }

    respin_offsets = [np.array([0], dtype=np.int64)]
    session_offsets = [np.array([0], dtype=np.int64)]
    step_base = 0
    respin_base = 0
    session_total = 0
    for _, arrays, step_count, respin_count, session_count in shards:
        respin_offsets.append(arrays["respin_step_offsets"][1:] + step_base)
        session_offsets.append(
            arrays["session_respin_offsets"][1:] + respin_base
        )
        step_base += step_count
        respin_base += respin_count
        session_total += session_count

    payload["respin_step_offsets"] = np.concatenate(respin_offsets)
    payload["session_respin_offsets"] = np.concatenate(session_offsets)
    payload["format_version"] = np.array(NPZ_FORMAT_VERSION, dtype=np.int16)
    payload["storage_kind"] = np.array("hold_and_spin")
    payload["step_count"] = np.array(step_base, dtype=np.int64)
    payload["respin_count"] = np.array(respin_base, dtype=np.int64)
    payload["session_count"] = np.array(session_total, dtype=np.int64)
    return _write_payload(payload, output_filename, overwrite)
