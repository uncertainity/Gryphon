from typing import NamedTuple
from pathlib import Path

import numpy as np

from .reels import REEL_DICT


_REELS_ROOT = Path(__file__).resolve().parents[2] / "Reels"


class JackpotConfig(NamedTuple):
    """Shared monetary rules for the four progressive jackpots."""

    jackpot_types: np.ndarray
    seed_values: np.ndarray
    increment_values: np.ndarray
    cap_multiplier: float


class BaseJackpotOverlayConfig(NamedTuple):
    """Rules for generating jackpot overlays on a base-game window."""

    count_values: np.ndarray
    count_probabilities: np.ndarray
    jackpot_type_probabilities: np.ndarray
    eligible_symbols: np.ndarray


JACKPOT_CONFIG = JackpotConfig(
    jackpot_types=np.arange(4, dtype=np.int8),
    # Placeholder x-bet values. These are intentionally isolated here so the
    # final math settings can replace them without changing the game logic.
    seed_values=np.array([2.0, 10.0, 100.0, 10_000.0], dtype=np.float64),
    increment_values=np.array([0.1, 0.5, 5.0, 50.0], dtype=np.float64),
    cap_multiplier=2.0,
)


BASE_JACKPOT_OVERLAY_CONFIG = BaseJackpotOverlayConfig(
    # Both the available counts and their weights are tuning variables. The
    # current 0-4 range and probabilities are placeholders.
    count_values=np.array([0, 1, 2, 3, 4], dtype=np.int8),
    count_probabilities=np.array(
        [0.70, 0.20, 0.07, 0.02, 0.01],
        dtype=np.float64,
    ),
    jackpot_type_probabilities=np.full(4, 0.25, dtype=np.float64),
    # Paying symbols H1-L6 plus Wild. Coin/Collect/feature symbols are not
    # eligible because the jackpot is an overlay, not a replacement symbol.
    eligible_symbols=np.arange(12, dtype=np.int16),
)


class BaseGameConfig(NamedTuple):
    reelset_path: str
    reelset_probabilities: np.ndarray
    num_rows: int
    num_reels: int
    reel_padding_symbol: int
    num_paying_symbols: int
    wild_symbol: int
    free_game_symbols: np.ndarray
    free_game_trigger_count: int
    coin_symbol: int
    collect_symbol: int
    max_active_collectors: int
    coin_credit_values: np.ndarray
    coin_credit_value_probabilities: np.ndarray


BASE_GAME_CONFIG = BaseGameConfig(
    reelset_path=str(_REELS_ROOT / "Base_Game"),
    reelset_probabilities=np.array([0.023, 0.977], dtype=np.float64),
    num_rows=3,
    num_reels=5,
    reel_padding_symbol=-1,
    num_paying_symbols=12,
    wild_symbol=REEL_DICT["WD"],
    free_game_symbols=np.array(
        [
            REEL_DICT["SC1"],
            REEL_DICT["SC2"],
            REEL_DICT["SC3"],
            REEL_DICT["SC4"],
            REEL_DICT["SC5"],
            REEL_DICT["SC6"],
        ],
        dtype=np.int16,
    ),
    free_game_trigger_count=1,
    coin_symbol=REEL_DICT["COIN"],
    collect_symbol=REEL_DICT["COLLECT"],
    max_active_collectors=75,
    coin_credit_values=np.array(
        [1, 2, 3, 5, 10, 15, 20, 25, 50],
        dtype=np.int16,
    ),
    coin_credit_value_probabilities=np.array(
        [0.405, 0.28, 0.14, 0.08, 0.045, 0.025, 0.015, 0.008, 0.002],
        dtype=np.float64,
    ),
)


BASE_PAY_TABLE = np.array(
    [
        [0.00, 0.00, 0.05, 0.25, 0.80],
        [0.00, 0.00, 0.00, 0.20, 0.50],
        [0.00, 0.00, 0.00, 0.20, 0.40],
        [0.00, 0.00, 0.00, 0.20, 0.25],
        [0.00, 0.00, 0.00, 0.15, 0.20],
        [0.00, 0.00, 0.00, 0.10, 0.30],
        [0.00, 0.00, 0.00, 0.10, 0.30],
        [0.00, 0.00, 0.00, 0.10, 0.30],
        [0.00, 0.00, 0.00, 0.10, 0.30],
        [0.00, 0.00, 0.00, 0.10, 0.25],
        [0.00, 0.00, 0.00, 0.10, 0.20],
        [0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float64,
)


class HoldAndSpinConfig(NamedTuple):
    num_rows: int
    num_reels: int
    starting_rows: int
    starting_scatter_counts: np.ndarray
    starting_scatter_count_probabilities: np.ndarray
    starting_coin_counts: np.ndarray
    starting_coin_count_probabilities: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    splitter_symbol: int
    grower_symbol: int
    booster_symbol: int
    multiplier_symbol: int
    collector_symbol: int
    expansion_symbol: int
    bag_symbols: np.ndarray
    bag_resolution_order: np.ndarray
    bag_symbol_actions: np.ndarray
    coin_types: np.ndarray
    coin_type_probabilities: np.ndarray
    p_coin_locked: float
    p_coin_unlocked: float
    respin_reset_count: int
    max_coin_value: int
    splitter_source_counts: np.ndarray
    splitter_source_count_probabilities: np.ndarray
    splitter_copy_counts: np.ndarray
    splitter_copy_count_probabilities: np.ndarray
    grower_coin_counts: np.ndarray
    grower_coin_count_probabilities: np.ndarray
    grower_increment_values: np.ndarray
    grower_increment_probabilities: np.ndarray
    booster_increment_values: np.ndarray
    booster_increment_probabilities: np.ndarray
    multiplier_values: np.ndarray
    multiplier_probabilities: np.ndarray
    max_collector_events: int
    rows_unlocked_per_expansion: int
    jackpot_token_probability: float
    jackpot_type_probabilities: np.ndarray
    jackpot_collection_targets: np.ndarray
    max_jackpot_tokens_per_respin: int


HOLD_AND_SPIN_CONFIG = HoldAndSpinConfig(
    num_rows=6,
    num_reels=5,
    starting_rows=3,
    starting_scatter_counts=np.array([1, 2], dtype=np.int8),
    starting_scatter_count_probabilities=np.array(
        [0.80, 0.20],
        dtype=np.float64,
    ),
    starting_coin_counts=np.array([1, 2, 3], dtype=np.int8),
    starting_coin_count_probabilities=np.array(
        [0.60, 0.30, 0.10],
        dtype=np.float64,
    ),
    coin_values=np.array([2, 5, 10, 20, 25], dtype=np.int16),
    coin_value_probabilities=np.array(
        [0.52, 0.30, 0.10, 0.05, 0.03],
        dtype=np.float64,
    ),
    splitter_symbol=REEL_DICT["SC1"],
    grower_symbol=REEL_DICT["SC2"],
    booster_symbol=REEL_DICT["SC3"],
    multiplier_symbol=REEL_DICT["SC4"],
    collector_symbol=REEL_DICT["SC5"],
    expansion_symbol=REEL_DICT["SC6"],
    bag_symbols=np.array(
        [
            REEL_DICT["SC1"],
            REEL_DICT["SC2"],
            REEL_DICT["SC3"],
            REEL_DICT["SC4"],
            REEL_DICT["SC5"],
            REEL_DICT["SC6"],
        ],
        dtype=np.int16,
    ),
    # Expansion, Splitter, Booster, Grower, Multiplier, Collector.
    bag_resolution_order=np.array([5, 0, 2, 1, 3, 4], dtype=np.int8),
    # 0 = remain, 1 = disappear, 2 = convert to a credit coin.
    bag_symbol_actions=np.array([2, 0, 1, 1, 1, 2], dtype=np.int8),
    coin_types=np.array(
        [
            REEL_DICT["COIN"],
            REEL_DICT["SC1"],
            REEL_DICT["SC2"],
            REEL_DICT["SC3"],
            REEL_DICT["SC4"],
            REEL_DICT["SC5"],
            REEL_DICT["SC6"],
        ],
        dtype=np.int16,
    ),
    coin_type_probabilities=np.array(
        [0.600, 0.055, 0.105, 0.090, 0.060, 0.050, 0.040],
        dtype=np.float64,
    ),
    p_coin_locked=0.03,
    p_coin_unlocked=0.05,
    respin_reset_count=3,
    max_coin_value=15000,
    splitter_source_counts=np.array([1, 2, 3], dtype=np.int8),
    splitter_source_count_probabilities=np.array(
        [0.72, 0.21, 0.07],
        dtype=np.float64,
    ),
    splitter_copy_counts=np.array([1, 2, 3], dtype=np.int8),
    splitter_copy_count_probabilities=np.array(
        [0.76, 0.19, 0.05],
        dtype=np.float64,
    ),
    grower_coin_counts=np.array([1, 2, 3], dtype=np.int8),
    grower_coin_count_probabilities=np.array(
        [0.70, 0.23, 0.07],
        dtype=np.float64,
    ),
    grower_increment_values=np.array([1, 2, 5], dtype=np.int16),
    grower_increment_probabilities=np.array(
        [0.74, 0.21, 0.05],
        dtype=np.float64,
    ),
    booster_increment_values=np.array([1, 2, 5, 10], dtype=np.int16),
    booster_increment_probabilities=np.array(
        [0.68, 0.22, 0.08, 0.02],
        dtype=np.float64,
    ),
    multiplier_values=np.array([2, 3, 4], dtype=np.int8),
    multiplier_probabilities=np.array(
        [0.76, 0.19, 0.05],
        dtype=np.float64,
    ),
    max_collector_events=3,
    rows_unlocked_per_expansion=1,
    # Placeholder free-game jackpot-token settings. Tokens attach only to
    # naturally landed, visible QHs and do not alter their credit values.
    jackpot_token_probability=0.16,
    jackpot_type_probabilities=np.full(4, 0.25, dtype=np.float64),
    jackpot_collection_targets=np.full(4, 3, dtype=np.int16),
    max_jackpot_tokens_per_respin=1,
)


class FullGameConfig(NamedTuple):
    """Composition of the independently tunable full-game rule sets."""

    base_game: BaseGameConfig
    base_jackpot_overlay: BaseJackpotOverlayConfig
    free_game: HoldAndSpinConfig
    jackpots: JackpotConfig


FULL_GAME_CONFIG = FullGameConfig(
    base_game=BASE_GAME_CONFIG,
    base_jackpot_overlay=BASE_JACKPOT_OVERLAY_CONFIG,
    free_game=HOLD_AND_SPIN_CONFIG,
    jackpots=JACKPOT_CONFIG,
)


PAY_LINES = np.array(
    [
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
        [0, 6, 12, 8, 4],
        [10, 6, 2, 8, 14],
        [0, 1, 7, 13, 14],
        [10, 11, 7, 3, 4],
        [5, 1, 2, 3, 9],
        [5, 11, 12, 13, 9],
        [0, 6, 7, 8, 4],
        [10, 6, 7, 8, 14],
        [5, 1, 7, 13, 9],
        [5, 11, 7, 3, 9],
        [0, 6, 2, 8, 4],
        [10, 6, 12, 8, 14],
        [5, 6, 2, 8, 9],
        [5, 6, 12, 8, 9],
        [0, 1, 12, 3, 4],
        [10, 11, 2, 13, 14],
        [0, 11, 12, 13, 4],
    ],
    dtype=np.int16,
)
