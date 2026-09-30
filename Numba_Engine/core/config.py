from typing import NamedTuple
from pathlib import Path

import numpy as np

from .reels import REEL_DICT


_REELS_ROOT = Path(__file__).resolve().parents[2] / "Reels"


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
    sc_symbol: int
    scatter_feature_symbols: np.ndarray
    scatter_feature_symbol_probabilities: np.ndarray
    combo_feature_symbols: np.ndarray
    combo_feature_probability: float
    coin_symbol: int
    collect_symbol: int
    max_active_collectors: int
    coin_drop_probabilities: np.ndarray
    coin_drop_counts: np.ndarray
    coin_drop_count_probabilities: np.ndarray
    coin_credit_values: np.ndarray
    coin_credit_value_probabilities: np.ndarray


BASE_GAME_CONFIG = BaseGameConfig(
    reelset_path=str(_REELS_ROOT / "Base_Game"),
    reelset_probabilities=np.array([0.50, 0.50], dtype=np.float64),
    num_rows=3,
    num_reels=5,
    reel_padding_symbol=-1,
    num_paying_symbols=12,
    wild_symbol=REEL_DICT["WD"],
    free_game_symbols=np.array(
        [REEL_DICT["SC"]],
        dtype=np.int16,
    ),
    free_game_trigger_count=1,
    sc_symbol=REEL_DICT["SC"],
    scatter_feature_symbols=np.array(
        [
            REEL_DICT["SC1"],
            REEL_DICT["SC2"],
            REEL_DICT["SC3"],
            REEL_DICT["SC4"],
            REEL_DICT["SC5"],
            REEL_DICT["SC6"],
            REEL_DICT["SC7"],
        ],
        dtype=np.int16,
    ),
    scatter_feature_symbol_probabilities=np.array(
        [0.145, 0.145, 0.145, 0.145, 0.145, 0.145, 0.130],
        dtype=np.float64,
    ),
    combo_feature_symbols=np.array(
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
    combo_feature_probability=1.0,
    coin_symbol=REEL_DICT["COIN"],
    collect_symbol=REEL_DICT["COLLECT"],
    max_active_collectors=75,
    coin_drop_probabilities=np.array(
        [0.30, 0.45, 0.65],
        dtype=np.float64,
    ),
    coin_drop_counts=np.array([2, 3, 4, 5, 6], dtype=np.int16),
    coin_drop_count_probabilities=np.array(
        [0.08, 0.16, 0.34, 0.28, 0.14],
        dtype=np.float64,
    ),
    coin_credit_values=np.array(
        [0.2, 0.3, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5],
        dtype=np.float64,
    ),
    coin_credit_value_probabilities=np.array(
        [0.030, 0.040, 0.070, 0.100, 0.100, 0.150, 0.170, 0.170, 0.120, 0.050],
        dtype=np.float64,
    ),
)


class JackpotTokenConfig(NamedTuple):
    symbols: np.ndarray
    collection_targets: np.ndarray
    awards: np.ndarray


JACKPOT_TOKEN_CONFIG = JackpotTokenConfig(
    symbols=np.array(
        [
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    collection_targets=np.array([3, 3, 3, 3], dtype=np.int16),
    awards=np.array([10.0, 25.0, 100.0, 1000.0], dtype=np.float64),
)


class FeatureRtpConfig(NamedTuple):
    target_total_rtp: float
    base_rtp_locked: float
    target_feature_rtp: float
    overall_feature_probability: float
    requested_single_feature_probability: float
    implemented_single_feature_probability: float
    combo_feature_probability: float
    feature_payout_multiplier: float


FEATURE_RTP_CONFIG = FeatureRtpConfig(
    target_total_rtp=0.94,
    base_rtp_locked=0.5517462399913647,
    target_feature_rtp=0.3882537600086353,
    overall_feature_probability=0.01,
    requested_single_feature_probability=1.0 / 600.0,
    implemented_single_feature_probability=1.0 / 625.0,
    combo_feature_probability=1.0 / 2500.0,
    feature_payout_multiplier=1.086911,
)


class ExpansionFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    starting_unlocked_rows: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    go_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    unlocked_landing_symbols: np.ndarray
    unlocked_landing_probabilities: np.ndarray
    locked_landing_symbols: np.ndarray
    locked_landing_probabilities: np.ndarray
    max_go_symbols: int
    max_locked_go_symbols: int
    starting_extra_coin_count: int


FEATURE_COIN_VALUES = np.array(
    [0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    dtype=np.float64,
)
FEATURE_COIN_VALUE_PROBABILITIES = np.array(
    [0.035, 0.040, 0.060, 0.075, 0.090, 0.130, 0.165, 0.160, 0.120, 0.075, 0.050],
    dtype=np.float64,
)


EXPANSION_FEATURE_CONFIG = ExpansionFeatureConfig(
    num_rows=6,
    num_reels=5,
    starting_unlocked_rows=3,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC6"],
    coin_symbol=REEL_DICT["COIN"],
    go_symbol=REEL_DICT["SC6"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    unlocked_landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC6"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    unlocked_landing_probabilities=np.array(
        [0.650, 0.275, 0.025, 0.026, 0.016, 0.006, 0.002],
        dtype=np.float64,
    ),
    locked_landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC6"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    locked_landing_probabilities=np.array(
        [0.800, 0.145, 0.010, 0.026, 0.014, 0.004, 0.001],
        dtype=np.float64,
    ),
    max_go_symbols=3,
    max_locked_go_symbols=1,
    starting_extra_coin_count=3,
)


class MultiplierFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    landing_symbols: np.ndarray
    landing_probabilities: np.ndarray
    multiplier_trigger_probability: float
    multiplier_counts: np.ndarray
    multiplier_count_probabilities: np.ndarray
    multiplier_values: np.ndarray
    multiplier_value_probabilities: np.ndarray
    starting_extra_coin_count: int


MULTIPLIER_FEATURE_CONFIG = MultiplierFeatureConfig(
    num_rows=3,
    num_reels=5,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC4"],
    coin_symbol=REEL_DICT["COIN"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    landing_probabilities=np.array(
        [0.675, 0.270, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    multiplier_trigger_probability=0.35,
    multiplier_counts=np.array([4, 5, 6], dtype=np.int16),
    multiplier_count_probabilities=np.array([0.45, 0.35, 0.20], dtype=np.float64),
    multiplier_values=np.array([2, 3, 4], dtype=np.int16),
    multiplier_value_probabilities=np.array([0.70, 0.22, 0.08], dtype=np.float64),
    starting_extra_coin_count=3,
)


class GrowFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    grower_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    landing_symbols: np.ndarray
    landing_probabilities: np.ndarray
    grow_trigger_probability: float
    grow_coin_counts: np.ndarray
    grow_coin_count_probabilities: np.ndarray
    grow_values: np.ndarray
    grow_value_probabilities: np.ndarray
    max_grower_symbols: int
    starting_extra_coin_count: int


GROW_VALUES = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    dtype=np.float64,
)
GROW_VALUE_PROBABILITIES = np.array(
    [0.08, 0.12, 0.16, 0.18, 0.16, 0.13, 0.10, 0.07],
    dtype=np.float64,
)


GROW_FEATURE_CONFIG = GrowFeatureConfig(
    num_rows=3,
    num_reels=5,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC2"],
    coin_symbol=REEL_DICT["COIN"],
    grower_symbol=REEL_DICT["SC2"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC2"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    landing_probabilities=np.array(
        [0.640, 0.270, 0.035, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    grow_trigger_probability=0.55,
    grow_coin_counts=np.array([3, 4, 5], dtype=np.int16),
    grow_coin_count_probabilities=np.array([0.45, 0.35, 0.20], dtype=np.float64),
    grow_values=GROW_VALUES,
    grow_value_probabilities=GROW_VALUE_PROBABILITIES,
    max_grower_symbols=3,
    starting_extra_coin_count=3,
)


class BoostFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    landing_symbols: np.ndarray
    landing_probabilities: np.ndarray
    boost_trigger_probability: float
    boost_values: np.ndarray
    boost_value_probabilities: np.ndarray
    starting_extra_coin_count: int


BOOST_FEATURE_CONFIG = BoostFeatureConfig(
    num_rows=3,
    num_reels=5,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC3"],
    coin_symbol=REEL_DICT["COIN"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    landing_probabilities=np.array(
        [0.675, 0.270, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    boost_trigger_probability=0.35,
    boost_values=GROW_VALUES,
    boost_value_probabilities=GROW_VALUE_PROBABILITIES,
    starting_extra_coin_count=3,
)


class CollectFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    collector_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    landing_symbols: np.ndarray
    landing_probabilities: np.ndarray
    max_collector_symbols: int
    starting_extra_coin_count: int


COLLECT_FEATURE_CONFIG = CollectFeatureConfig(
    num_rows=3,
    num_reels=5,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC5"],
    coin_symbol=REEL_DICT["COIN"],
    collector_symbol=REEL_DICT["SC5"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC5"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    landing_probabilities=np.array(
        [0.640, 0.270, 0.035, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    max_collector_symbols=3,
    starting_extra_coin_count=3,
)


class SplitterFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    respin_reset_count: int
    trigger_symbol: int
    coin_symbol: int
    splitter_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    splitter_coin_values: np.ndarray
    splitter_coin_value_probabilities: np.ndarray
    landing_symbols: np.ndarray
    landing_probabilities: np.ndarray
    splitter_coin_counts: np.ndarray
    splitter_coin_count_probabilities: np.ndarray
    min_splitter_symbols: int
    starting_extra_coin_count: int


SPLITTER_FEATURE_CONFIG = SplitterFeatureConfig(
    num_rows=3,
    num_reels=5,
    respin_reset_count=3,
    trigger_symbol=REEL_DICT["SC1"],
    coin_symbol=REEL_DICT["COIN"],
    splitter_symbol=REEL_DICT["SC1"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    splitter_coin_values=FEATURE_COIN_VALUES,
    splitter_coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC1"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    landing_probabilities=np.array(
        [0.600, 0.240, 0.105, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    splitter_coin_counts=np.array([2, 3], dtype=np.int16),
    splitter_coin_count_probabilities=np.array([0.70, 0.30], dtype=np.float64),
    min_splitter_symbols=3,
    starting_extra_coin_count=3,
)


class MegaComboFeatureConfig(NamedTuple):
    num_rows: int
    num_reels: int
    starting_unlocked_rows: int
    respin_reset_count: int
    trigger_symbols: np.ndarray
    coin_symbol: int
    splitter_symbol: int
    grower_symbol: int
    booster_symbol: int
    multiplier_symbol: int
    collector_symbol: int
    go_symbol: int
    jackpot_symbols: np.ndarray
    jackpot_collection_targets: np.ndarray
    jackpot_awards: np.ndarray
    coin_values: np.ndarray
    coin_value_probabilities: np.ndarray
    splitter_coin_values: np.ndarray
    splitter_coin_value_probabilities: np.ndarray
    unlocked_landing_symbols: np.ndarray
    unlocked_landing_probabilities: np.ndarray
    locked_landing_symbols: np.ndarray
    locked_landing_probabilities: np.ndarray
    additional_coin_counts: np.ndarray
    additional_coin_count_probabilities: np.ndarray
    multiplier_trigger_probability: float
    multiplier_counts: np.ndarray
    multiplier_count_probabilities: np.ndarray
    multiplier_values: np.ndarray
    multiplier_value_probabilities: np.ndarray
    boost_trigger_probability: float
    boost_values: np.ndarray
    boost_value_probabilities: np.ndarray
    grow_trigger_probability: float
    grow_coin_counts: np.ndarray
    grow_coin_count_probabilities: np.ndarray
    grow_values: np.ndarray
    grow_value_probabilities: np.ndarray
    splitter_coin_counts: np.ndarray
    splitter_coin_count_probabilities: np.ndarray
    max_go_symbols: int
    max_locked_go_symbols: int
    max_grower_symbols: int
    max_collector_symbols: int
    max_splitter_symbols: int
    min_splitter_symbols: int


MEGA_COMBO_FEATURE_CONFIG = MegaComboFeatureConfig(
    num_rows=6,
    num_reels=5,
    starting_unlocked_rows=3,
    respin_reset_count=3,
    trigger_symbols=np.array(
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
    coin_symbol=REEL_DICT["COIN"],
    splitter_symbol=REEL_DICT["SC1"],
    grower_symbol=REEL_DICT["SC2"],
    booster_symbol=REEL_DICT["SC3"],
    multiplier_symbol=REEL_DICT["SC4"],
    collector_symbol=REEL_DICT["SC5"],
    go_symbol=REEL_DICT["SC6"],
    jackpot_symbols=JACKPOT_TOKEN_CONFIG.symbols,
    jackpot_collection_targets=JACKPOT_TOKEN_CONFIG.collection_targets,
    jackpot_awards=JACKPOT_TOKEN_CONFIG.awards,
    coin_values=FEATURE_COIN_VALUES,
    coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    splitter_coin_values=FEATURE_COIN_VALUES,
    splitter_coin_value_probabilities=FEATURE_COIN_VALUE_PROBABILITIES,
    unlocked_landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC6"],
            REEL_DICT["SC2"],
            REEL_DICT["SC5"],
            REEL_DICT["SC1"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    unlocked_landing_probabilities=np.array(
        [0.555, 0.260, 0.025, 0.030, 0.030, 0.045, 0.032, 0.016, 0.005, 0.002],
        dtype=np.float64,
    ),
    locked_landing_symbols=np.array(
        [
            -1,
            REEL_DICT["COIN"],
            REEL_DICT["SC6"],
            REEL_DICT["SC2"],
            REEL_DICT["SC5"],
            REEL_DICT["SC1"],
            REEL_DICT["JP_MINI"],
            REEL_DICT["JP_MINOR"],
            REEL_DICT["JP_MAJOR"],
            REEL_DICT["JP_GRAND"],
        ],
        dtype=np.int16,
    ),
    locked_landing_probabilities=np.array(
        [0.715, 0.170, 0.010, 0.020, 0.020, 0.025, 0.026, 0.010, 0.003, 0.001],
        dtype=np.float64,
    ),
    additional_coin_counts=np.array([1, 2, 3], dtype=np.int16),
    additional_coin_count_probabilities=np.array([0.20, 0.40, 0.40], dtype=np.float64),
    multiplier_trigger_probability=0.35,
    multiplier_counts=np.array([4, 5, 6], dtype=np.int16),
    multiplier_count_probabilities=np.array([0.45, 0.35, 0.20], dtype=np.float64),
    multiplier_values=np.array([2, 3, 4], dtype=np.int16),
    multiplier_value_probabilities=np.array([0.70, 0.22, 0.08], dtype=np.float64),
    boost_trigger_probability=0.35,
    boost_values=np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5],
        dtype=np.float64,
    ),
    boost_value_probabilities=np.array(
        [0.07, 0.10, 0.14, 0.17, 0.16, 0.13, 0.10, 0.08, 0.05],
        dtype=np.float64,
    ),
    grow_trigger_probability=0.55,
    grow_coin_counts=np.array([3, 4, 5], dtype=np.int16),
    grow_coin_count_probabilities=np.array([0.45, 0.35, 0.20], dtype=np.float64),
    grow_values=GROW_VALUES,
    grow_value_probabilities=GROW_VALUE_PROBABILITIES,
    splitter_coin_counts=np.array([2, 3], dtype=np.int16),
    splitter_coin_count_probabilities=np.array([0.70, 0.30], dtype=np.float64),
    max_go_symbols=3,
    max_locked_go_symbols=1,
    max_grower_symbols=3,
    max_collector_symbols=3,
    max_splitter_symbols=3,
    min_splitter_symbols=3,
)


BASE_PAY_TABLE = np.array(
    [
        [0.00, 0.00, 1.00, 3.00, 10.00],
        [0.00, 0.00, 0.60, 2.00, 5.00],
        [0.00, 0.00, 0.40, 1.60, 4.00],
        [0.00, 0.00, 0.40, 1.00, 3.00],
        [0.00, 0.00, 0.40, 1.00, 3.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
        [0.00, 0.00, 0.20, 0.60, 2.00],
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


HOLD_AND_SPIN_CONFIG = HoldAndSpinConfig(
    num_rows=6,
    num_reels=5,
    starting_rows=3,
    starting_scatter_counts=np.array([1, 6], dtype=np.int8),
    starting_scatter_count_probabilities=np.array(
        [0.96, 0.04],
        dtype=np.float64,
    ),
    starting_coin_counts=np.array([1, 2, 3], dtype=np.int8),
    starting_coin_count_probabilities=np.array(
        [0.60, 0.30, 0.10],
        dtype=np.float64,
    ),
    coin_values=np.array([1, 2, 4, 8, 10], dtype=np.int16),
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
