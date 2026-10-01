from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


REEL_DICT = {
    "H1": 0,
    "H2": 1,
    "H3": 2,
    "H4": 3,
    "H5": 4,
    "L1": 5,
    "L2": 6,
    "L3": 7,
    "L4": 8,
    "L5": 9,
    "L6": 10,
    "WD": 11,
    "COIN": 12,
    "COLLECT": 13,
    "SC": 14,
    "SC1": 15,
    "SC2": 16,
    "SC3": 17,
    "SC4": 18,
    "SC5": 19,
    "SC6": 20,
    "JP_MINI": 22,
    "JP_MINOR": 23,
    "JP_MAJOR": 24,
    "JP_GRAND": 25,
}


class ReelCollections(NamedTuple):
    lengths: np.ndarray
    reelsets: np.ndarray
    weights: np.ndarray


def csv_to_numba(csv_reelset, reel_dict, rules):
    reel_list = []
    for column in csv_reelset.columns:
        reel = csv_reelset[column]
        reel = [symbol for symbol in reel if str(symbol) != "nan"]
        reel = [reel_dict[symbol] for symbol in reel]
        reel_list.append(reel)

    if len(reel_list) != rules.num_reels:
        raise ValueError(
            f"Expected {rules.num_reels} reels, received {len(reel_list)}"
        )

    reel_lens = np.zeros(rules.num_reels, dtype=np.int16)
    for reel_idx in range(len(reel_list)):
        reel_lens[reel_idx] = len(reel_list[reel_idx])

    reelset_numba = np.full(
        (np.max(reel_lens), rules.num_reels),
        rules.reel_padding_symbol,
        dtype=np.int16,
    )
    for reel_idx in range(len(reel_lens)):
        reelset_numba[:reel_lens[reel_idx], reel_idx] = reel_list[reel_idx]

    return reel_lens, reelset_numba


def reelMaker(path, rules, pattern="ReelSet_*.csv"):
    if isinstance(path, (str, Path)):
        source = Path(path)
        if source.is_dir():
            reel_paths = sorted(source.glob(pattern))
        else:
            reel_paths = [source]
    else:
        reel_paths = sorted(Path(reel_path) for reel_path in path)

    if len(reel_paths) == 0:
        raise ValueError(f"No reelset CSV files found for {path}")

    lengths = np.zeros(
        (len(reel_paths), rules.num_reels),
        dtype=np.int16,
    )
    temporary_reelset_holder = []

    for reelset_idx, reel_path in enumerate(reel_paths):
        if not reel_path.is_file():
            raise FileNotFoundError(f"Reelset file not found: {reel_path}")

        csv_reelset = pd.read_csv(reel_path)
        reel_lens, reelset_numba = csv_to_numba(
            csv_reelset,
            REEL_DICT,
            rules,
        )
        lengths[reelset_idx] = reel_lens
        temporary_reelset_holder.append(reelset_numba)

    max_length = int(np.max(lengths))
    reelsets = np.full(
        (len(reel_paths), max_length, rules.num_reels),
        rules.reel_padding_symbol,
        dtype=np.int16,
    )

    for reelset_idx, reelset_numba in enumerate(temporary_reelset_holder):
        reelsets[
            reelset_idx,
            :reelset_numba.shape[0],
            :,
        ] = reelset_numba

    return lengths, reelsets


def make_reel_collection(
    path,
    rules,
    weights=None,
    pattern="ReelSet_*.csv",
):
    lengths, reelsets = reelMaker(
        path,
        rules,
        pattern=pattern,
    )
    num_reelsets = lengths.shape[0]

    if weights is None:
        weights = getattr(rules, "reelset_probabilities", None)

    if weights is None:
        normalized_weights = np.full(
            num_reelsets,
            1.0 / num_reelsets,
            dtype=np.float64,
        )
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)

        if normalized_weights.ndim != 1 or len(normalized_weights) != num_reelsets:
            raise ValueError(
                "Number of weights must match the number of reelsets"
            )
        if not np.all(np.isfinite(normalized_weights)):
            raise ValueError("Reelset weights must be finite")
        if np.any(normalized_weights < 0):
            raise ValueError("Reelset weights cannot be negative")

        total_weight = np.sum(normalized_weights)
        if total_weight <= 0:
            raise ValueError("At least one reelset weight must be positive")

        normalized_weights = normalized_weights / total_weight

    return ReelCollections(
        lengths=lengths,
        reelsets=reelsets,
        weights=normalized_weights,
    )


def make_reel_collections(
    rules,
    initial_weights=None,
    cascade_weights=None,
):
    initial_reels = make_reel_collection(
        rules.initial_reels_path,
        rules,
        weights=initial_weights,
    )
    cascade_reels = make_reel_collection(
        rules.cascade_reels_path,
        rules,
        weights=cascade_weights,
    )
    return initial_reels, cascade_reels
