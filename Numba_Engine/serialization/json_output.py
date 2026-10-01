import json
from pathlib import Path

from ..core.reels import REEL_DICT


JSON_LIBRARY_DIR = (
    Path(__file__).resolve().parents[1] / "output" / "json_library"
)

_HOLD_ROUTE_NAMES = (
    "Splitter",
    "Grow",
    "Boost",
    "Multiplier",
    "Collect",
    "Expansion",
    "Mega Combo",
    "Plain",
)
_HOLD_BAG_SYMBOLS = tuple(REEL_DICT[f"SC{index}"] for index in range(1, 7))


def _hold_route_details(starting_symbols):
    """Infer route name and logical rows from the valid starting state."""
    distinct_symbols = set(int(symbol) for symbol in starting_symbols)
    if not distinct_symbols:
        return _HOLD_ROUTE_NAMES[7], 3
    if distinct_symbols == set(_HOLD_BAG_SYMBOLS):
        return _HOLD_ROUTE_NAMES[6], 6
    if len(distinct_symbols) == 1:
        symbol = next(iter(distinct_symbols))
        if symbol in _HOLD_BAG_SYMBOLS:
            feature_index = _HOLD_BAG_SYMBOLS.index(symbol)
            return _HOLD_ROUTE_NAMES[feature_index], (
                6 if feature_index == 5 else 3
            )
    return "Invalid partial combination", (
        6 if REEL_DICT["SC6"] in distinct_symbols else 3
    )


def _step_to_dict(storage, step_index):
    return {
        "board": storage.boards[step_index].tolist(),
        "win": float(storage.wins[step_index]),
        "symbol_wins": storage.symbol_wins[step_index].tolist(),
        "multiplier_board": storage.multiplier_boards[step_index].tolist(),
        "winning_mask": storage.winning_masks[step_index].tolist(),
        "payout_multiplier": int(storage.payout_multipliers[step_index]),
    }


def _spin_to_dict(storage, spin_index):
    step_start = int(storage.spin_step_offsets[spin_index])
    step_end = int(storage.spin_step_offsets[spin_index + 1])
    return {
        "spin_index": spin_index,
        "win": float(storage.spin_wins[spin_index]),
        "free_game_trigger": bool(storage.spin_triggers[spin_index]),
        "global_multiplier_before": int(
            storage.spin_global_before[spin_index]
        ),
        "global_multiplier_after": int(storage.spin_global_after[spin_index]),
        "step_start": step_start,
        "step_end": step_end,
        "steps": [
            _step_to_dict(storage, step_index)
            for step_index in range(step_start, step_end)
        ],
    }


def _legacy_storage_to_dict(storage):
    """Convert the original cascade storage into JSON-compatible objects."""
    spin_count = int(storage.spin_count)
    session_count = int(storage.session_count)
    payload = {
        "format_version": 1,
        "step_count": int(storage.step_count),
        "spin_count": spin_count,
        "session_count": session_count,
    }

    if session_count == 0:
        payload["spins"] = [
            _spin_to_dict(storage, spin_index)
            for spin_index in range(spin_count)
        ]
        return payload

    sessions = []
    for session_index in range(session_count):
        spin_start = int(storage.session_spin_offsets[session_index])
        spin_end = int(storage.session_spin_offsets[session_index + 1])
        sessions.append(
            {
                "session_index": session_index,
                "win": float(storage.session_wins[session_index]),
                "initial_spins": int(
                    storage.session_initial_spins[session_index]
                ),
                "total_spins": int(
                    storage.session_total_spins[session_index]
                ),
                "spin_start": spin_start,
                "spin_end": spin_end,
                "spins": [
                    _spin_to_dict(storage, spin_index)
                    for spin_index in range(spin_start, spin_end)
                ],
            }
        )
    payload["sessions"] = sessions
    return payload


def base_game_storage_to_dict(storage):
    """Convert the walking-Collector base-game storage."""
    rounds = []
    for round_index in range(int(storage.round_count)):
        spin_start = int(storage.round_spin_offsets[round_index])
        spin_end = int(storage.round_spin_offsets[round_index + 1])
        spins = []
        for spin_index in range(spin_start, spin_end):
            winning_lines = []
            for entry_index in range(storage.winning_line_numbers.shape[1]):
                line_number = int(
                    storage.winning_line_numbers[spin_index, entry_index]
                )
                if line_number < 0:
                    break
                winning_lines.append(
                    {
                        "line_number": line_number,
                        "symbol": int(
                            storage.line_winning_symbols[
                                spin_index, entry_index
                            ]
                        ),
                        "symbol_count": int(
                            storage.line_symbol_counts[
                                spin_index, entry_index
                            ]
                        ),
                        "win": float(
                            storage.line_win_amounts[
                                spin_index, entry_index
                            ]
                        ),
                    }
                )

            spins.append(
                {
                    "spin_index": spin_index,
                    "board": storage.boards[spin_index].tolist(),
                    "coin_value_board": (
                        storage.coin_value_boards[spin_index].tolist()
                    ),
                    "jackpot_overlay_board": (
                        storage.jackpot_overlay_boards[spin_index].tolist()
                    ),
                    "jackpot_values_before": (
                        storage.jackpot_values_before[spin_index].tolist()
                    ),
                    "jackpot_values_after": (
                        storage.jackpot_values_after[spin_index].tolist()
                    ),
                    "jackpot_increment_counts": (
                        storage.jackpot_increment_counts[spin_index].tolist()
                    ),
                    "win": float(storage.wins[spin_index]),
                    "line_win": float(storage.line_wins[spin_index]),
                    "collect_win": float(storage.collect_wins[spin_index]),
                    "free_game_trigger": bool(
                        storage.spin_triggers[spin_index]
                    ),
                    "collector_count": int(
                        storage.collector_counts[spin_index]
                    ),
                    "symbol_wins": storage.symbol_wins[spin_index].tolist(),
                    "symbol_hit_counts": (
                        storage.symbol_hit_counts[spin_index].tolist()
                    ),
                    "winning_lines": winning_lines,
                }
            )

        rounds.append(
            {
                "round_index": round_index,
                "win": float(storage.round_wins[round_index]),
                "free_game_triggers": int(
                    storage.round_triggers[round_index]
                ),
                "spin_start": spin_start,
                "spin_end": spin_end,
                "spins": spins,
            }
        )

    return {
        "format_version": 1,
        "storage_type": "base_game",
        "spin_count": int(storage.spin_count),
        "round_count": int(storage.round_count),
        "rounds": rounds,
    }


def hold_and_spin_storage_to_dict(storage):
    """Convert Hold-and-Spin session/respin/step storage."""
    sessions = []
    for session_index in range(int(storage.session_count)):
        starting_count = int(
            storage.session_starting_symbol_counts[session_index]
        )
        starting_symbols = storage.session_starting_symbols[
            session_index, :starting_count
        ].tolist()
        feature_route, logical_board_rows = _hold_route_details(
            starting_symbols
        )
        stored_board_rows = storage.boards.shape[1]
        row_offset = stored_board_rows - logical_board_rows
        num_reels = storage.boards.shape[2]
        respin_start = int(storage.session_respin_offsets[session_index])
        respin_end = int(storage.session_respin_offsets[session_index + 1])
        respins = []
        for respin_index in range(respin_start, respin_end):
            step_start = int(storage.respin_step_offsets[respin_index])
            step_end = int(storage.respin_step_offsets[respin_index + 1])
            steps = []
            for step_index in range(step_start, step_end):
                feature_position = int(
                    storage.feature_positions[step_index]
                )
                if feature_position >= 0:
                    feature_position -= row_offset * num_reels
                steps.append(
                    {
                        "step_index": step_index,
                        "board": storage.boards[
                            step_index, row_offset:
                        ].tolist(),
                        "coin_mask": storage.coin_masks[
                            step_index, row_offset:
                        ].tolist(),
                        "feature_type": int(storage.feature_types[step_index]),
                        "feature_position": feature_position,
                        "remaining_spins": int(
                            storage.step_remaining_spins[step_index]
                        ),
                        "locked_row_index": int(
                            storage.step_locked_row_indices[step_index]
                        ) - row_offset,
                        "collector_meter": int(
                            storage.step_collector_meters[step_index]
                        ),
                    }
                )

            respins.append(
                {
                    "respin_index": respin_index,
                    "remaining_before": int(
                        storage.respin_remaining_before[respin_index]
                    ),
                    "remaining_after": int(
                        storage.respin_remaining_after[respin_index]
                    ),
                    "reset": bool(storage.respin_reset_flags[respin_index]),
                    "jackpot_overlay_board": (
                        storage.respin_jackpot_overlay_boards[
                            respin_index, row_offset:
                        ].tolist()
                    ),
                    "jackpot_meters_before": storage.respin_jackpot_meters_before[
                        respin_index
                    ].tolist(),
                    "jackpot_meters_after": storage.respin_jackpot_meters_after[
                        respin_index
                    ].tolist(),
                    "jackpot_awards": storage.respin_jackpot_awards[
                        respin_index
                    ].tolist(),
                    "step_start": step_start,
                    "step_end": step_end,
                    "steps": steps,
                }
            )

        sessions.append(
            {
                "session_index": session_index,
                "feature_route": feature_route,
                "logical_board_shape": [logical_board_rows, num_reels],
                "storage_board_shape": [stored_board_rows, num_reels],
                "win": float(storage.session_wins[session_index]),
                "coin_win": float(storage.session_coin_wins[session_index]),
                "collector_win": float(
                    storage.session_collector_wins[session_index]
                ),
                "total_respins": int(
                    storage.session_total_respins[session_index]
                ),
                "starting_symbols": starting_symbols,
                "jackpot_meters": storage.session_jackpot_meters[
                    session_index
                ].tolist(),
                "jackpot_awards": storage.session_jackpot_awards[
                    session_index
                ].tolist(),
                "respin_start": respin_start,
                "respin_end": respin_end,
                "respins": respins,
            }
        )

    return {
        "format_version": 1,
        "storage_type": "hold_and_spin",
        "step_count": int(storage.step_count),
        "respin_count": int(storage.respin_count),
        "session_count": int(storage.session_count),
        "sessions": sessions,
    }


def full_game_storage_to_dict(
    full_storage,
    base_storage,
    hold_and_spin_storage=None,
):
    """Join full-game, base, and optional detailed H&S storage."""
    if int(full_storage.spin_count) != int(base_storage.spin_count):
        raise ValueError("Full-game and base-game spin counts do not match")
    if int(full_storage.round_count) != int(base_storage.round_count):
        raise ValueError("Full-game and base-game round counts do not match")

    base_payload = base_game_storage_to_dict(base_storage)
    base_spins = [
        spin
        for round_result in base_payload["rounds"]
        for spin in round_result["spins"]
    ]
    feature_sessions = None
    if hold_and_spin_storage is not None:
        feature_payload = hold_and_spin_storage_to_dict(
            hold_and_spin_storage
        )
        feature_sessions = feature_payload["sessions"]
        if len(feature_sessions) != int(full_storage.feature_session_count):
            raise ValueError(
                "Full-game and Hold-and-Spin session counts do not match"
            )
    feature_names = (
        "Splitter",
        "Grow",
        "Boost",
        "Multiplier",
        "Collect",
        "Expansion",
        "Mega Combo",
        "Plain",
    )
    rounds = []
    for round_index in range(int(full_storage.round_count)):
        spin_start = int(full_storage.round_spin_offsets[round_index])
        spin_end = int(full_storage.round_spin_offsets[round_index + 1])
        spins = []
        for spin_index in range(spin_start, spin_end):
            session_index = int(
                full_storage.spin_feature_session_indices[spin_index]
            )
            feature_session = None
            if feature_sessions is not None and session_index >= 0:
                if session_index >= len(feature_sessions):
                    raise ValueError(
                        "Invalid Hold-and-Spin session link in full game"
                    )
                feature_session = feature_sessions[session_index]
            feature_mask = int(full_storage.spin_feature_masks[spin_index])
            feature_routes = [
                name
                for feature_index, name in enumerate(feature_names)
                if feature_mask & (1 << feature_index)
            ]
            spins.append(
                {
                    "spin_index": spin_index,
                    "base_spin": base_spins[spin_index],
                    "feature_session_index": session_index,
                    "feature_session": feature_session,
                    "feature_routes": feature_routes,
                    "base_win": float(full_storage.spin_base_wins[spin_index]),
                    "feature_win": float(
                        full_storage.spin_feature_wins[spin_index]
                    ),
                    "jackpot_win": float(
                        full_storage.spin_jackpot_wins[spin_index]
                    ),
                    "total_win": float(
                        full_storage.spin_total_wins[spin_index]
                    ),
                    "jackpot_values_before_feature": (
                        full_storage.spin_jackpot_values_before_feature[
                            spin_index
                        ].tolist()
                    ),
                    "jackpot_awards": full_storage.spin_jackpot_awards[
                        spin_index
                    ].tolist(),
                    "jackpot_award_amounts": (
                        full_storage.spin_jackpot_award_amounts[
                            spin_index
                        ].tolist()
                    ),
                    "jackpot_values_after_feature": (
                        full_storage.spin_jackpot_values_after_feature[
                            spin_index
                        ].tolist()
                    ),
                }
            )
        rounds.append(
            {
                "round_index": round_index,
                "spin_start": spin_start,
                "spin_end": spin_end,
                "base_win": float(full_storage.round_base_wins[round_index]),
                "feature_win": float(
                    full_storage.round_feature_wins[round_index]
                ),
                "jackpot_win": float(
                    full_storage.round_jackpot_wins[round_index]
                ),
                "total_win": float(
                    full_storage.round_total_wins[round_index]
                ),
                "spins": spins,
            }
        )

    return {
        "format_version": 1,
        "storage_type": "full_game",
        "spin_count": int(full_storage.spin_count),
        "round_count": int(full_storage.round_count),
        "feature_session_count": int(full_storage.feature_session_count),
        "feature_names": list(feature_names),
        "feature_trigger_counts": full_storage.feature_trigger_counts.tolist(),
        "feature_win_amounts": full_storage.feature_win_amounts.tolist(),
        "feature_spin_counts": full_storage.feature_spin_counts.tolist(),
        "respin_count": (
            int(hold_and_spin_storage.respin_count)
            if hold_and_spin_storage is not None
            else None
        ),
        "step_count": (
            int(hold_and_spin_storage.step_count)
            if hold_and_spin_storage is not None
            else None
        ),
        "rounds": rounds,
    }


def storage_to_dict(storage):
    """Dispatch the supplied Storage class to its JSON representation."""
    if hasattr(storage, "round_count"):
        return base_game_storage_to_dict(storage)
    if hasattr(storage, "respin_count"):
        return hold_and_spin_storage_to_dict(storage)
    return _legacy_storage_to_dict(storage)


def write_json(storage, filename, overwrite=False, indent=2):
    """Write populated storage inside the package's JSON library."""
    return write_json_payload(
        storage_to_dict(storage),
        filename,
        overwrite=overwrite,
        indent=indent,
    )


def write_json_payload(payload, filename, overwrite=False, indent=2):
    """Write an already composed JSON payload inside the JSON library."""
    filename = Path(filename)
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("JSON filename must not contain a directory path")
    if filename.suffix == "":
        filename = filename.with_suffix(".json")
    elif filename.suffix.lower() != ".json":
        raise ValueError("JSON filename must use the .json extension")

    JSON_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = JSON_LIBRARY_DIR / filename
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"JSON output already exists: {output_path}")

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=indent)
    return output_path


def append_json_history(payload, filename, storage_type, indent=2):
    """Append one payload to a persistent, valid-JSON gameplay history."""
    filename = Path(filename)
    if filename.is_absolute() or len(filename.parts) != 1:
        raise ValueError("JSON filename must not contain a directory path")
    if filename.suffix == "":
        filename = filename.with_suffix(".json")
    elif filename.suffix.lower() != ".json":
        raise ValueError("JSON filename must use the .json extension")
    if payload.get("storage_type") != storage_type:
        raise ValueError("Payload storage type does not match gameplay history")

    JSON_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = JSON_LIBRARY_DIR / filename
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as input_file:
            history = json.load(input_file)
        if (
            history.get("format_version") != 1
            or history.get("storage_type") != storage_type
            or not isinstance(history.get("plays"), list)
        ):
            raise ValueError(f"Invalid gameplay history: {output_path}")
    else:
        history = {
            "format_version": 1,
            "storage_type": storage_type,
            "plays": [],
        }

    history["plays"].append(payload)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(history, output_file, indent=indent)
    return output_path
