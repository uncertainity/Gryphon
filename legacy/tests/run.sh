#!/usr/bin/env bash

set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TEST_DIR}/.." && pwd)"
LOG_FILE="${TEST_DIR}/simulation_10000.log"

cd "${PROJECT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

{
    echo "Gryphon 10,000-run simulation test"
    echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo

    echo "========== BASE GAME: 10,000 ROUNDS =========="
    python -c 'from Numba_Engine.simulations.base_game import run_sims; run_sims(num_rounds=10_000, output_filename="test_base_game_10000.npz", overwrite=True, print_statistics=True)'
    echo

    echo "========== FREE GAME: 10,000 SESSIONS =========="
    python -c 'from Numba_Engine.simulations.free_game import run_sims; run_sims(num_sessions=10_000, output_filename="test_free_game_10000.npz", overwrite=True, print_statistics=True)'
    echo

    echo "========== FULL GAME: 10,000 ROUNDS =========="
    python -c 'from Numba_Engine.simulations.full_game import run_sims; run_sims(num_rounds=10_000, output_filename="test_full_game_10000.npz", overwrite=True, print_statistics=True)'
    echo

    echo "========== DASHBOARD =========="
    python bundle/generate_dashboard.py
    echo
    echo "Completed: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
} 2>&1 | tee "${LOG_FILE}"

echo "Simulation statistics written to ${LOG_FILE}"
