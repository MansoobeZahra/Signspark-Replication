#!/bin/bash
# Waits for ver_03_madis's step-4 sweep (another session) to finish, then runs the cost benchmarks one at a time.
cd "$(dirname "$0")/.."
until grep -q "STEP4 DONE" ../ver_03_madis/logs/step4_sweep.log; do sleep 30; done
gpu() { echo "--- $(date +%T) GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader); apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; }
gpu
../ver_03_madis/.venv-madis/bin/python scripts/bench_b_madis.py > logs/bench_b_madis.log 2>&1; echo "B exit $?"; gpu
../ver-01/.venv-signspark/bin/python scripts/vq_eval.py > logs/vq_eval.log 2>&1; echo "vq exit $?"
../ver-01/.venv-signspark/bin/python scripts/bench_a_signspark_xattn.py hand > logs/bench_a_hand.log 2>&1; echo "A hand exit $?"; gpu
../ver-01/.venv-signspark/bin/python scripts/bench_a_signspark_xattn.py body > logs/bench_a_body.log 2>&1; echo "A body exit $?"; gpu
echo "BENCH DONE"
