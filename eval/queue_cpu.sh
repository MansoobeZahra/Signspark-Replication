#!/bin/bash
# CPU-only queue while the GPU belongs to the MaDiS session: dense-regime KF2P, then SR-T2P on PHOENIX and CSL.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
eval/run_dense.sh
for d in PHOENIX-2014T CSL-Daily; do
  ../ver-01/.venv-signspark/bin/python -W ignore eval/sr_t2p.py --dataset $d --steps 10 --device cpu
done
echo QUEUE DONE
