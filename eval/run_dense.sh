#!/bin/bash
# Stitching regime (--keyframes dense): SLERP vs released SignSparK, 100 test clips per dataset, on the CPU
# (the GPU is held by the MaDiS queue). ~50 s/clip on 16 threads.
cd "$(dirname "$0")/.."
PY=../ver-01/.venv-signspark/bin/python; CK=../ver-01/SignSparK/checkpoints
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""   # keep the shared GPU free
done_() { [ -f outputs/eval/$1.json ]; }
for d in PHOENIX-2014T CSL-Daily How2Sign; do
  done_ slerp_${d}_kf2pdense_n100 || $PY -W ignore eval/evaluate.py --device cpu --name slerp_${d}_kf2pdense_n100 --dataset $d --baseline slerp --keyframes dense --n 100
  done_ released_${d}_kf2pdense_n100 || $PY -W ignore eval/evaluate.py --name released_${d}_kf2pdense_n100 --dataset $d --mode kf2p --keyframes dense --n 100 \
      --batch 10 --device cpu --hand-ckpt $CK/hand/ema_0.9999_200000.pt --body-ckpt $CK/body/ema_0.9999_200000.pt
done
