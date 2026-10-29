#!/bin/bash
# Official SignSparK sample.py on the How2Sign test split, README command (inference config, 50 Euler steps,
# no text guidance). Outputs land next to the symlinked checkpoints in ./ckpt/<stream>/ (sample.py's behaviour).
cd "$(dirname "$0")"
export DATA_ROOT="$PWD/data"
PY="/home/cvl/Desktop/Speech to sign/ver_02/.venv-signspark/bin/python"
SS="/home/cvl/Desktop/Speech to sign/ver-01/SignSparK"
run() {  # stream note extra-args...
  s=$1; n=$2; shift 2
  echo "=== $(date +%T) START $s $n $*"
  "$PY" "$SS/sample.py" data_v2=$s data_v2/common@common=inference eval.model_path="$PWD/ckpt/$s/ema_0.9999_200000.pt" \
     eval.split=test eval.ode_stepnum=50 "test_data=[How2Sign]" eval.note=$n "$@" > logs/${s}_${n}.log 2>&1
  echo "=== $(date +%T) END $s $n exit=$? $(grep -o 'Written the decoded output to .*' logs/${s}_${n}.log)"
}
mkdir -p logs
run hand kf2p
run hand t2p eval.keyframes_mask_all=True
run body kf2p
run body t2p eval.keyframes_mask_all=True
run hand kf2p_flipTrue_512 flip_left_hand=True eval.max_samples=512
echo "=== ALL DONE $(date +%T)"
