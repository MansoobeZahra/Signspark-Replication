#!/bin/bash
# Paper's T2P row used classifier-free guidance ("zero-keyframe inference ... through classifier-free guidance").
# Same README command + eval.keyframes_mask_all=True + eval.text_guidance_scale=2.5 (scale not stated in the paper).
cd "$(dirname "$0")"
while ! grep -q "ALL DONE" progress.log; do sleep 20; done
export DATA_ROOT="$PWD/data"
PY="/home/cvl/Desktop/Speech to sign/ver_02/.venv-signspark/bin/python"
SS="/home/cvl/Desktop/Speech to sign/ver-01/SignSparK"
for s in hand body; do
  echo "=== $(date +%T) START $s t2p_cfg2.5"
  "$PY" "$SS/sample.py" data_v2=$s data_v2/common@common=inference eval.model_path="$PWD/ckpt/$s/ema_0.9999_200000.pt" \
     eval.split=test eval.ode_stepnum=50 "test_data=[How2Sign]" eval.note=t2p_cfg2.5 eval.keyframes_mask_all=True \
     eval.text_guidance_scale=2.5 > logs/${s}_t2p_cfg2.5.log 2>&1
  echo "=== $(date +%T) END $s t2p_cfg2.5 exit=$? $(grep -o 'Written the decoded output to .*' logs/${s}_t2p_cfg2.5.log)"
done
echo "=== CFG DONE $(date +%T)"
