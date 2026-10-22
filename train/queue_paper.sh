#!/usr/bin/env bash
# Unattended queue for the paper-config replication (one GPU: train and eval cannot overlap).
#   1. wait for the body run (already started by train/run.sh paper_body_100k body 100000 microbatch=32)
#   2. evaluate the retrained body (with the released hand; body joints depend only on the body stream)
#   3. train the hand stream, 4. evaluate the retrained pair.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/../ver-01/.venv-signspark/bin/python"
REL="$REPO/../ver-01/SignSparK/checkpoints"
ck() { ls "$REPO"/outputs/train/$1/checkpoints/*/ema_0.9999_$(printf %06d $2).pt; }

while pgrep -f "paper_body_100k" > /dev/null; do sleep 120; done

ev() {  # name hand body mode steps [extra]
  local n=$1 h=$2 b=$3 m=$4 s=$5; shift 5
  for ds in How2Sign CSL-Daily PHOENIX-2014T; do
    "$PY" eval/evaluate.py --name "${n}_${ds}_${m}_s${s}" --dataset $ds --mode $m --steps $s \
      --hand-ckpt "$h" --body-ckpt "$b" "$@" 2>&1 | tail -9
  done
}
RH="$REL/hand/ema_0.9999_200000.pt"; RB="$REL/body/ema_0.9999_200000.pt"
for s in 1 10 50; do ev released "$RH" "$RB" t2p $s --controls; done
ev released "$RH" "$RB" kf2p 10
B=$(ck paper_body_100k 100000)
for s in 1 10 50; do ev paperbody "$REL/hand/ema_0.9999_200000.pt" "$B" t2p $s --controls; done
ev paperbody "$REL/hand/ema_0.9999_200000.pt" "$B" kf2p 10

PYTORCH_ALLOC_CONF=expandable_segments:True train/run.sh paper_hand_100k hand 100000 microbatch=32 > /dev/null 2>&1
Hd=$(ck paper_hand_100k 100000)
for s in 1 10 50; do ev paper "$Hd" "$B" t2p $s --controls; done
ev paper "$Hd" "$B" kf2p 10
echo QUEUE DONE
