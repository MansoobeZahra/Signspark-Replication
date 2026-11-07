#!/bin/bash
# Full ASL Citizen build, after the 8 aslc_build.py --shard units: recognizer train (GPU) in parallel with the
# all-gloss medoid/consensus (full_merge.py, 4 shards), then gather + recognizer check. Log: logs/full_chain.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv-sltk/bin/python
while [ -n "$(systemctl --user list-units 'aslc_full_s*' 'aslc_good_fix2*' --state=active --no-legend)" ]; do sleep 30; done
echo "$(date) build shards + demo rebuild done: $(find data/asl_citizen/dict/clips -mindepth 2 -maxdepth 2 | wc -l) clips"
grep -h "^shard" logs/full_s*.log
OMP_NUM_THREADS=8 $PY dictionary/recognizer.py train > logs/recognizer_train.log 2>&1 &
TR=$!
for k in 0 1 2 3; do
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $PY dictionary/full_merge.py --shard $k/4 > logs/full_merge_s$k.log 2>&1 &
done
wait $(jobs -p | grep -v "^$TR$")
echo "$(date) full_merge shards done"
$PY dictionary/full_merge.py --gather
wait $TR
echo "$(date) recognizer train exit $?"
tail -5 logs/recognizer_train.log
$PY dictionary/recognizer.py check > logs/recognizer_check.log 2>&1
echo "$(date) recognizer check exit $?"
tail -15 logs/recognizer_check.log
echo FULL_CHAIN_DONE
