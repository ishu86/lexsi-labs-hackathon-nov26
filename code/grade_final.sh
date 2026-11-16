#!/bin/zsh
# grade_final.sh RUN CONFIGS [--skip-xstest] : both frozen rubrics on one run. Two passes each + a third on disagreement. No GPU.
#   three-label (refused / complied / other) on the harmful prompts and, unless skipped, the safe XSTest prompts
#   harm v3 (harmful or not) on the harmful prompts only
# Resumable: finished chunks are kept. Scores land in judge/RUN{,_p2,_p3}/{judge_deepseek3,judge_ds_harm3}/. Text is never logged.
cd "$(dirname "$0")"  # expects AZURE_URL, AZURE_DEEPSEEK_API_KEY and HF_TOKEN in the environment
R=$1; C=$2; SX=$3; J=../../judge
mkdir -p $J/$R $J/${R}_p2 $J/${R}_p3
for r in $R ${R}_p2; do
  ( LABELS3=1 /opt/homebrew/bin/python3.11 ds_judge.py --run-root $J/$r --pull --run-name $R --configs $C $SX --chunk-size 50 --concurrency 8 >> $J/${r}_l3.log 2>&1
    RUBRIC_FILE=rubrics/harm_v3.txt RUBRIC_TAG=harm3 /opt/homebrew/bin/python3.11 ds_judge.py --run-root $J/$r --configs $C --skip-xstest --chunk-size 50 --concurrency 8 >> $J/${r}_harm3.log 2>&1 ) &
done; wait
SKIP_XSTEST=$([ -n "$SX" ] && echo 1 || echo 0) LABELS3=1 CONC=8 /opt/homebrew/bin/python3.11 third_pass.py $J/$R $J/${R}_p2 $J/${R}_p3 $C 2>&1 | grep 'third pass\|errors'
SKIP_XSTEST=1 RUBRIC_FILE=rubrics/harm_v3.txt RUBRIC_TAG=harm3 CONC=8 /opt/homebrew/bin/python3.11 third_pass.py $J/$R $J/${R}_p2 $J/${R}_p3 $C 2>&1 | grep 'third pass\|errors'
for f in $J/${R}_l3.log $J/${R}_p2_l3.log $J/${R}_harm3.log $J/${R}_p2_harm3.log; do grep -h ' done ' $f | grep -o 'usd=[0-9.]*' | awk -F= '{s+=$2} END {printf "usd %.3f ", s}'; done; echo
