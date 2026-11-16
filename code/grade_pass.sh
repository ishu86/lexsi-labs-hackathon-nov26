#!/bin/zsh
# grade_pass.sh ROOT RUN CONFIGS KIND(l3|harm3) [--skip-xstest] : one grading pass of one rubric on one root, retried until it completes.
cd "$(dirname "$0")"  # expects AZURE_URL, AZURE_DEEPSEEK_API_KEY and HF_TOKEN in the environment
ROOT=$1; RUN=$2; C=$3; K=$4; SX=$5
for attempt in 1 2 3 4 5 6 7 8; do
  if [ "$K" = "l3" ]; then
    LABELS3=1 /opt/homebrew/bin/python3.11 ds_judge.py --run-root ../../judge/$ROOT --pull --run-name $RUN --configs $C $SX --chunk-size 50 --concurrency 12 >> ../../judge/${ROOT}_l3.log 2>&1 && break
  else
    RUBRIC_FILE=rubrics/harm_v3.txt RUBRIC_TAG=harm3 /opt/homebrew/bin/python3.11 ds_judge.py --run-root ../../judge/$ROOT --pull --run-name $RUN --configs $C --skip-xstest --chunk-size 50 --concurrency 12 >> ../../judge/${ROOT}_harm3.log 2>&1 && break
  fi
  echo "$(date -u +%H:%M:%S) $ROOT $K attempt $attempt failed; retrying in 20 s" >> ../../judge/grade_retries.log; /bin/sleep 20
done
echo "$(date -u +%H:%M:%S) $ROOT $K done" >> ../../judge/grade_done.log
