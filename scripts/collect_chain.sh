#!/bin/bash
# Run the investor-split collection in back-to-back slices until nothing is
# left, then stop.
#
# WHY THERE IS NO "WAIT FOR THE PREVIOUS RUN TO FINISH" LOOP HERE.
# The first version opened with:
#
#     until ! pgrep -f investor_split_collect >/dev/null; do sleep 30; done
#
# and deadlocked for 47 minutes without collecting anything. The script had
# been created by a bash heredoc, so the PARENT shell's command line contained
# the whole script text -- including the string `investor_split_collect`.
# pgrep -f matches against command lines, so it matched the parent, forever.
#
# The same false positive made every "is the collector running?" status check
# report RUNNING when nothing was running. This repo hit the identical trap
# once before with flow_panel_collect. Two lessons, both encoded here:
#
#   1. This file lives on disk and is invoked by path, so no launching command
#      line ever contains the pattern.
#   2. The wait is unnecessary anyway -- each slice blocks until it exits, so
#      running them in sequence is enough. Do not add a pgrep guard back.
#
# Progress is judged by COUNTING CACHED FILES, never by pgrep. A file count is
# unambiguous; a process match is a guess about someone else's command line.
set -u
cd "$(dirname "$0")/.."
LOG="${1:-/tmp/collect_chain.log}"
DECILES="${2:-9}"

remaining() {
  python3 scripts/investor_split_collect.py --deciles "$DECILES" --dry-run \
    2>/dev/null | grep REMAINING | awk '{print $2}' | tr -d ,
}

for i in $(seq 1 12); do
  R=$(remaining)
  echo "=== slice $i start $(date +%H:%M:%S) remaining=$R" >> "$LOG"
  if [ -z "$R" ] || [ "$R" = "0" ]; then
    echo "=== nothing left" >> "$LOG"
    break
  fi
  timeout 3300 python3 scripts/investor_split_collect.py --deciles "$DECILES" \
      --max-seconds 3150 --budget 1500 >> "$LOG" 2>&1
  echo "=== slice $i end   $(date +%H:%M:%S) remaining=$(remaining)" >> "$LOG"
done
echo "=== CHAIN DONE $(date +%H:%M:%S) remaining=$(remaining)" >> "$LOG"
