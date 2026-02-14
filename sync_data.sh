#!/bin/bash
# Sync latest experiment data to the dashboard repo.
# Run after iterations complete to update the cloud dashboard.
#
# Usage: ./sync_data.sh

SRC="$HOME/ai-scientist/data/gradient_descent"
DST="$(dirname "$0")/data/gradient_descent"

cp "$SRC/state.json" "$DST/" 2>/dev/null
for d in "$SRC"/iteration_*; do
    iter=$(basename "$d")
    mkdir -p "$DST/$iter"
    cp "$d/iteration_log.json" "$DST/$iter/" 2>/dev/null
    cp "$d/transfer_array.json" "$DST/$iter/" 2>/dev/null
done

cd "$(dirname "$0")"
git add -A
git commit -m "Update experiment data $(date '+%Y-%m-%d %H:%M')"
git push
echo "Data synced and pushed."
