#!/usr/bin/env bash
# Run the full pipeline in order: data -> labels -> features -> baseline models.
# Stops at the first error. Run from the repo root: bash run_all.sh
#
# panel_full.csv is fixed (see CLAUDE.md), so the data step is skipped when it
# already exists. Set REBUILD_PANEL=1 to rebuild it. The rebuild reads raw files
# from the data/raw cache and never passes --refresh.
set -euo pipefail

cd "$(dirname "$0")"

INDICES=(GSPC N225 FTSE)
# The 2026-10-27 panel fixes EPU_US (see CLAUDE.md); panel_full.csv is the unchanged
# 2026-09-16 original. The path must match panel_path in configs/*.yaml.
PANEL="data/processed/panel_full_2026-10-27.csv"

if [[ -f "$PANEL" && "${REBUILD_PANEL:-0}" != "1" ]]; then
    echo "=== data_pipeline.py: skipped, $PANEL exists (REBUILD_PANEL=1 to rebuild) ==="
else
    echo "=== data_pipeline.py ==="
    python code/data_pipeline.py \
        --epu-us data/raw/US_Policy_Uncertainty_Data.xlsx \
        --epu-japan data/raw/Japan_Policy_Uncertainty_Data.xlsx \
        --kaggle data/raw/kaggle_indices_data.csv \
        --out "$PANEL"
fi

for idx in "${INDICES[@]}"; do
    echo "=== labelling.py --index $idx ==="
    python code/labelling.py --index "$idx"
done

for idx in "${INDICES[@]}"; do
    echo "=== features.py --index $idx ==="
    python code/features.py --index "$idx"
done

for idx in "${INDICES[@]}"; do
    echo "=== baseline_models.py --index $idx ==="
    python code/baseline_models.py --index "$idx"
done

echo "=== run_all.sh finished ==="
