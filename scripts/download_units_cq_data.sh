#!/usr/bin/env bash
set -euo pipefail

# Download only the datasets needed for the UniTS-CQ plan:
#   Forecasting: WeatherP96, ECLP96
#   Anomaly detection: SMD, PSM
#
# Usage:
#   bash scripts/download_units_cq_data.sh
#   DATA_DIR=../dataset bash scripts/download_units_cq_data.sh
#
# By default this writes to ./dataset inside the UniTS repo. If you want to use
# the original UniTS YAML files that point to ../dataset, run with
# DATA_DIR=../dataset.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-dataset}"
ZIP_PATH="$DATA_DIR/all_datasets.zip"
GDRIVE_URL="https://drive.google.com/file/d/1pmXvqWsfUeXWCMz5fqsP8WLKXR5jxY8z/view?usp=drive_link"

mkdir -p "$DATA_DIR"

ensure_gdown() {
    if command -v gdown >/dev/null 2>&1; then
        return
    fi

    echo "[setup] gdown not found; installing with python -m pip"
    python -m pip install gdown
}

download_archive() {
    if [ -f "$ZIP_PATH" ]; then
        echo "[skip] Found existing archive: $ZIP_PATH"
        return
    fi

    ensure_gdown
    echo "[download] Fetching TimesNet/UniTS all_datasets.zip"
    gdown "$GDRIVE_URL" --fuzzy -O "$ZIP_PATH"
}

extract_selected() {
    echo "[extract] Extracting Weather, electricity, SMD, and PSM only"
    unzip -qn "$ZIP_PATH" \
        "all_datasets/weather/*" \
        "all_datasets/electricity/*" \
        "all_datasets/SMD/*" \
        "all_datasets/PSM/*" \
        -d "$DATA_DIR"

    for name in weather electricity SMD PSM
    do
        if [ ! -d "$DATA_DIR/$name" ] && [ -d "$DATA_DIR/all_datasets/$name" ]; then
            mv "$DATA_DIR/all_datasets/$name" "$DATA_DIR/$name"
        fi
    done

    rmdir "$DATA_DIR/all_datasets" 2>/dev/null || true
}

verify_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "[missing] $path"
        return 1
    fi
    echo "[ok] $path"
}

verify_data() {
    echo "[verify] Checking required files"
    local missing=0

    verify_file "$DATA_DIR/weather/weather.csv" || missing=1
    verify_file "$DATA_DIR/electricity/electricity.csv" || missing=1
    verify_file "$DATA_DIR/SMD/SMD_train.npy" || missing=1
    verify_file "$DATA_DIR/SMD/SMD_test.npy" || missing=1
    verify_file "$DATA_DIR/SMD/SMD_test_label.npy" || missing=1
    verify_file "$DATA_DIR/PSM/train.csv" || missing=1
    verify_file "$DATA_DIR/PSM/test.csv" || missing=1
    verify_file "$DATA_DIR/PSM/test_label.csv" || missing=1

    if [ "$missing" -ne 0 ]; then
        echo
        echo "[error] Some required dataset files are missing."
        echo "        Check whether $ZIP_PATH downloaded correctly and whether the archive layout changed."
        exit 1
    fi
}

download_archive
extract_selected
verify_data

echo
echo "[done] UniTS-CQ datasets are ready under: $DATA_DIR"
echo "       Forecast configs need:"
echo "         $DATA_DIR/weather/weather.csv"
echo "         $DATA_DIR/electricity/electricity.csv"
echo "       AD configs need:"
echo "         $DATA_DIR/SMD"
echo "         $DATA_DIR/PSM"
