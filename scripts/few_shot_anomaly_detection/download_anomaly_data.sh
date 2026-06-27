#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p ../dataset

if ! command -v gdown &> /dev/null
then
    echo "installing gdown, for downloading from google drive"
    pip install gdown
fi

ZIP_PATH="../dataset/all_datasets.zip"

if [ ! -f "$ZIP_PATH" ]; then
    gdown "https://drive.google.com/file/d/1pmXvqWsfUeXWCMz5fqsP8WLKXR5jxY8z/view?usp=drive_link" --fuzzy -O "$ZIP_PATH"
fi

mkdir -p ../dataset

# Extract only the anomaly detection datasets used by data_provider/anomaly_detection.yaml.
unzip -q "$ZIP_PATH" \
    "all_datasets/MSL/*" \
    "all_datasets/PSM/*" \
    "all_datasets/SMAP/*" \
    "all_datasets/SMD/*" \
    "all_datasets/SWaT/*" \
    -d ../dataset/

for name in MSL PSM SMAP SMD SWaT
do
    if [ ! -d "../dataset/$name" ] && [ -d "../dataset/all_datasets/$name" ]; then
        mv "../dataset/all_datasets/$name" "../dataset/$name"
    fi
done

rmdir ../dataset/all_datasets 2>/dev/null || true

echo "Anomaly detection datasets are ready under ../dataset:"
ls -d ../dataset/MSL ../dataset/PSM ../dataset/SMAP ../dataset/SMD ../dataset/SWaT
