#!/bin/bash
set -e

TAG="final"
ZIP_NAME="AutoPETIII_datacentric_baseline.zip"

URL="https://github.com/ClinicalDataScience/autoPETIII/releases/download/${TAG}/${ZIP_NAME}"

mkdir -p weights
tmpdir=$(mktemp -d)

echo "Download: ${ZIP_NAME}"
curl -L -o "${tmpdir}/${ZIP_NAME}" "${URL}"

echo "Extracting files"
unzip -j "${tmpdir}/${ZIP_NAME}" \
  config.yml \
  last.ckpt \
  events.out.tfevents.all_samples.0 \
  -d weights

rm -rf "${tmpdir}"
