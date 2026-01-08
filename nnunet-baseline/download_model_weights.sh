#!/bin/bash
set -e

TAG="final"
ZIP_NAME="AutoPETIII_nnunet_baseline.zip"

URL="https://github.com/ClinicalDataScience/autoPETIII/releases/download/${TAG}/${ZIP_NAME}"

echo "Download: ${ZIP_NAME}"
curl -L -o "${ZIP_NAME}" "${URL}"

echo "Extracting nnUNet_results"
unzip "${ZIP_NAME}" "AutoPETIII_nnunet_baseline/nnUNet_results/*"
mv AutoPETIII_nnunet_baseline/nnUNet_results .
rm -rf AutoPETIII_nnunet_baseline "${ZIP_NAME}"

