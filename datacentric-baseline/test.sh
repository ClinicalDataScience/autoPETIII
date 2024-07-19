#!/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create datacentric_baseline-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --gpus="all"  \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v datacentric_baseline-output-$VOLUME_SUFFIX:/output/ \
        datacentric_baseline

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t dc_eval .

docker run --name dc_eval_container \
        -v datacentric_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output_dc/:/expected_output/ \
        dc_eval python3 -c """
import SimpleITK as sitk
import os
import json

print('Start')
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
print(file)
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/PRED.nii.gz'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse <= 10:
    print('Test passed!')
else:
    print(f'Test failed! MSE={mse}')

print(os.listdir('/output'))
with open('/output/data-centric-model.json', 'r') as f:
    data = json.load(f)
    assert (data == True)
    print('Data Centric Model: Passed!')
"""
# Copy the file from the container to the host
#docker cp dc_eval_container:/output/images/automated-petct-lesion-segmentation/PRED.nii.gz YOUR_LOCAL_PATH/results/PRED.nii.gz
docker stop dc_eval_container
docker rm dc_eval_container

docker volume rm datacentric_baseline-output-$VOLUME_SUFFIX

# Capture the end time and print difference
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total runtime: $elapsed_time seconds"