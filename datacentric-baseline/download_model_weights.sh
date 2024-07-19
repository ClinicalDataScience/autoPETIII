#!/bin/bash

declare -a arr_data=(
  "fiQpSvvDinQGwCixsZmh1k/config.yml"
  "fiDMtokM94d6LPgyxAixzx/last.ckpt"
  "fiNi5TKUvZ6GJVwe3WqhZY/events.out.tfevents.all_samples.0"
  )

for path in "${arr_data[@]}"; do
    file=$(basename "$path")
    echo "Download: ${file}"
    curl --create-dirs -o "weights/${file}" "https://syncandshare.lrz.de/dl/${path// /%20}"
done

