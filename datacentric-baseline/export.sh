#!/bin/bash

./build.sh

docker save datacentric_baseline | gzip -c > datacentric_baseline.tar.gz
