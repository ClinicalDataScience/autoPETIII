#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t datacentric_baseline "$SCRIPTPATH"
