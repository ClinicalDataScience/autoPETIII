#!/bin/bash

path="fiF42Tyu4n3EPTwjN5E7x5/.dir"
echo "Download: ${path}"
curl  -o "weights.zip" "https://syncandshare.lrz.de/dl/${path// /%20}"
unzip "weights.zip"
rm "weights.zip"

