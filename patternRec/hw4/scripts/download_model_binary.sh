#!/usr/bin/env sh

python2 python/download_model_binary.py uris/${1}/
mv uris/${1}/*.caffemodel snapshots/${1}_init.caffemodel
