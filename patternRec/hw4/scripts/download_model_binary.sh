#!/usr/bin/env sh

if [ ! -f snapshots/${1}_init.caffemodel ]; then
  python2 python/download_model_binary.py uris/${1}/
  mv uris/${1}/*.caffemodel snapshots/${1}_init.caffemodel
fi
