#!/usr/bin/env sh

cmd_line_args="--solver=models/${1}_solver.prototxt"
$CAFFE_TOOLS/caffe train $cmd_line_args
