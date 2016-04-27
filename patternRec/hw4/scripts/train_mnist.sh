#!/usr/bin/env sh

CAFFE_TOOLS=$CAFFE_ROOT/build/tools

$CAFFE_TOOLS/caffe train --solver=mnist_solver.prototxt
