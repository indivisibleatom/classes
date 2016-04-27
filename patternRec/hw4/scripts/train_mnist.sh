#!/usr/bin/env sh

CAFFE_TOOLS=$CAFFE_ROOT/build/tools

$CAFFE_TOOLS/caffe train --solver=models/mnist_solver.prototxt
