#!/usr/bin/env sh

CAFFE_TOOLS=$CAFFE_ROOT/build/tools

DATA_ROOT=.
TRAIN_DATA_ROOT=$DATA_ROOT/data/mnist/train
TEST_DATA_ROOT=$DATA_ROOT/data/mnist/test
BIN_DIR=bin

echo "Creating training lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --shuffle \
    $DATA_ROOT \
    $TRAIN_DATA_ROOT/train.txt \
    $BIN_DIR/mnist_train_lmdb

echo "Creating testing lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --shuffle \
    $DATA_ROOT \
    $TEST_DATA_ROOT/test.txt \
    $BIN_DIR/mnist_test_lmdb

echo "Done."
