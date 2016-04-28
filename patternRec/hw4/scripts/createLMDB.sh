#!/usr/bin/env sh

CAFFE_TOOLS=$CAFFE_ROOT/build/tools

DATA_ROOT=.
TRAIN_DATA_ROOT=$DATA_ROOT/data/${1}/train
TEST_DATA_ROOT=$DATA_ROOT/data/${1}/test
BIN_DIR=bin

if [ -f $BIN_DIR/${1}_train_lmdb ]; then
  rm -f $BIN_DIR/${1}_train_lmdb
fi

if [ -f $BIN_DIR/${1}_test_lmdb ]; then
  rm -f $BIN_DIR/${1}_test_lmdb
fi

convert_params="--shuffle"
if [ ${2} != 0 ]; then
  convert_params="$convert_params --${2}"
fi


echo "Creating training lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    ${convert_params} \
    $DATA_ROOT \
    $BIN_DIR/${1}_train.txt \
    $BIN_DIR/${1}_train_lmdb

echo "Creating testing lmdb..."

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    ${convert_params} \
    $DATA_ROOT \
    $BIN_DIR/${1}_test.txt \
    $BIN_DIR/${1}_test_lmdb

echo "Done."
