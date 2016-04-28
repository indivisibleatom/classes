#!/usr/bin/env sh

DATASET=${1}
shift #Dataset name
shift #Architecture index
shift #Train or finetune
DATA_ROOT=.
TRAIN_DATA_ROOT=$DATA_ROOT/data/${DATASET}/train
TEST_DATA_ROOT=$DATA_ROOT/data/${DATASET}/test
BIN_DIR=bin
TOOLS_DIR=tools


if [ -f $BIN_DIR/${DATASET}_train_lmdb ]; then
  rm -f $BIN_DIR/${DATASET}_train_lmdb
fi

if [ -f $BIN_DIR/${DATASET}_test_lmdb ]; then
  rm -f $BIN_DIR/${DATASET}_test_lmdb
fi

convert_params="--shuffle $@"

echo "Creating training lmdb..."

cmd_line_train=" \
    ${TOOLS_DIR}/convert_imageset \
    ${convert_params} \
    $DATA_ROOT \
    ${BIN_DIR}/${DATASET}_train.txt \
    ${BIN_DIR}/${DATASET}_train_lmdb"

echo "Command to run $cmd_line_train"
GLOG_logtostderr=1 $cmd_line_train

echo "Creating testing lmdb..."

cmd_line_test=" \
    ${TOOLS_DIR}/convert_imageset \
    ${convert_params} \
    $DATA_ROOT \
    ${BIN_DIR}/${DATASET}_test.txt \
    ${BIN_DIR}/${DATASET}_test_lmdb"


echo "Command to run $cmd_line_test"
GLOG_logtostderr=1 $cmd_line_test

echo "Done."
