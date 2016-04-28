#!/usr/bin/env sh


if [ "${2}" == "--arch1" ]; then
  cmd_line_args="--solver=models/${1}1_solver.prototxt"
elif [ "${2}" == "--arch2" ]; then
  cmd_line_args="--solver=models/${1}2_solver.prototxt"
else
  cmd_line_args="--solver=models/${1}_solver.prototxt"
fi

if [ "${3}" == "--finetune" ]; then
  cmd_line_args="${cmd_line_args} -weights snapshots/${1}_init.caffemodel"
fi

echo "*************TRAINING**************************"
echo "Command line for caffe train = ${cmd_line_args}"

$CAFFE_TOOLS/caffe train ${cmd_line_args}
