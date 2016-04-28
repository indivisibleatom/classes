#!/usr/bin/env sh

mkdir -p bin

if [ -f bin/Sunset_test.txt ]; then
  rm bin/mnist_test.txt
fi

if [ -f bin/Sunset_train.txt ]; then
  rm bin/mnist_train.txt
fi

cp data/mnist/train/train.txt > bin/mnist_train.txt
cp data/mnist/test/test.txt > bin/mnist_test.txt

