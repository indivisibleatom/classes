rm -r bin
rm -r logs

mkdir bin
mkdir logs

source scripts/create_mnist.sh
source scripts/train_mnist.sh 2>&1 | tee logs/train_mnist.log
