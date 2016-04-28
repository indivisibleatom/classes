rm -r bin
rm -r logs

mkdir bin
mkdir logs

CAFFE_TOOLS=$CAFFE_ROOT/build/tools

source scripts/populate_${1}_labels.sh
source scripts/createLMDB.sh ${1} ${2}
source scripts/train.sh ${1} 2>&1 | tee logs/train_${1}.log
