mkdir -p bin
mkdir -p logs
mkdir -p snapshots

CAFFE_TOOLS=${CAFFE_ROOT}/build/tools
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CAFFE_ROOT}/distribute/lib

if [ "${2}" == "--arch1" ]; then
  LOG_FILE_POSTFIX="${1}1"
elif [ "${2}" == "--arch2" ]; then
  LOG_FILE_POSTFIX="${1}2"
else
  LOG_FILE_POSTFIX="${1}"
fi

source scripts/setup_${1}.sh
source scripts/createLMDB.sh "$@" 2>&1 | tee logs/train_${LOG_FILE_POSTFIX}.log
source scripts/train.sh "$@" 2>&1 | tee -a logs/train_${LOG_FILE_POSTFIX}.log
