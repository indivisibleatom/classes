#!/usr/bin/env sh

bash scripts/download_model_binary.sh Sunset

if [ -f bin/Sunset_test.txt ]; then
  rm bin/Sunset_test.txt
fi

if [ -f bin/Sunset_train.txt ]; then
  rm bin/Sunset_train.txt
fi

ls data/Sunset/train/sunset/ > bin/Sunset_train1.txt
sed -i 's/$/ 1/' bin/Sunset_train1.txt
sed -i 's/^/\/data\/Sunset\/train\/sunset\//' bin/Sunset_train1.txt

ls data/Sunset/train/nonsunset/ > bin/Sunset_train0.txt
sed -i 's/$/ 0/' bin/Sunset_train0.txt
sed -i 's/^/\/data\/Sunset\/train\/nonsunset\//' bin/Sunset_train0.txt

cat bin/Sunset_train1.txt > bin/Sunset_train.txt
cat bin/Sunset_train0.txt >> bin/Sunset_train.txt

rm bin/Sunset_train1.txt
rm bin/Sunset_train0.txt

ls data/Sunset/test/sunset/ > bin/Sunset_test1.txt
sed -i 's/$/ 1/' bin/Sunset_test1.txt
sed -i 's/^/\/data\/Sunset\/test\/sunset\//' bin/Sunset_test1.txt

ls data/Sunset/test/nonsunset/ > bin/Sunset_test0.txt
sed -i 's/$/ 0/' bin/Sunset_test0.txt
sed -i 's/^/\/data\/Sunset\/test\/nonsunset\//' bin/Sunset_test0.txt

cat bin/Sunset_test1.txt > bin/Sunset_test.txt
cat bin/Sunset_test0.txt >> bin/Sunset_test.txt

rm bin/Sunset_test1.txt
rm bin/Sunset_test0.txt
