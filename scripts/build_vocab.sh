#!/usr/bin/env bash
# Usage:
# $ bash scripts/build_vocab.sh anet

glove_path=$1 # /path/to/glove
dset_name="yc2"
min_word_count=3
train_path="./densevid_eval3/json_data/our_yc2_data_100/train_results_anet.json"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running on ${dset_name} Dataset"

python src/build_vocab.py \
--train_path ${train_path} \
--dset_name ${dset_name} \
--cache ./cache \
--min_word_count ${min_word_count} \
--raw_glove_path ${glove_path}
