#!/usr/bin/env bash
# Usage:
# $ bash scripts/build_vocab.sh anet

dset_name=$1  # [ymk, yc2]
glove_path=$2 # /path/to/glove

echo "---------------------------------------------------------"
echo ">>>>>>>> Running on ${dset_name} Dataset"
if [[ ${dset_name} == "ymk" ]]; then
    min_word_count=1
    train_path="./densevid_eval/our_ymk_data_100/train_results_anet.json"
elif [[ ${dset_name} == "yc2" ]]; then
    min_word_count=3
    train_path="./densevid_eval/our_yc2_data/train_results_anet.json"
else
    echo "Wrong option for your first argument, select between ymk and yc2"
fi

python src/build_vocab.py \
--train_path ${train_path} \
--dset_name ${dset_name} \
--cache ./cache \
--min_word_count ${min_word_count} \
--raw_glove_path ${glove_path}
