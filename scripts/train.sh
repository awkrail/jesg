#!/usr/bin/env bash
feature=$1 # [resnet (yc2), mil]
joint=$2 # [joint, seperate]
query_num=$3 # yc2=[25, 50, 100, 200]
tau=$4 # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
model_path=$5 # path to save the model
v_feat_dir=$6 # path to save the feature path
dur_file=$7 # duration file

dset_name="yc2"
model_type="base"
data_dir="./densevid_eval3/json_data/our_${dset_name}_data_${query_num}"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
model_path="${model_path}/${model_type}_feat_${feature}_mode_${joint}_query_${query_num}_tau_${tau}"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"
max_n_sen=13
max_t_len=22  # including "BOS" and "EOS"
max_v_len=$((query_num+1))
dur_file="misc/${dset_name}_duration_frame.csv"

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=(--recurrent)

if [[ ${joint} == "joint" ]]; then # joint update of memories
    extra_args+=(--joint)
fi

if [[ ${feature} == "mil" ]]; then
    extra_args+=(--video_feature_size)
    extra_args+=(512)
elif [[ ${feature} == "resnet" ]]; then
    extra_args+=(--video_feature_size)
    extra_args+=(3072)
fi

python src/train.py \
--dset_name ${dset_name} \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--v_duration_file ${dur_file} \
--save_model ${model_path} \
--word2idx_path ${word2idx_path} \
--glove_path ${glove_path} \
--max_n_sen ${max_n_sen} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--feature ${feature} \
--query_num ${query_num} \
--tau ${tau} \
--exp_id init \
${extra_args[@]} 
