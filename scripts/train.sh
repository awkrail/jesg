#!/usr/bin/env bash
dset_name=$1  # [ymk, yc2]
model_type=$2  # [mart]
feature=$3 # [resnet (yc2), mil, resnet50 (ymk)]
joint=$4 # [joint, seperate]
query_num=$5 # yc2=[25, 50, 100, 200]
tau=$6 # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
modality=$7 # modalies: vonly (video-only), multimodal

data_dir="./densevid_eval/our_${dset_name}_data_${query_num}"
v_feat_dir="/mnt/LSTA6/data/nishimura/youcook2/features/training"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
model_path="/mnt/LSTA6/data/nishimura/jesg/models/jesg_models/${dset_name}/${model_type}_${feature}_${joint}_${query_num}_tau_${tau}_${modality}"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"
if [[ ${dset_name} == "ymk" ]]; then
    max_n_sen=15
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=$((query_num+1))
    v_feat_dir="/mnt/LSTA6/data/nishimura/YouMakeup/data/features/resnet50_npy"
    dur_file=".nonexisting_file" # duration file is not necessary for Youmakeup because fps of all of videos is 1.
elif [[ ${dset_name} == "yc2" ]]; then
    max_n_sen=13
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=$((query_num+1))
    v_feat_dir="/mnt/LSTA6/data/nishimura/youcook2/features/"
    dur_file="/mnt/LSTA6/data/nishimura/youcook2/features/yc2/${dset_name}_duration_frame.csv"
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=()
if [[ ${model_type} == "mart" ]]; then   # MART
    extra_args+=(--recurrent)
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

if [[ ${joint} == "joint" ]]; then # joint update of memories
    extra_args+=(--joint)
fi

if [[ ${feature} == "mil" ]]; then
    if [[ ${modality} == "vonly" ]]; then
        extra_args+=(--video_feature_size)
        extra_args+=(512)
    else
        extra_args+=(--video_feature_size)
        extra_args+=(1024)
    fi
elif [[ ${feature} == "resnet50" ]]; then
    extra_args+=(--video_feature_size)
    extra_args+=(2048)
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
--modality ${modality} \
--exp_id init \
${extra_args[@]} 
