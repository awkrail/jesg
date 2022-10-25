import os
import json
import numpy as np
import argparse
from statistics import mean

def compute_max_tIoU_hist(input_dir, N, split_type, dset_name):
    def compute_tIoU(pred, gt):
        start_i, end_i = pred
        start, end = gt
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    if dset_name == "yc2":
        input_path = os.path.join(input_dir, dset_name + "_tsn_pdvc_" + str(N), split_type + "_results_anet.json")
    else:
        input_path = os.path.join(input_dir, dset_name + "_c3d_pdvc_" + str(N), split_type + "_results_anet.json")

    with open(input_path) as f:
        input_data = json.load(f)

    iou_scores = []
    all_zero_values = []

    if split_type == "train":
        split_type_name = 'training'
    else:
        split_type_name = 'validation'

    # search oracle selection
    tiou_zero_ratio = 0
    tiou_total = 0

    for vid, ann in input_data.items():
        oracle_outputs = [ann["outputs"][index] for index in ann["max_indices"]]
        subset = ann["annotation"]['subset']
        annotations = ann["annotation"]['annotations']
        if subset == split_type_name:
            assert len(oracle_outputs) == len(annotations)
            for oracle_output, gt_segment in zip(oracle_outputs, annotations):
                tiou = compute_tIoU(oracle_output["timestamp"], gt_segment["segment"])
                if tiou < 0.1:
                    tiou_zero_ratio += 1
                tiou_total += 1
    print(tiou_zero_ratio / tiou_total)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", required=True, type=str, help="directory of saved results as a anet format")
    args = parser.parse_args()

    Ns = [25, 50, 100, 200]
    print("Dataset: YouCook2")
    for N in Ns:
        print("-" * 89)
        print("N=", N)
        print("type: train")
        compute_max_tIoU_hist(args.input_dir, N, split_type="train", dset_name="yc2")
        print("type: val")
        compute_max_tIoU_hist(args.input_dir, N, split_type="val", dset_name="yc2")
        print("-" * 89)

    Ns = [5, 10, 20, 40]
    print("Dataset: ActivityNet")
    for N in Ns:
        print("-" * 89)
        print("N=", N)
        print("type: train")
        compute_max_tIoU_hist(args.input_dir, N, split_type="train", dset_name="anet")
        print("type: val")
        compute_max_tIoU_hist(args.input_dir, N, split_type="val", dset_name="anet")
        print("-" * 89)


if __name__ == "__main__":
    main()
