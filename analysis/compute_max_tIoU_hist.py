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

    def show_hist(iou_scores):
        threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        score_dist = []
        for i in range(len(threshold)-1):
            i_th_thres, next_thres = threshold[i], threshold[i+1]
            freq = len([score for score in iou_scores if i_th_thres <= score < next_thres])
            score_dist.append(freq)
        return score_dist

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
    for vid, ann in input_data.items():
        outputs = ann["outputs"]
        annotation = ann["annotation"]
        if annotation["subset"] == split_type_name:
            sorted_results_arr = []
            for segment in annotation["annotations"]:
                sorted_results = []
                gt_segment = segment["segment"]
                for x in outputs:
                    tiou = compute_tIoU(x["timestamp"], gt_segment)
                    sorted_results.append([x, tiou])

                sorted_results_arr.append([x[1] for x in sorted_results])
                max_iou_output = sorted(sorted_results, key=lambda x:x[1], reverse=True)[0]
                iou_scores.append(max_iou_output[1])
            
            sorted_results_arr = np.array(sorted_results_arr)
            all_zero_num = np.all(sorted_results_arr == 0, axis=0).sum()
            all_zero_values.append([all_zero_num, N])
    
    print("mean: ", mean(iou_scores))
    print("hist: ", show_hist(iou_scores))
    print("all zero percentage (%): ", sum([x[0] for x in all_zero_values])/sum([x[1] for x in all_zero_values]))

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
