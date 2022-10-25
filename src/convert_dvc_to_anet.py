import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_json_anet(annotation_path):
    def convert_anet_to_yc2_format(annotation, split_type):
        duration = annotation["duration"]
        timestamps = annotation["timestamps"]
        sentences = [sentence.lower() for sentence in annotation["sentences"]]

        yc2_format_dict = { 'duration':duration, 'subset':split_type }
        yc2_format_dict["annotations"] = []

        for i, (timestamp, sentence) in enumerate(zip(timestamps, sentences)):
            yc2_format_dict['annotations'].append({
                "segment": timestamp,
                "id": i,
                "sentence": sentence
                })
        return yc2_format_dict
        
    filenames = ['train.json', 'val_1.json', 'val_2.json']
    all_annotation_dict = { "database": {} }
    for filename in filenames:
        if filename == 'train.json':
            split_type = 'training'
        else:
            split_type = 'validation'

        with open(os.path.join(annotation_path, filename)) as f:
            data = json.load(f)

        for vid, annotation in data.items():
            all_annotation_dict["database"][vid[2:]] = convert_anet_to_yc2_format(annotation, split_type)
    return all_annotation_dict

def sort_dcap_outputs_with_start_seconds(dcap_outputs):
    dcap_outputs = dcap_outputs["results"]
    for vid, outputs in dcap_outputs.items():
        sorted_outputs = sorted(outputs, key=lambda x:x["timestamp"][0])
        dcap_outputs[vid] = sorted_outputs
    return dcap_outputs

def compute_maximum_tIoU_outputs(dcap_results, annotations):
    def compute_tIoU(pred, gt):
        start_i, end_i = pred
        start, end = gt
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    new_results = {}
    max_tIoU_arr = []

    for vid, outputs in dcap_results.items():
        duration = annotations["database"][vid[2:]]["duration"]
        gt_story = annotations["database"][vid[2:]]["annotations"]
        max_indices = []
        for idx, gt_segment in enumerate(gt_story):
            max_tIoU = 0
            max_index = 0
            for jdx, output in enumerate(outputs):
                tiou = compute_tIoU(output["timestamp"], gt_segment["segment"])
                if tiou > max_tIoU:
                    max_tIoU = tiou
                    max_index = jdx
            max_indices.append(max_index)
            max_tIoU_arr.append(max_tIoU)
        new_results[vid] = {
                "annotation" : annotations["database"][vid[2:]],
                "max_indices" : max_indices,
                "outputs" : outputs
            }
    return new_results, max_tIoU_arr

def bar_plot_tIoU(train_tIoU_arr, val_tIoU_arr):
    train_tIoU = np.array(train_tIoU_arr)
    val_tIoU = np.array(val_tIoU_arr)
    plt.hist([train_tIoU, val_tIoU], bins=10, stacked=False)
    plt.savefig("./tIoU.png")
    print("[Train] mean tIoU score: ", train_tIoU.mean())
    print("[Train] #samples of less than < 0.5: ", len(train_tIoU[train_tIoU < 0.5])/len(train_tIoU))
    print("[Val] mean tIoU score: ", val_tIoU.mean())
    print("[Val] #samples of less than < 0.5: ", len(val_tIoU[val_tIoU < 0.5])/len(val_tIoU))

def main(args):
    input_dir, output_dir, annotation_path, dset_name = args.input_dir, args.output_dir, args.annotation_path, args.dset_name
    
    train_path = os.path.join(input_dir, "results_train.json")
    val_path = os.path.join(input_dir, "results_val.json")

    train_dcap_outputs = load_json(train_path)
    val_dcap_outputs = load_json(val_path)

    if args.dset_name == "yc2":
        annotations = load_json(annotation_path)
    else:
        annotations = load_json_anet(annotation_path)

    # sort all output sequence with start seconds
    train_dcap_outputs = sort_dcap_outputs_with_start_seconds(train_dcap_outputs)
    val_dcap_outputs = sort_dcap_outputs_with_start_seconds(val_dcap_outputs)

    # create oracle (maximum tIoU results)
    train_max_outputs, train_tIoU_arr = compute_maximum_tIoU_outputs(train_dcap_outputs, annotations)
    val_max_outputs, val_tIoU_arr = compute_maximum_tIoU_outputs(val_dcap_outputs, annotations)
    
    # (Optional): bar plot for max tIoU scores
    bar_plot_tIoU(train_tIoU_arr, val_tIoU_arr)

    # save results
    out_train_path = os.path.join(output_dir, "train_results_anet.json")
    out_val_path = os.path.join(output_dir, "val_results_anet.json")
    save_json(train_max_outputs, out_train_path)
    save_json(val_max_outputs, out_val_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="input directory of DVC")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--annotation_path", "-a", type=str, required=True, help="the path for annotation data")
    parser.add_argument("--dset_name", "-d", type=str, required=True, help="name of dataset, ActivityNet (anet) or YouCook2 (yc2)")
    args = parser.parse_args()
    main(args)
