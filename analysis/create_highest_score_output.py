import json
import argparse

def create_maximum_tIoU_outputs(input_dir, ann_dir):
    def compute_tIoU(pred, gt):
        start_i, end_i = pred
        start, end = gt
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou
    
    with open(input_dir) as f:
        input_data = json.load(f)
    with open(ann_dir) as f:
        ann_data = json.load(f)

    iou_scores = []

    for vid, ann in ann_data["database"].items():
        if ann["subset"] == "validation":
            new_results = []
            for segment in ann["annotations"]:
                sorted_results = []
                gt_segment = segment["segment"]

                for x in input_data["results"]["v_" + vid]:
                    tiou = compute_tIoU(x["timestamp"], gt_segment)
                    sorted_results.append([x, tiou])
                max_iou_output = sorted(sorted_results, key=lambda x:x[1], reverse=True)[0]
                new_results.append(max_iou_output[0])
                iou_scores.append(max_iou_output[1])
            input_data["results"]["v_" + vid] = new_results
    return input_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, required=True, help="input path of DVC outputs")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="output path of oracle selection")
    args = parser.parse_args()
    ann_path = "/mnt/LSTA6/data/nishimura/youcook2/annotations/youcookii_annotations_trainval.json"
    extracted_results = create_maximum_tIoU_outputs(args.input_path, ann_path)
    with open(args.output_path, "w") as f:
        json.dump(extracted_results, f)

if __name__ == "__main__":
    main()
