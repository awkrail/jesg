
from collections import defaultdict
import json

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def convert_val_to_submission(val2_data):
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    for vid, data in val2_data.items():
        for timestamp, sentence in zip(data['timestamps'], data['sentences']):
            batch_res['results'][vid].append({
                'sentence': sentence,
                'timestamp': timestamp
                })

    with open("./val2_submission.json", "w") as f:
        json.dump(batch_res, f)


def main():
    val2_data = load_json("/mnt/LSTA6/data/nishimura/ActivityNet/captions/val_2.json")
    val2_submission = convert_val_to_submission(val2_data)

if __name__ == "__main__":
    main()
