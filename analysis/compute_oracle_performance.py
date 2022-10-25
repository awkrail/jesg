"""
oracle選択を行なった時のパフォーマンスを測定
YouCook2, AcitivityNetともに対応
"""
import os
import json
import argparse
import subprocess
import numpy as np

from statistics import mean

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

def merge_results(dvc_eval_output_path, soda_output_path):
    dvc_eval_data = load_json(dvc_eval_output_path)
    soda_eval_data = load_json(soda_output_path)
    dvc_eval_data.update(soda_eval_data)
    return dvc_eval_data

def gather_all_results(result_list):
    def compute_dvc_eval_avg(result_list, key):
        avg_score = mean([x["average_2021"][key] for x in result_list])
        return avg_score
    
    def compute_soda_avg(result_list, key):
        avg_score = mean([x[key][-1] for x in result_list])
        return avg_score
    
    def compute_tIoU_overlap_avg(result_list, key):
        details_score = [x["details_2021"][key] for x in result_list]
        return np.array(details_score).mean(axis=0).tolist()

    # dvc_eval
    b4_avg = compute_dvc_eval_avg(result_list, "Bleu_4")
    m_avg = compute_dvc_eval_avg(result_list, "METEOR")
    c_avg = compute_dvc_eval_avg(result_list, "CIDEr")

    # soda
    soda_m_avg = compute_soda_avg(result_list, "Meteor_soda")
    soda_c_avg = compute_soda_avg(result_list, "Cider_soda")
    soda_tIoU_avg = compute_soda_avg(result_list, "tIoU_soda")

    # tIoU details
    recall_avg = compute_dvc_eval_avg(result_list, "Recall")
    precision_avg = compute_dvc_eval_avg(result_list, "Precision")
    recall_avg_list = compute_tIoU_overlap_avg(result_list, "Recall")
    precision_avg_list = compute_tIoU_overlap_avg(result_list, "Precision")


    # merge them and output
    result = {
            "Bleu_4": b4_avg,
            "Meteor": m_avg,
            "Cider": c_avg,
            "SODA_Meteor": soda_m_avg,
            "SODA_Cider": soda_c_avg,
            "SODA_tIoU": soda_tIoU_avg,
            "Recall_threshold": recall_avg_list,
            "Precision_threshold": precision_avg_list,
            "Recall": recall_avg,
            "Precision": precision_avg
            }

    return result

def save_model_performance(dvc_format_paths, dvc_eval_output_path, soda_output_path, merged_output_path, references):
    result_list = []
    for i, dvc_format_path in enumerate(dvc_format_paths):
        tmp_dvc_eval_output_path =  dvc_eval_output_path + ".{}".format(i)
        tmp_soda_output_path = soda_output_path + ".{}".format(i)

        dvc_eval_cmd = ["python", "evaluate2021.py", "-s", dvc_format_path, "-ot", tmp_dvc_eval_output_path, "-v", "-r"] + references
        soda_cmd = ["python", "soda.py", "-p", dvc_format_path, "-ot", tmp_soda_output_path, "-v", "-r"] + references
        
        eval_tool_dir = "./densevid_eval3"
        soda_dir = os.path.join(eval_tool_dir, "SODA")

        subprocess.call(dvc_eval_cmd, cwd=eval_tool_dir)
        subprocess.call(soda_cmd, cwd=soda_dir)
        result = merge_results(tmp_dvc_eval_output_path, tmp_soda_output_path)
        result_list.append(result)
    
    # report results
    avg_result = gather_all_results(result_list)
    with open(merged_output_path, "w") as f:
        json.dump(avg_result, f)
    
    # remove temp files
    for i, dvc_format_path in enumerate(dvc_format_paths):
        tmp_dvc_eval_output_path =  dvc_eval_output_path + ".{}".format(i)
        tmp_soda_output_path = soda_output_path + ".{}".format(i)
        os.remove(tmp_dvc_eval_output_path)
        os.remove(tmp_soda_output_path)

def main(dset_name):
    if dset_name == "yc2":
        references = ["/mnt/LSTA6/data/nishimura/youcook2/features/yc2/val_yc2.json"]
        Ns = [25, 50, 100, 200]
        feature_name = "tsn"
        filenames = ["highest_tIoU_results.json"]
        baseline_prediction_file = "prediction/num457_epoch19.json_rerank_alpha1.0_temp2.0.json"
    else:
        references = ["/mnt/LSTA6/data/nishimura/ActivityNet/captions/val_1.json", "/mnt/LSTA6/data/nishimura/ActivityNet/captions/val_2.json"]
        Ns = [5, 10, 20, 40]
        feature_name = "c3d"
        filenames = ["highest_score_output_1.json", "highest_score_output_2.json"]
        baseline_prediction_file = "prediction/num4917_epoch19.json_rerank_alpha1.0_temp2.0.json"
    
    eval_tool_dir = "./densevid_eval3"
    soda_dir = os.path.join(eval_tool_dir, "SODA")
    
    dvc_file_root_dir = "/mnt/LSTA6/data/nishimura/jesg/models/dvc_models/"

    for n in Ns:
        data_dir = "{}_{}_pdvc_{}".format(dset_name, feature_name, n)

        # PDVC
        dvc_filepaths = [os.path.join(dvc_file_root_dir, data_dir, baseline_prediction_file)]
        dvc_eval_outputpath = os.path.join(dvc_file_root_dir, data_dir, "baseline_dvc_eval_results.json")
        soda_outputpath = os.path.join(dvc_file_root_dir, data_dir, "baseline_soda_results.json")
        merged_output_path = os.path.join(dvc_file_root_dir, data_dir, "baseline_merged_results.json")
        save_model_performance(dvc_filepaths, dvc_eval_outputpath, soda_outputpath, merged_output_path, references)

        # oracle
        oracle_dvc_filepath = [os.path.join(dvc_file_root_dir, data_dir, filename) for filename in filenames]
        dvc_eval_outputpath = os.path.join(dvc_file_root_dir, data_dir, "oracle_dvc_eval_results.json")
        soda_outputpath = os.path.join(dvc_file_root_dir, data_dir, "oracle_soda_results.json")
        merged_output_path = os.path.join(dvc_file_root_dir, data_dir, "oracle_merged_results.json")
        save_model_performance(oracle_dvc_filepath, dvc_eval_outputpath, soda_outputpath, merged_output_path, references)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", "-d", type=str, default="anet", help="dataset name: yc2 (or) anet")
    args = parser.parse_args()
    main(args.dset_name)
