"""
compute histogram of predicted events, between PDVC and our model
"""
import json

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def compute_num_of_pred_events(outputs):
    nums = {k:len(x) for k, x in outputs["results"].items()}
    return nums

def compute_num_of_gt_events(gt_outputs):
    nums = {k:len(v["timestamps"]) for k,v in gt_outputs.items()}
    return nums

def compute_cumsum_and_print(acc, sample_num):
    acc[1] = acc[0]+acc[1]
    acc[2] = acc[1]+acc[2]
    acc[3] = acc[2]+acc[3]
    print("========================")
    print("sample_num: ", sample_num)
    print("=0: ", acc[0]/sample_num)
    print("<1: ", acc[1]/sample_num)
    print("<2: ", acc[2]/sample_num)
    print("<3: ", acc[3]/sample_num)
    print("========================")

def compute_histo_of_pred_events(ev_nums):
    histo_arr = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    for k, v in ev_nums.items():
        if v>=12:
            histo_arr[12] += 1
        else:
            histo_arr[v] += 1
    print("histgram: ", histo_arr)

def main():
    pdvc_path = "/mnt/LSTA6/data/nishimura/jesg/models/dvc_models/yc2_tsn_pdvc_100/2021-11-22-15-33-25_yc2_tsn_pdvc_v_2021-11-01-10-10-44_epoch19_num457_alpha1.0.json_rerank_alpha1.0_temp2.0.json"
    ours_path = "/mnt/LSTA6/data/nishimura/jesg/models/jesg_models/yc2/mart_mil_joint_100_tau_0.5_vonly_best_greedy_pred_val.json"
    gt_path = "/home/nishimura/research/recipe_generation/jesg/densevid_eval/our_yc2_data/val_yc2.json"

    #pdvc_path = "/mnt/LSTA6/data/nishimura/jesg/models/dvc_models/anet_c3d_pdvc_10/prediction/num4917_epoch19.json_rerank_alpha1.0_temp2.0.json"
    #ours_path = "/mnt/LSTA6/data/nishimura/jesg/models/jesg_models/anet/mart_clip4clip_joint_10_tau_0.5_vonly_reconst_best_greedy_pred_val.json"
    #gt_path = "/mnt/LSTA6/data/nishimura/ActivityNet/captions/val_1.json"

    pdvc_outputs = load_json(pdvc_path)
    ours_outputs = load_json(ours_path)
    gt_outputs = load_json(gt_path)

    pdvc_ev_nums = compute_num_of_pred_events(pdvc_outputs)
    ours_ev_nums = compute_num_of_pred_events(ours_outputs)
    gt_ev_nums = compute_num_of_gt_events(gt_outputs)

    # accuracy of num of events
    pdvc_acc = {0:0, 1:0, 2:0, 3:0}
    ours_acc = {0:0, 1:0, 2:0, 3:0}

    for vid, gt_ev_num in gt_ev_nums.items():
        if vid in pdvc_ev_nums and vid in ours_ev_nums:
            if abs(pdvc_ev_nums[vid]-gt_ev_num)<4:
                pdvc_diff = abs(pdvc_ev_nums[vid]-gt_ev_num)
                pdvc_acc[pdvc_diff] += 1
            
            if abs(ours_ev_nums[vid]-gt_ev_num)<4:
                ours_diff = abs(ours_ev_nums[vid]-gt_ev_num)
                ours_acc[ours_diff] += 1

    # print accuracy of events
    print("PDVC baseline")
    compute_cumsum_and_print(pdvc_acc, len(pdvc_ev_nums))
    print("Our model")
    compute_cumsum_and_print(ours_acc, len(ours_ev_nums))

    # print distribution of event predicts
    print("PDVC baseline")
    compute_histo_of_pred_events(pdvc_ev_nums)
    print("Our model")
    compute_histo_of_pred_events(ours_ev_nums)
    print("Ground Truth")
    compute_histo_of_pred_events(gt_ev_nums)

if __name__ == "__main__":
    main()
