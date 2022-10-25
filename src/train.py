"""
This script handles the training process.
"""

import argparse
import math
import time

import random
import numpy as np
import os
import json
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from src.rtransformer.recursive_caption_dataset import \
    caption_collate, single_sentence_collate, prepare_batch_inputs
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from src.rtransformer.model import RecursiveTransformer, NonRecurTransformer, NonRecurTransformerUntied, TransformerXL
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.optimization import BertAdam, EMA
from src.translator import Translator
from src.translate import run_translate
from src.utils import save_parsed_args_to_json, save_json, load_json, \
    count_parameters, merge_dicts
from easydict import EasyDict as EDict
from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger(__name__)

def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(RCDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct

def train_epoch(model, training_data, optimizer, ema, device, opt, writer, epoch):
    model.train()

    total_cap_loss = 0
    total_event_loss = 0

    n_word_total = 0
    n_word_correct = 0

    torch.autograd.set_detect_anomaly(True)

    for batch_idx, batch in tqdm(enumerate(training_data), mininterval=2,
                                 desc="  Training =>", total=len(training_data)):
        niter = epoch * len(training_data) + batch_idx
        writer.add_scalar("Train/LearningRate", float(optimizer.param_groups[0]["lr"]), niter)
        if opt.recurrent:
            # prepare data
            batched_data = [prepare_batch_inputs(step_data, device=device, non_blocking=opt.pin_memory)
                            for step_data in batch[0]] # batch[0]に情報が全部入っている

            video_feature = batched_data[0]["video_feature"]
            event_feature = batched_data[0]["event_feature"]
            event_abs_feature = batched_data[0]["event_abs_feature"]
            video_mask = batched_data[0]["video_mask"]
            gt_timestamps_list = [e["gt_normed_timestamp"] for e in batched_data]
            event_labels_list = [e["max_index"] for e in batched_data]
            text_ids_list = [e["text_ids"] for e in batched_data]
            text_mask_list = [e["text_mask"] for e in batched_data]
            text_labels_list = [e["text_labels"] for e in batched_data]
            tiou_scores_list = [e["tiou_scores"] for e in batched_data]

            # forward & backward
            optimizer.zero_grad()
            cap_loss, event_loss, pred_scores_list = model(video_feature, video_mask, event_feature, event_abs_feature,
                                                           event_labels_list, text_ids_list, text_mask_list, text_labels_list, 
                                                           tiou_scores_list, gt_timestamps_list)
            loss = cap_loss + event_loss
            input_labels_list = text_labels_list

        loss.backward()
        if opt.grad_clip != -1:  # enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels_list):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(RCDataset.IGNORE)
            n_word += valid_label_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct
        total_cap_loss += cap_loss.item()
        total_event_loss += event_loss.item()

        if opt.debug:
            break

    torch.autograd.set_detect_anomaly(False)
    loss_per_word = 1.0 * total_cap_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    avg_event_loss = total_event_loss / len(training_data)
    return loss_per_word, accuracy, avg_event_loss


def eval_epoch(model, validation_data, device, opt):
    """The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference"""
    model.eval()

    total_cap_loss = 0
    total_event_loss = 0

    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc="  Validation =>"):
            if opt.recurrent:
                # prepare data
                batched_data = [prepare_batch_inputs(step_data, device=device, non_blocking=opt.pin_memory)
                                for step_data in batch[0]]

                video_feature = batched_data[0]["video_feature"]
                event_feature = batched_data[0]["event_feature"]
                event_abs_feature = batched_data[0]["event_abs_feature"]
                video_mask = batched_data[0]["video_mask"]
                gt_timestamps_list = [e["gt_normed_timestamp"] for e in batched_data]
                event_labels_list = [e["max_index"] for e in batched_data]
                text_ids_list = [e["text_ids"] for e in batched_data]
                text_mask_list = [e["text_mask"] for e in batched_data]
                text_labels_list = [e["text_labels"] for e in batched_data]
                tiou_scores_list = [e["tiou_scores"] for e in batched_data]

                cap_loss, event_loss, pred_scores_list = model(video_feature, video_mask, event_feature, event_abs_feature, 
                                                               event_labels_list, text_ids_list, text_mask_list, 
                                                               text_labels_list, tiou_scores_list, gt_timestamps_list)
                input_labels_list = text_labels_list

            # keep logs
            n_correct = 0
            n_word = 0
            for pred, gold in zip(pred_scores_list, input_labels_list):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(RCDataset.IGNORE)
                n_word += valid_label_mask.sum().item()

            n_word_total += n_word
            n_word_correct += n_correct
            total_cap_loss += cap_loss.item()
            total_event_loss += event_loss.item()

            if opt.debug:
                break

    loss_per_word = 1.0 * total_cap_loss / n_word_total
    accuracy = 1.0 * n_word_correct / n_word_total
    avg_event_loss = total_event_loss / len(validation_data)
    return loss_per_word, accuracy, avg_event_loss


def eval_language_metrics(checkpoint, eval_data_loader, opt, model=None, eval_mode="val"):
    """eval_mode can only be set to `val` here, as setting to `test` is cheating
    0, run inference
    1, Get METEOR, BLEU1-4, CIDEr scores
    2, Get vocab size, sentence length
    """
    translator = Translator(opt, checkpoint, model=model)
    json_res = run_translate(eval_data_loader, translator, opt=opt)
    res_filepath = os.path.abspath(opt.save_model + "_tmp_greedy_pred_{}.json".format(eval_mode))

    # byte string to sentence
    for json_key, json_data in json_res["results"].items():
        for data in json_res["results"][json_key]:
            data["sentence"] = data["sentence"].decode()
    
    save_json(json_res, res_filepath, save_pretty=True)

    if opt.dset_name == "ymk":
        reference_files_map = {"val": ["/home/nishimura/research/recipe_generation/jesg_base/densevid_eval/our_ymk_data_100/ymk_val.json"]}
    else:  # yc2
        reference_files_map = {"val": ["/home/nishimura/research/recipe_generation/jesg_base/densevid_eval/our_yc2_data_100/val_yc2.json"]}
    
    # COCO language evaluation w/ densevid eval
    eval_references = reference_files_map[eval_mode]
    lang_filepath_2018 = res_filepath.replace(".json", "_lang_2018.json")
    lang_filepath_2021 = res_filepath.replace(".json", "_lang_2021.json")
    lang_filepath_soda = res_filepath.replace(".json", "_lang_soda.json")

    # evaluate 2018 mode
    eval_cmd = ["python", "evaluate2018.py", "-s", res_filepath, "-ot", lang_filepath_2018, "-v", "-r"] + eval_references
    subprocess.call(eval_cmd, cwd=opt.eval_tool_dir)

    # evaluate 2021 mode
    eval_cmd = ["python", "evaluate2021.py", "-s", res_filepath, "-ot", lang_filepath_2021, "-v", "-r"] + eval_references
    subprocess.call(eval_cmd, cwd=opt.eval_tool_dir)

    # evaluate SODA (c)
    soda_dir = os.path.join(opt.eval_tool_dir, "SODA")
    eval_cmd = ["python", "soda.py", "-p", res_filepath, "-ot", lang_filepath_soda, "-v", "-r"] + eval_references
    subprocess.call(eval_cmd, cwd=soda_dir)

    # save results
    logger.info("Finished eval {}.".format(eval_mode))
    metric_filepaths = [lang_filepath_2018, lang_filepath_2021, lang_filepath_soda]
    all_metrics = merge_dicts([load_json(e) for e in metric_filepaths])
    all_metrics_filepath = res_filepath.replace(".json", "_all_metrics.json")
    save_json(all_metrics, all_metrics_filepath, save_pretty=True)
    return all_metrics, [res_filepath, all_metrics_filepath]


def train(model, training_data, validation_data, device, opt):
    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    if opt.ema_decay != -1:
        ema = EMA(opt.ema_decay)
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.register(name, p.data)
    else:
        ema = None

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")

    writer = SummaryWriter(opt.res_dir)
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy,SODA_meteor,SODA_cider,SODA_tiou,BLEU1,BLEU4,METEOR,ROUGE-L,CIDEr,Recall,Precision\n")

    prev_best_score = 0.
    es_cnt = 0
    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))

        # schedule sampling prob update, TODO not implemented yet

        start = time.time()
        if ema is not None and epoch_i != 0:  # use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc, avg_event_loss = train_epoch(
            model, training_data, optimizer, ema, device, opt, writer, epoch_i)
        logger.info("[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f}% \
                    event_loss: {event_loss}, elapse {elapse:3.3f} min".format(ppl=math.exp(min(train_loss, 100)), 
                     acc=100*train_acc, 
                     elapse=(time.time()-start)/60.,
                     event_loss=avg_event_loss))
        niter = (epoch_i + 1) * len(training_data)  # number of bart
        writer.add_scalar("Train/Acc", train_acc, niter)
        writer.add_scalar("Train/Loss", train_loss, niter)

        start = time.time()

        # Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # EMA model
        val_loss, val_acc, avg_event_loss = eval_epoch(model, validation_data, device, opt)
        logger.info("[Val]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f}% \
                    event_loss: {event_loss} elapse {elapse:3.3f} min".format(ppl=math.exp(min(val_loss, 100)), 
               acc=100*val_acc, 
               elapse=(time.time()-start)/60.,
               event_loss=avg_event_loss))
        writer.add_scalar("Val/Acc", val_acc, niter)
        writer.add_scalar("Val/Loss", val_loss, niter)

        # Note here we use greedy generated words to predicted next words, the true inference situation.
        checkpoint = {
            "model": model.state_dict(),  # EMA model
            "model_cfg": model.config,
            "opt": opt,
            "epoch": epoch_i}

        val_greedy_output, filepaths = eval_language_metrics(
            checkpoint, validation_data, opt, eval_mode="val", model=model)
        
        soda_cider = val_greedy_output["Cider_soda"][-1]
        soda_meteor = val_greedy_output["Meteor_soda"][-1]
        soda_tiou = val_greedy_output["tIoU_soda"][-1]
        bleu1 = val_greedy_output["average_2021"]["Bleu_1"]
        bleu4 = val_greedy_output["average_2021"]["Bleu_4"]
        meteor = val_greedy_output["average_2021"]["METEOR"]
        rouge_l = val_greedy_output["average_2021"]["ROUGE_L"]
        cider = val_greedy_output["average_2021"]["CIDEr"]
        recall = val_greedy_output["average_2021"]["Recall"]
        precision = val_greedy_output["average_2021"]["Precision"]

        logger.info("[Val] SODA:METEOR {sm:.2f}, SODA:CIDEr {sc:.2f}, \
                Bleu@1 {b1:.2f} Bleu@4 {b4:.2f} METEOR: {m:.2f}, ROUGE-L: {r:.2f}, \
                CIDEr: {c:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}"
                    .format(sm=soda_meteor*100,
                            sc=soda_cider*100,
                            b1=bleu1*100,
                            b4=bleu4*100,
                            m=meteor*100,
                            r=rouge_l*100,
                            c=cider*100,
                            recall=recall*100,
                            precision=precision*100))

        if opt.save_mode == "all":
            model_name = opt.save_model + "_e{e}_b{b}_m{m}_c{c}_r{r}.chkpt".format(
                e=epoch_i, b=round(bleu4*100, 2), m=round(meteor*100, 2),
                c=round(cider*100, 2), r=round(r4*100, 2))
            torch.save(checkpoint, model_name)
        elif opt.save_mode == "best":
            model_name = opt.save_model + ".chkpt"
            if soda_cider > prev_best_score:
                es_cnt = 0
                prev_best_score = soda_cider
                torch.save(checkpoint, model_name)
                new_filepaths = [e.replace("tmp", "best") for e in filepaths]
                for src, tgt in zip(filepaths, new_filepaths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if es_cnt > opt.max_es_cnt:  # early stop
                    logger.info("Early stop at {} with CIDEr {}".format(epoch_i, prev_best_score))
                    break
        cfg_name = opt.save_model + ".cfg.json"
        save_parsed_args_to_json(opt, cfg_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:

                log_tf.write("{epoch},{loss:8.5f},{ppl:8.5f},{acc:3.3f}\n".format(
                    epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), acc=100*train_acc))

                log_vf.write("{epoch},{loss:8.5f},{ppl:8.5f},{acc:3.3f},{sm:.2f},{sc:.2f},\
                        {st:.2f},{b1:.2f},{b4:.2f},{m:.2f},{r:.2f},{c:.2f},{recall:.2f},{precision:.2f}\n".format(
                        epoch=epoch_i, loss=val_loss, ppl=math.exp(min(val_loss, 100)), acc=100*val_acc,
                        sm=soda_meteor*100, sc=soda_cider*100, st=soda_cider*100, b1=bleu1*100, b4=bleu4*100, m=meteor*100, r=rouge_l*100,
                        c=cider*100, recall=recall*100, precision=precision*100))

    writer.close()

def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", type=str, default="ymk", choices=["yc2", "ymk"],
                        help="Name of the dataset, will affect data loader, evaluation, etc")

    # model config
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=300)
    parser.add_argument("--video_feature_size", type=int, default=512, help="2048 appearance + 1024 flow")
    parser.add_argument("--max_v_len", type=int, default=100, help="max length of video feature")
    parser.add_argument("--max_t_len", type=int, default=25,
                        help="max length of text (sentence or paragraph), 30 for ymk, 20 for yc2")
    parser.add_argument("--max_n_sen", type=int, default=6,
                        help="for recurrent, max number of sentences, 6 for ymk, 10 for yc2")
    parser.add_argument("--n_memory_cells", type=int, default=1, help="number of memory cells in each layer")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="video as 0, text as 1")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--memory_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("--freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("--share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    parser.add_argument("--recurrent", action="store_true", help="Run recurrent model")
    parser.add_argument("--untied", action="store_true", help="Run untied model")
    parser.add_argument("--xl", action="store_true", help="transformer xl model, when specified, "
                                                          "will automatically set recurrent = True, "
                                                          "since the data loading part is the same")
    parser.add_argument("--xl_grad", action="store_true",
                        help="enable back-propagation for xl model, only useful when `-xl` flag is enabled."
                             "Note, the original transformerXL model does not allow back-propagation.")
    parser.add_argument("--mtrans", action="store_true",
                        help="Masked transformer model for single sentence generation")

    # training config -- learning rate
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")

    parser.add_argument("--data_dir", required=True, help="dir containing the splits data files")
    parser.add_argument("--video_feature_dir", required=True, help="dir containing the video features")
    parser.add_argument("--v_duration_file", required=True, help="filepath to the duration file")
    parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.json")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_es_cnt", type=int, default=10,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=25, help="inference batch size")

    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")

    # others
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("---num_workers", type=int, default=0,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--exp_id", type=str, default="res", help="id of the current run")
    parser.add_argument("--res_root_dir", type=str, default="results", help="dir to containing all the results")
    parser.add_argument("--save_model", default="model")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval3")

    # additional arguments
    parser.add_argument("--feature", type=str, required=True, help="feature vectors used for the inputs to the model. Please specify from [resnet, mil, vswin]")
    parser.add_argument("--query_num", type=int, required=True, help="the number of queries")
    parser.add_argument("--joint", action="store_true", help="seperately learning memories or not")
    parser.add_argument("--tau", type=float, default=0.5, help="hyper-parameter tau of gumbel-softmax")
    parser.add_argument("--modality", type=str, required=True, help="input modality: vonly or multimodal")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.recurrent = True if opt.xl else opt.recurrent
    assert not (opt.recurrent and opt.untied), "cannot be True for both"
    assert not (opt.recurrent and opt.mtrans), "cannot be True for both"
    assert not (opt.untied and opt.mtrans), "cannot be True for both"
    if opt.xl_grad:
        assert opt.xl, "`-xl` flag must be set when using `-xl_grad`."

    if opt.recurrent:  # recurrent + xl
        if opt.xl:
            model_type = "xl_grad" if opt.xl_grad else "xl"
        else:
            model_type = "re"
    else:  # single sentence
        if opt.untied:
            model_type = "untied_single"
        elif opt.mtrans:
            model_type = "mtrans_single"
        else:
            model_type = "single"

    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.dset_name, model_type, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if opt.debug:
        opt.res_dir = "debug_" + opt.res_dir

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt


def main():
    opt = get_args()

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.dset_name == "yc2":
        train_video_feature_dir = os.path.join(opt.video_feature_dir, "training")
        val_video_feature_dir = os.path.join(opt.video_feature_dir, "validation")
    elif opt.dset_name == "ymk":
        train_video_feature_dir = opt.video_feature_dir
        val_video_feature_dir = opt.video_feature_dir
    else:
        raise ValueError("Expecting dset_name to be one of [`ymk`, `yc2`], got {}".format(opt.dset_name))

    train_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=train_video_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen, mode="train",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans, feature_name=opt.feature,
        query_num=opt.query_num, joint=opt.joint, modality=opt.modality)

    # add 10 at max_n_sen to make the inference stage use all the segments
    val_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=val_video_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen+10, mode="val",
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans, feature_name=opt.feature,
        query_num=opt.query_num, joint=opt.joint, modality=opt.modality)

    if opt.recurrent:
        collate_fn = caption_collate
    else:  # single sentence (including untied)
        collate_fn = single_sentence_collate
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    opt.vocab_size = len(train_dataset.word2idx)
    print(json.dumps(vars(opt), indent=4, sort_keys=True))

    device = torch.device("cuda" if opt.cuda else "cpu")
    rt_config = EDict(
        xl_grad=opt.xl_grad,  # enable back-propagation for transformerXL model
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        video_feature_size=opt.video_feature_size,
        max_position_embeddings=opt.max_v_len + opt.max_t_len,  # get from max_seq_len
        max_v_len=opt.max_v_len,  # max length of the videos
        max_t_len=opt.max_t_len,  # max length of the text
        type_vocab_size=opt.type_vocab_size,
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_hidden_layers=opt.num_hidden_layers,  # number of transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        n_memory_cells=opt.n_memory_cells,  # memory size will be (n_memory_cells, D)
        memory_dropout_prob=opt.memory_dropout_prob,
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight,
        feature_name=opt.feature,
        query_num=opt.query_num,
        joint=opt.joint,
        tau=opt.tau,
        modality=opt.modality,
    )

    if opt.recurrent:
        if opt.xl:
            logger.info("Use recurrent model - TransformerXL" + " (with gradient)" if opt.xl_grad else "")
            model = TransformerXL(rt_config)
        else:
            logger.info("Use recurrent model - Mine")
            model = RecursiveTransformer(rt_config)
    else:  # single sentence, including untied
        if opt.untied:
            logger.info("Use untied non-recurrent single sentence model")
            model = NonRecurTransformerUntied(rt_config)
        elif opt.mtrans:
            logger.info("Use masked transformer -- another non-recurrent single sentence model")
            model = MTransformer(rt_config)
        else:
            logger.info("Use non-recurrent single sentence model")
            model = NonRecurTransformer(rt_config)

    if opt.glove_path is not None:
        if hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    count_parameters(model)
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        count_parameters(model.embeddings.word_embeddings)

    train(model, train_loader, val_loader, device, opt)


if __name__ == "__main__":
    main()
