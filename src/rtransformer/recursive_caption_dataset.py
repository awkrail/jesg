import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import pickle

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """
    def __init__(self, dset_name, data_dir, video_feature_dir, duration_file, word2idx_path,
                 max_t_len, max_v_len, max_n_sen, mode="train", recurrent=True, untied=False,
                 feature_name="resnet", query_num=100, joint=True, modality="vonly"):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir  # a set of .h5 files
        self.duration_file = duration_file
        self.frame_to_second = self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen
        self.feature_name = feature_name
        self.query_num = query_num
        self.joint = joint
        self.modality = modality

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()
        self.num_sens = None  # number of sentence for each video, set in self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logging.info("Mode {}".format(mode))
        self.mode = mode
        if self.dset_name == "ymk":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "train_results_anet.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "val_results_anet.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`], got {}".format(mode))
        elif self.dset_name == "yc2":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "train_results_anet.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "val_results_anet.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`], got {}".format(mode))
        else:
            raise ValueError
        self._load_data(data_path)

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):

            video_name = e["name"][2:]
            if self.feature_name == "resnet":
                cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
                cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
                filepaths = [cur_path_resnet, cur_path_bn]
            elif self.feature_name == "mil":
                mode_folder = 'training' if self.mode == 'train' else 'validation'
                mil_dir = os.path.join("/mnt/LSTA6/data/nishimura/jesg/data/{}_mil/video_data_n".format(self.dset_name) + str(self.query_num), mode_folder)
                filepaths = [os.path.join(mil_dir, video_name + ".pkl")]
            elif self.feature_name == "resnet50":
                cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
                filepaths = [cur_path_resnet]

            for p in filepaths:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)

        print("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        print("Missing {}".format(set(self.missing_video_names)))
        self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        frame_to_second = {}
        
        if self.dset_name == "yc2":
            sampling_sec = 0.5  # hard coded, only support 0.5 (yc2)
        else:
            sampling_sec = 1. # YouMakeup

        if self.dset_name == "ymk":
            # for Youmakeup, all of the videos
            video_names = [x.replace("_resnet.npy", "") for x in os.listdir(self.video_feature_dir)]
            for video_name in video_names:
                frame_to_second[video_name] = 1.

        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)  # for yc2

        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))

        return frame_to_second

    def _load_data(self, data_path):
        """
        {
            "duration" : xxx,
            "timestamps" : [[start, end], [start, end], ...,],
            "sentences" : ["sent1", "sent2", ...]
        }
        """
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []

        for k, line in tqdm(raw_data.items()):
            """
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            """
            line["name"] = k
            line["pred_timestamps"] = [x["timestamp"] for x in line["outputs"]]
            line["sentences"] = [x["sentence"] for x in line["annotation"]["annotations"]][:self.max_n_sen] + ["[DUMMY]"]
            line["gt_timestamp"] = [x["segment"] for x in line["annotation"]["annotations"]][:self.max_n_sen] + [[0, 0]]
            line["max_indices"] = [x+1 for x in line["max_indices"][:self.max_n_sen]] + [0] # [0]=END OF EVENT
            line["duration"] = line["annotation"]["duration"]
            data.append(line)
        
        # z-score duration normalization
        dur_np = np.array([l["duration"] for l in data])
        dur_mean, dur_std = dur_np.mean(), dur_np.std()
        for line in data:
            line["norm_duration"] = (line["duration"] - dur_mean) / dur_std

        if self.recurrent:  # recurrent
            self.data = data

        logging.info("Loading complete! {} examples".format(len(self)))

    def convert_example_to_features(self, example):
        # 特徴ベクトルへ変換
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:]

        if self.feature_name == 'resnet':
            feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)
        elif self.feature_name == 'resnet50':
            feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            video_feature = np.load(feat_path_resnet)
        else:
            video_feature = None

        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature_untied(example["name"],
                                                                          example["pred_timestamps"],
                                                                          example["sentences"][clip_idx],
                                                                          example["max_indices"][clip_idx],
                                                                          example["duration"],
                                                                          example["norm_duration"],
                                                                          example["gt_timestamp"][clip_idx],
                                                                          video_feature)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature
            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["pred_timestamps"],
                                                 example["sentences"],
                                                 example["max_index"],
                                                 example["duration"],
                                                 example["norm_duration"],
                                                 example["gt_timestamp"],
                                                 video_feature)
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamps, sentence, max_index, duration, norm_duration, gt_timestamp, video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: np array
        """
        #frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]
        frm2sec = self.frame_to_second[name[2:]]

        # video + text tokens
        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_feature, timestamps, 
                                                                          frm2sec, duration, norm_duration,
                                                                          name)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        # 論文中のTE(token type embedding)のためのもの
        input_tokens = video_tokens + text_tokens

        # input_ids -> [[VIDEO..], [(Videoの)PAD..], [BOS], [words], [EOS], [PAD...]]
        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        # 単語の部分だけに単語のID, その他-1で構成されている -> CrossEntropyを計算するため
        input_labels = \
            [self.IGNORE] * len(video_tokens) + \
            [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + \
            [self.IGNORE]
        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        data = dict(
            name=name,
            input_tokens=input_tokens,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64), # input_ids -> [[VIDEO..], [(Videoの)PAD..], [BOS], [words], [EOS], [PAD...]]
            input_labels=np.array(input_labels).astype(np.int64), # input_labels -> [-1, -1, ..., (単語のところのID), -1, -1...]
            input_mask=np.array(input_mask).astype(np.float32), # [videoのmaskされていないところ(=1 or 0(maskあり))] + [textのmaskされていないところ(=1 or 0)]
            token_type_ids=np.array(token_type_ids).astype(np.int64), # [videoの部分=0] + [textの部分=1]
            video_feature=feat.astype(np.float32) # segmentぶんの特徴ベクトル
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamps, sentence, max_index, 
                                        duration, norm_duration, gt_timestamp, raw_video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            raw_video_feature: np array, N x D, for the whole video
        """
        if self.feature_name == "resnet" or self.feature_name == "resnet50":
            frm2sec = self.frame_to_second[name[2:]]
        else:
            frm2sec = None
        
        is_end_of_event = sentence == '[DUMMY]'

        # video + text tokens
        video_feature, event_feats, event_abs_feats, \
                video_mask = self._load_indexed_video_feature_untied(raw_video_feature, timestamps, 
                                                                     frm2sec, duration, norm_duration, name)
        tiou_scores = self._compute_tiou_scores(timestamps, gt_timestamp, is_end_of_event)
        gt_normed_timestamp = np.array(gt_timestamp) / duration

        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        text_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] + [self.IGNORE]

        data = dict(
            name=name,
            text_tokens=text_tokens,
            # model inputs
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            event_feature=event_feats.astype(np.float32),
            event_abs_feature=event_abs_feats.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
            max_index=np.array(max_index).astype(np.int64),
            tiou_scores=tiou_scores.astype(np.float32),
            gt_normed_timestamp=gt_normed_timestamp.astype(np.float32)
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamps,
            sentence=sentence,
            max_index=max_index,
            duration=duration,
            gt_timestamp=gt_timestamp
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _compute_tiou_scores(self, timestamps, gt_timestamp, is_end_of_event):
        def _compute_tIoU(pred, gt):
            start_i, end_i = pred
            start, end = gt
            intersection = max(0, min(end, end_i) - max(start, start_i))
            union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
            iou = float(intersection) / (union + 1e-8)
            return iou

        tiou_arr = np.zeros((len(timestamps)))
        for i, timestamp in enumerate(timestamps):
            tiou = _compute_tIoU(timestamp, gt_timestamp)
            tiou_arr[i] = tiou

        # add EOS
        tiou_arr = np.concatenate(([0], tiou_arr)) 
        if is_end_of_event:
            tiou_arr[0] = 1.
        
        return tiou_arr

    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l+1] = raw_feat[downsamlp_indices]  # truncate, sample???

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1 # valid_l -> videoの入っている分だけはVID_TOKENを入れる
            feat[1:valid_l+1] = raw_feat[st:ed + 1] ## ここでraw_feat(videoのベクトル)からst:edまでを抽出
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        return feat, video_tokens, mask

    def _load_indexed_video_feature_untied(self, raw_feat, timestamps, frm2sec, duration, norm_duration, name):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        def _zscore(feats):
            xmean = np.mean(feats, axis=0)
            xstd = np.std(feats, axis=0)
            return (feats - xmean) / (xstd + 1e-8)
        
        def _load_MIL_NCE_vfeat(name):
            path = "/mnt/LSTA6/data/nishimura/jesg/data/{}_mil/video_data_n".format(self.dset_name) + str(self.query_num)
            if self.mode == "train":
                mode = "training"
            else:
                mode = "validation"
            file_path = os.path.join(path, mode, name[2:] + ".pkl")
            with open(file_path, "rb") as f:
                vfeat = pickle.load(f)
            
            if self.modality == "multimodal":
                v_emb_feats = np.array([x["v_emb"] for x in vfeat])
                t_emb_feats = np.array([x["t_emb"] for x in vfeat])
                feat = np.concatenate([v_emb_feats, t_emb_feats], axis=1)
            else:
                feat = np.array([x["v_emb"] for x in vfeat])
            return feat
        
        def _load_resnet_vfeat(name, feat_dim, feat_len, n_queries):
            feat = np.zeros((n_queries, feat_dim))
            for i, timestamp in enumerate(timestamps):
                st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
                feat[i] = raw_feat[st:ed+1].mean(axis=0)
            return feat

        n_queries = self.query_num
        if self.feature_name == "mil":
            feat = _load_MIL_NCE_vfeat(name)
        elif self.feature_name == "resnet" or self.feature_name == "resnet50":
            feat_len = len(raw_feat)
            feat = _load_resnet_vfeat(name, raw_feat.shape[-1], feat_len, n_queries)
        else:
            print("Oops, your feature name may be wrong. Please specify it from [mil, resnet, coot]")
            exit(1)
        
        # I implemented event-level temporal representations as Wang et al. 2021.
        # See https://ieeexplore.ieee.org/document/9160989
        event_feats = np.zeros((n_queries, n_queries*3))
        log_eps = 1e-7
        for i, timestamp in enumerate(timestamps):
            i_start, i_end = timestamp
            i_length = i_end - i_start
            assert i_length > 0

            for j, timestamp in enumerate(timestamps):
                j_start, j_end = timestamp
                j_length = j_end - j_start
                assert j_length > 0
                
                start_elem = (i_start - j_start) / duration
                end_elem = (i_end - j_end) / duration
                length_elem = math.log((j_length / i_length) + log_eps)
                event_feats[i, j] = start_elem
                event_feats[i, n_queries+j] = end_elem
                event_feats[i, n_queries*2+j] = length_elem
        
        # Add absolute positional information of events: start and end timestamps
        event_abs_feats = np.array(timestamps) / duration
        feat = np.concatenate([np.zeros((1, feat.shape[-1])), feat], axis=0)
        event_feats = np.concatenate([np.zeros((1, n_queries*3)), event_feats], axis=0)
        event_abs_feats = np.concatenate([np.zeros((1, 2)), event_abs_feats], axis=0)
        mask = [1] * len(feat)
        return feat, event_feats, event_abs_feats, mask

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        if sentence == "[DUMMY]":
            max_t_len = self.max_t_len
            sentence_tokens = ["[DUMMY]"] * max_t_len
            mask = [0] * max_t_len
            return sentence_tokens, mask
        else:
            max_t_len = self.max_t_len
            sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
            sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]
            # pad
            valid_l = len(sentence_tokens)
            mask = [1] * valid_l + [0] * (max_t_len - valid_l)
            sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
            return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            # ここでbatchぶん回収
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            gt_timestamp=[],
            gt_sentence=[],
            max_indices=[],
            timestamp=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_timestamp"].append(d["gt_timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
            cur_meta["max_indices"].append(d["max_index"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["text_labels"][:] = RecursiveCaptionDataset.IGNORE
    padding_clip_sen_data["max_index"] = RecursiveCaptionDataset.IGNORE

    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta

def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"],
                   "gt_timestamp": e[1]["gt_timestamp"],
                   "max_index": e[1]["max_index"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
