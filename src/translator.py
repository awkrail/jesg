""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F

from src.rtransformer.model import RecursiveTransformer
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.model_config.max_t_len
        self.max_v_len = self.model_config.max_v_len
        self.num_hidden_layers = self.model_config.num_hidden_layers

        if model is None:
            if opt.recurrent:
                if opt.xl:
                    logger.info("Use recurrent model - TransformerXL")
                    model = TransformerXL(self.model_config).to(self.device)
                else:
                    logger.info("Use recurrent model - Mine")
                    model = RecursiveTransformer(self.model_config).to(self.device)
            else:
                if opt.untied:
                    logger.info("Use untied non-recurrent single sentence model")
                    model = NonRecurTransformerUntied(self.model_config).to(self.device)
                elif opt.mtrans:
                    logger.info("Use masked transformer -- another non-recurrent single sentence model")
                    model = MTransformer(self.model_config).to(self.device)
                else:
                    logger.info("Use non-recurrent single sentence model")
                    model = NonRecurTransformer(self.model_config).to(self.device)
            # model = RecursiveTransformer(self.model_config).to(self.device)
            model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

        # self.eval_dataset = eval_dataset

    def translate_batch_greedy(self, video_features, event_features, event_abs_features,
                               video_masks, max_index, text_input_ids, text_masks, text_input_labels, rt_model):
        def greedy_decoding_step(v_prev_ms, t_prev_ms, video_feats, event_feats, event_abs_feats,
                                 v_masks, max_idx, text_input_id, text_mask, text_input_label, model,
                                 max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """RTransformer The first few args are the same to the input to the forward_step func
            Note:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            config = model.config
            max_t_len = config.max_t_len
            bsz = video_feats.shape[0]

            embeddings, embed_mask = model.compute_embeddings(video_feats, v_masks, event_feats, event_abs_feats)
            v_prev_ms, encoder_layer_outputs = model.encoder(v_prev_ms, embeddings, embed_mask, output_all_encoded_layers=False)
            selected_event_feats, attn_max_indices = model.select_event(video_prev_ms, encoder_layer_outputs[-1], max_idx, inference=True)

            text_input_ids = text_input_id.new_zeros(text_input_id.size())
            text_masks = text_mask.new_zeros(text_mask.size())
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )

            for dec_idx in range(max_t_len):
                text_input_ids[:, dec_idx] = next_symbols
                text_masks[:, dec_idx] = 1
                copied_t_prev_ms = copy.deepcopy(t_prev_ms)
                _, _, pred_scores = model.decode(copied_t_prev_ms, selected_event_feats, text_input_ids, text_masks, text_input_label)
                pred_scores[:, :, unk_idx] = -1e10
                next_words = pred_scores[:, dec_idx].max(1)[1]
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            #input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            text_input_ids, text_masks = mask_tokens_after_eos(text_input_ids, text_masks)
            t_prev_ms, _, _ = model.decode(t_prev_ms, selected_event_feats, text_input_ids, text_masks, text_input_label) 
            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return v_prev_ms, t_prev_ms, text_input_ids, attn_max_indices  # (N, max_t_len == L-max_v_len)

        config = rt_model.config
        with torch.no_grad():
            #prev_ms = [None] * config.num_hidden_layers
            video_prev_ms = [None] * config.num_hidden_layers
            text_prev_ms = [None] * config.num_hidden_layers
            step_size = len(text_input_labels)
            dec_seq_list = []
            selected_event_indices_list = []

            for idx in range(step_size):
                video_prev_ms, text_prev_ms, dec_seq, selected_event_indices = greedy_decoding_step(
                        video_prev_ms, text_prev_ms, video_features, event_features, event_abs_features,
                        video_masks, max_index[idx], text_input_ids[idx], text_masks[idx],
                        text_input_labels[idx], rt_model, config.max_v_len, config.max_t_len)

                if config.joint:
                    video_prev_ms, text_prev_ms = rt_model.memory_mixer(video_prev_ms, text_prev_ms)

                dec_seq_list.append(dec_seq)
                selected_event_indices_list.append(selected_event_indices)
            return dec_seq_list, selected_event_indices_list

    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        video_features, event_features, video_masks, event_abs_features, \
                max_index, text_input_ids, text_masks, text_input_labels = model_inputs
        return self.translate_batch_greedy(
                video_features, event_features, event_abs_features, video_masks, max_index,
                text_input_ids, text_masks, text_input_labels, self.model)
