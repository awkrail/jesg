import sys
import json
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
import snorkel.classification as snorkel_classification
logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if torch.isnan(context_layer).any():
            import ipdb; ipdb.set_trace()

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len+max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len+max_v_len:] = \
        torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemory, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertLayerVideoWithMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerVideoWithMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.memory_initilizer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = BertSelfAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = BertOutput(config)

    def forward(self, prev_m, hidden_states, attention_mask):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        max_t_len = 0
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)

        if prev_m is None:
            # only allow the initializer to see video part, not text part,
            # as it will be used for generation at current step
            init_memory_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initilizer(intermediate_output, init_memory_mask)  # (N, L, Di)

        # update memory, use raw attention_mask, no need to hide the text -> S_t^l ~ M_t^lの計算
        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)  # (N, M, Di)

        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # [(N, M, Di); (N, L, Di)] => [N, M+L, Di]
        bsz, n_memory_cells = prev_m.shape[:2]
        raw_memory_attention_mask = torch.cat(
            [attention_mask.new_ones(bsz, n_memory_cells), attention_mask], -1)  # (N, M+L)
        memory_attention_mask = make_pad_shifted_mask(
            raw_memory_attention_mask, max_v_len, max_t_len, memory_len=n_memory_cells)
        memory_attention_output = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, memory_attention_mask)  # (N, L, Di)
        memory_attention_output = self.memory_projection(memory_attention_output)  # (N, L, Di) -> (N, L, D)

        layer_output = self.output(memory_attention_output, attention_output)  # (N, L, D)
        return updated_m, layer_output

class BertEncoderWithVideoMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderWithVideoMemory, self).__init__()
        self.layer = nn.ModuleList([BertLayerVideoWithMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, prev_ms, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step. Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            prev_ms[layer_idx], hidden_states = layer_module(prev_ms[layer_idx], hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return prev_ms, all_encoder_layers

class BertLayerWithTextMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerWithTextMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.memory_initilizer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = BertSelfAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = BertOutput(config)

    def forward(self, prev_m, hidden_states, attention_mask, selected_event_feats):
        """
        Args:
            prev_m: (N, M, D)
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        max_v_len = 0
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)

        if prev_m is None:
            # only allow the initializer to see video part, not text part,
            # as it will be used for generation at current step
            #init_memory_mask = make_video_only_mask(attention_mask, max_v_len)
            init_memory_mask = torch.ones([selected_event_feats.shape[0], selected_event_feats.shape[1]]).cuda()
            prev_m = self.memory_initilizer(selected_event_feats, init_memory_mask)  # (N, L, Di)

        # update memory, use raw attention_mask, no need to hide the text -> S_t^l ~ M_t^lの計算
        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)  # (N, M, Di)

        concat_mh = torch.cat([prev_m, intermediate_output], dim=1)  # [(N, M, Di); (N, L, Di)] => [N, M+L, Di]
        bsz, n_memory_cells = prev_m.shape[:2]
        raw_memory_attention_mask = torch.cat(
            [attention_mask.new_ones(bsz, n_memory_cells), attention_mask], -1)  # (N, M+L)
        memory_attention_mask = make_pad_shifted_mask(
            raw_memory_attention_mask, max_v_len, max_t_len, memory_len=n_memory_cells)
        memory_attention_output = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, memory_attention_mask)  # (N, L, Di)
        memory_attention_output = self.memory_projection(memory_attention_output)  # (N, L, Di) -> (N, L, D)

        layer_output = self.output(memory_attention_output, attention_output)  # (N, L, D)

        return updated_m, layer_output

class BertDecoderWithTextMemory(nn.Module):
    def __init__(self, config):
        super(BertDecoderWithTextMemory, self).__init__()
        self.layer = nn.ModuleList([BertLayerWithTextMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, prev_ms, hidden_states, attention_mask, selected_event_feats, output_all_encoded_layers=True):
        """
        Args:
            prev_ms: [(N, M, D), ] * num_hidden_layers or None at first step. Memory states for each layer
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            prev_ms[layer_idx], hidden_states = layer_module(prev_ms[layer_idx], hidden_states, attention_mask, selected_event_feats)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return prev_ms, all_encoder_layers


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    論文3.2のH^0 = [H^0_{video} ; H^0_{text}]に対応. [.]はconcat. -> H^0を返すクラス
    """
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=config.max_position_embeddings)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, input_ids, video_features, token_type_ids):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:

        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + video_embeddings + token_type_embeddings
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


class BertEmbeddingsTextUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsTextUntied, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_text = PositionEncoding(n_filters=config.hidden_size,
                                                         max_len=config.max_position_embeddings)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, text_input_ids):
        """ text_input_ids: (N, Lt) """
        words_embeddings = self.word_fc(self.word_embeddings(text_input_ids))  # (N, Lt, D)
        words_embeddings = self.position_embeddings_text(words_embeddings)
        return words_embeddings  # (N, Lt, D)


class BertEmbeddingsVideoUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsVideoUntied, self).__init__()
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_video = PositionEncoding(n_filters=config.hidden_size,
                                                          max_len=config.max_position_embeddings)

    def forward(self, video_features):
        """ video_features: (N, Lv, D) """
        video_embeddings = self.video_embeddings(video_features)  # (N, Lv, D)
        video_embeddings = self.position_embeddings_video(video_embeddings)
        return video_embeddings  # (N, Lv, D)


class BertLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, diagonal_mask=False):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = attention_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = hidden_states.size(1)
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)
        attention_output = self.attention(hidden_states, self_attention_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemoryUntied(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, diagonal_mask=False, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertDecoderLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)  # (N, Lt, D)
        attention_output = self.norm1(attention_output + dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Is the attention mask correct?
        # Yes! Use the mask associated with key/value, not query. (query, key, value)
        # Additionally, there is no need to do subsequent masking, since each word has the right to see
        # all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            attention_output, enc_outputs, enc_outputs, enc_mask.unsqueeze(1))  # (N, Lt, D)
        dec_enc_attention_output = self.norm2(attention_output + dec_enc_attention_output)  # (N, Lt, D)

        # 3, linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return dec_enc_attention_output  # (N, Lt, D)


class BertDecoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertDecoderLayerNoMemoryUntied(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


class MemoryInitializer(nn.Module):
    def __init__(self, config):
        super(MemoryInitializer, self).__init__()
        # init memory
        self.n_memory_cells = config.n_memory_cells
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, config.n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            BertLayerNorm(config.hidden_size),
            nn.Dropout(config.memory_dropout_prob)
        )

    def forward(self, input_states, attention_mask):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            attention_mask: (N, L)
        """
        pooled_input_states = torch.sum(input_states * attention_mask.unsqueeze(-1), dim=1)   # (N, D)
        pooled_input_states = pooled_input_states / attention_mask.sum(1, keepdim=True)  # (N, D) no zero here
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory


class MemoryUpdater(nn.Module):
    def __init__(self, config):
        super(MemoryUpdater, self).__init__()
        self.memory_update_attention = BertSelfAttention(config)

        self.mc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.mz = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sz = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, prev_m, input_states, attention_mask):
        """ This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        # memory attended inputs
        n_memory_cells = prev_m.shape[1]
        update_mask = attention_mask.unsqueeze(1).repeat(1, n_memory_cells, 1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)  # (N, M, D),

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))  # (N, M, D)

        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))  # (N, M, D)

        updated_memory = (1 - z_t) * c_t + z_t * prev_m  # (N, M, D)
        return updated_memory


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


"""
TransformerXL specific modules, 
from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py 
"""


class PositionalEmbeddingXL(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbeddingXL, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))  # D
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # pos_seq (float)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x, zero_triu=False):
        """x: (Lq, Lk, N, N_h)"""
        zero_pad = torch.zeros([x.size(0), 1] + list(x.size()[2:]),
                               device=x.device, dtype=x.dtype)  # (Lq, 1, N, N_h)
        x_padded = torch.cat([zero_pad, x], dim=1)  # (Lq, 1+Lk, N, N_h)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])  # (Lk+1, Lq, N, N_h)

        x = x_padded[1:].view_as(x)

        if zero_triu:  # mask is applied outside
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, **inputs):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, verbose=False):
        """
        Args:
            w: input, (L, N, D)
            r: relative positional Embedding, (2L, 1, D) [sine, cosine]
            r_w_bias: (Nh, Nh)
            r_r_bias: (Nh, Nh)
            attn_mask:  (L, 2L, 1)
            mems:  0 or (L, N, D)
            verbose:

        Returns:

        """
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)  # (2L, N, D) or (L, N, D) when mems is empty
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)  # r (2L, 1, D) -->  r_head_k (2L, 1, N_h, Dh)
            if verbose:
                print("r: {}".format(r.shape))
                print("r_head_k: {}".format(r_head_k.shape))

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)  # scaled dot product, (L, 2L, N, Nh)
        if verbose:
            print("attn_score {}".format(attn_score.size()))

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None].bool(), -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)  # (L, 2L, N, Nh)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))  # (L, N, N_h, D_h)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)  # (L, N, D)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output  # (L, N, D)


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, verbose=False):
        """
        Args:
            dec_inp: input, (L, N, D)
            r:  relative positional Embedding, (2L, 1, D)
            r_w_bias: (Nh, Nh)
            r_r_bias: (Nh, Nh)
            dec_attn_mask:  (L, 2L, 1)
            mems:  0 or (L, N, D)
            verbose:

        Returns:

        """
        if verbose:
            print("------RelPartialLearnableDecoderLayer forward")
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems, verbose=verbose)  # (L, N, D)
        if verbose:
            print("output {}".format(output.size()))
        output = self.pos_ff(output)  # (L, N, D)
        if verbose:
            print("after pos_ff output {}".format(output.size()))
        return output


class TransformerXLEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerXLEncoder, self).__init__()
        self.xl_grad = config.xl_grad  # whether to enable back-propagation for the memory
        self.d_head = int(config.hidden_size / config.num_attention_heads)
        self.pos_emb = PositionalEmbeddingXL(config.hidden_size)
        self.r_w_bias = nn.Parameter(torch.Tensor(config.num_attention_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(config.num_attention_heads, self.d_head))
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([
            RelPartialLearnableDecoderLayer(n_head=config.num_attention_heads,
                                            d_model=config.hidden_size,
                                            d_head=self.d_head,
                                            d_inner=config.hidden_size,
                                            dropout=config.hidden_dropout_prob,
                                            dropatt=0,
                                            pre_lnorm=False)
            for _ in range(config.num_hidden_layers)
        ])

    def _update_mems(self, hids, mems):
        """
        hids: [(L, N, D)] * (n_layers + 1)
        mems: [(L, N, D)] * (n_layers + 1) or [(0), ] * (n_layers + 1)
        """
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # only use hidden_states from last segment
        with torch.no_grad():
            new_mems = []
            for i in range(len(hids)):
                new_mems.append(hids[i] if self.xl_grad else hids[i].detach())
        return new_mems

    def forward(self, mems, raw_embeddings, attention_mask):
        """
        Args:
            mems: [(L, N, D), ] * (num_hidden_layers + 1) or None at first step. Memory states for each layer
            raw_embeddings: (L, N, D), initial video-text embeddings
            attention_mask: (L, L or 2L, 1), dim=3
        Returns:

        """
        qlen, bsz = raw_embeddings.shape[:-1]
        mlen = len(mems[0]) if mems is not None else 0
        klen = mlen + qlen
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=raw_embeddings.device,
                               dtype=raw_embeddings.dtype)  # [start, end)  (2L, )
        pos_emb = self.pos_emb(pos_seq)  # pos_seq (2L, ), pos_emb (2L, 1, D)

        core_out = self.drop(raw_embeddings)
        pos_emb = self.drop(pos_emb)

        hidden_states_layerwise = [core_out, ]  # [word_emb] + [layer_output] * n_layers
        for layer_idx, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[layer_idx]
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                             dec_attn_mask=attention_mask, mems=mems_i, verbose=False)
            hidden_states_layerwise.append(core_out)

        core_out = self.drop(core_out)  # (L, N, D)
        new_mems = self._update_mems(hidden_states_layerwise, mems)  # [(L, N, D), ] * (num_hidden_layers + 1)
        return core_out, new_mems


class TransformerXL(nn.Module):
    def __init__(self, config):
        super(TransformerXL, self).__init__()
        self.config = config
        # since positional embedding is added after
        self.embeddings = BertEmbeddingsWithVideo(config, add_postion_embeddings=False)
        assert config.hidden_size % config.num_attention_heads == 0
        self.encoder = TransformerXLEncoder(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_mems(self):
        mems = []
        param = next(self.parameters())
        for i in range(self.config.num_hidden_layers + 1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)
        return mems

    def make_mask(self, input_mask, prev_seg_input_masks=None):
        """ 1, subsequent_mask (each word cannot see its subsequent word) + padding mask
            2, integrate memory mask, where only padding mask should be used (be careful when eval)
        Args:
            input_mask: (N, L), FloatTensor, padding mask, with 1 indicates non-pad bits,
            prev_seg_input_masks: (N, L) or None, FloatTensor, padding mask, no need to have subsequent mask?

        Returns:
            (N, L, L) or (N, 2L, L), ByteTensor, with 0 indicate valid bits,
        """
        subsequent_mask = make_shifted_mask(input_mask, self.config.max_v_len,
                                            self.config.max_t_len, memory_len=0)  # (N, L, L), subsequent mask
        attn_mask2 = subsequent_mask * input_mask.unsqueeze(1)  # (N, L, L), padding + subsequent mask, for input

        attn_mask1 = None if prev_seg_input_masks is None else \
            prev_seg_input_masks[:, :, None].expand_as(attn_mask2)  # (N, L, L)
        attn_mask = attn_mask2 if attn_mask1 is None else \
            torch.cat([attn_mask1, attn_mask2], dim=2)  # (N, L, L) or (N, L, 2L)

        attn_mask = torch.einsum("bij->ijb", attn_mask)   # (L, L, N) or (L, 2L, N)  query-key
        attn_mask = (1 - attn_mask).byte()  # with 1 indicates padding or subsequent locations
        return attn_mask

    def forward_step(self, prev_ms, input_ids, video_features, token_type_ids, input_masks, prev_masks):
        """single step forward in the recursive structure"""
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)  # (N, L, D)

        prev_ms = [e.transpose(0, 1) if e.numel() > 0 else e for e in prev_ms]  # (N, L, D) -> (L, N, D)
        embeddings = embeddings.transpose(0, 1)  # (N, L, D) -> (L, N, D)
        attn_mask = self.make_mask(input_masks, prev_masks)  # (L, L, N) or (L, 2L, N)

        last_layer_output, prev_ms = self.encoder(prev_ms, embeddings, attn_mask)
        last_layer_output.transpose_(0, 1)  # (N, L, D)
        prev_ms = [e.transpose(0, 1) for e in prev_ms]  # (L, N, D) -> (N, L, D)
        prediction_scores = self.decoder(last_layer_output)  # (N, L, vocab_size)
        return prev_ms, last_layer_output, prediction_scores

    def forward(self, input_ids_list, video_features_list, input_masks_list,
                token_type_ids_list, input_labels_list):
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            video_features_list: [(N, L, D_v)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids_list: [(N, L)] * step_size, with `0` on the first `max_v_len` bits, `1` on the last `max_t_len`
            input_labels_list: [(N, L)] * step_size, with `-1` on ignored positions

        Returns:

        """
        prev_ms = self.init_mems()  # will be initialized internally
        step_size = len(input_ids_list)
        memory_list = []  # [(N, M, D)] * (num_hidden_layers + 1) * step_size
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size
        for idx in range(step_size):
            prev_masks = None if idx == 0 else input_masks_list[idx-1]
            prev_ms, encoded_layer_output, prediction_scores = \
                self.forward_step(prev_ms, input_ids_list[idx], video_features_list[idx],
                                  token_type_ids_list[idx], input_masks_list[idx], prev_masks)
            memory_list.append(prev_ms)
            encoded_outputs_list.append(encoded_layer_output)
            prediction_scores_list.append(prediction_scores)

        # compute loss, get predicted words
        caption_loss = 0.
        for idx in range(step_size):
            caption_loss += self.loss_func(prediction_scores_list[idx].view(-1, self.config.vocab_size),
                                           input_labels_list[idx].view(-1))

        return caption_loss, prediction_scores_list  # , predicted_token_ids_list

class BertEmbeddingsEventUntied(nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsEventUntied, self).__init__()
        self.event_embeddings = nn.Sequential(
            BertLayerNorm(config.query_num*3, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.query_num*3, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
    
    def forward(self, event_features):
        event_embeddings = self.event_embeddings(event_features)
        return event_embeddings

class BertEmbeddingsEventAbsoluteUntied(nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsEventAbsoluteUntied, self).__init__()
        self.event_embeddings = nn.Sequential(
            nn.Linear(2, config.hidden_size), # 2=[start, end]
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
    
    def forward(self, event_features):
        event_embeddings = self.event_embeddings(event_features)
        return event_embeddings

class NonRecurTransformerUntied(nn.Module):
    def __init__(self, config):
        super(NonRecurTransformerUntied, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddingsTextUntied(config)  # text embedding
        self.video_embeddings = BertEmbeddingsVideoUntied(config)
        self.event_embeddings = BertEmbeddingsEventUntied(config)
        self.encoder = BertEncoderNoMemoryUntied(config)
        self.decoder = BertDecoderNoMemoryUntied(config)
        
        # score-predictor
        self.scorer = nn.Sequential(
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid())

        # text generation loss
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder_classifier = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.event_loss_func = LabelSmoothingLoss(config.label_smoothing, 100, ignore_index=-1) # event selection loss func
        self.tiou_score_loss_func = nn.MSELoss(reduction='sum')

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, video_features, video_masks, event_features):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
        """
        video_embeddings = self.video_embeddings(video_features)  # (N, Lv, D),
        event_embeddings = self.event_embeddings(event_features)  # (N, Lv, D)

        input_embeddings = video_embeddings + event_embeddings
        input_masks = video_masks

        encoder_outputs = self.encoder(
            input_embeddings, input_masks, diagonal_mask=False)[-1]  # (N, Lv, D)

        return encoder_outputs

    def decode(self, text_input_ids, text_masks, text_input_labels, encoder_outputs, encoder_masks):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lv, D)
            encoder_masks: (N, Lv)
        """
        text_embeddings = self.embeddings(text_input_ids)  # (N, Lt, D)
        decoder_outputs = self.decoder(
            text_embeddings, text_masks, encoder_outputs, encoder_masks, diagonal_mask=True)[-1]  # (N, Lt, D)
        prediction_scores = self.decoder_classifier(decoder_outputs)  # (N, Lt, vocab_size)
        caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                      text_input_labels.view(-1))
        return caption_loss, prediction_scores

    def select_event(self, encoder_outputs, event_labels, tiou_scores, inference=False):
        #h_selector = self.linear_encoder(encoder_outputs[:, 0])
        #h_encoded_outputs = self.linear_decoder(encoder_outputs[:, 1:])
        h_selector = encoder_outputs[:, 0]
        h_encoded_outputs = encoder_outputs[:, 1:]
        attn_energy = torch.einsum('ab,acb->ac', h_selector, h_encoded_outputs) / math.sqrt(self.config.hidden_size)
        
        # select event w/ gumbel softmax
        if inference:
            attn_max_indices = F.softmax(attn_energy, dim=-1).max(dim=1)[1]
            selected_event_feats = torch.zeros((attn_max_indices.shape[0], encoder_outputs.shape[-1])).cuda()
            for i, attn_max_index in enumerate(attn_max_indices):
                selected_event_feats[i] = encoder_outputs[:, 1:][i, attn_max_index]
            return selected_event_feats, attn_max_indices
            #attn_max_indices = F.softmax(attn_energy, dim=-1).max(dim=1)[1]
            #attn_prob = F.softmax(attn_energy, dim=-1)
            #selected_event_feats = torch.einsum('ab,abc->ac', attn_prob, encoder_outputs[:, 1:])
            #return selected_event_feats, attn_max_indices
        else:
            # replace the current loss_func with tIoU computation
            #tiou_softmax_scores = F.softmax(tiou_scores, dim=-1)
            #attn_logits = F.log_softmax(attn_energy, dim=-1)
            #event_loss = F.kl_div(attn_logits, tiou_softmax_scores, reduction='sum')

            event_loss = self.event_loss_func(attn_energy, event_labels)
            attn_logits = F.log_softmax(attn_energy, dim=-1)
            selected_events = F.gumbel_softmax(attn_logits, tau=0.5, hard=True)
            selected_event_feats = torch.einsum('ab,abc->ac', selected_events, encoder_outputs[:, 1:])
            #selected_event_feats = torch.einsum('ab,abc->ac', F.softmax(attn_energy, dim=-1), encoder_outputs[:, 1:])

            return event_loss, selected_event_feats

    def forward(self, video_features, video_masks, event_features, event_labels,
                text_input_ids, text_masks, text_input_labels, tiou_scores):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
        """
        encoder_outputs = self.encode(video_features, video_masks, event_features)  # (N, Lv, D)
        event_selection_loss, selected_event_feats = self.select_event(encoder_outputs, event_labels, tiou_scores)

        """
        caption_loss, prediction_scores = self.decode(
            text_input_ids, text_masks, text_input_labels, encoder_outputs, video_masks)
        return caption_loss, prediction_scores
        """
        event_mask = torch.ones((selected_event_feats.shape[0], 1)).cuda()
        selected_event_feats = selected_event_feats.unsqueeze(1)

        caption_loss, prediction_scores = self.decode(
            text_input_ids, text_masks, text_input_labels, selected_event_feats, event_mask)
        return caption_loss, event_selection_loss, prediction_scores


class NonRecurTransformer(nn.Module):
    def __init__(self, config):
        super(NonRecurTransformer, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddingsWithVideo(config, add_postion_embeddings=True)
        self.encoder = BertEncoderNoMemory(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, video_features, input_masks, token_type_ids, input_labels):
        """
        Args:
            input_ids: [(N, L)]
            video_features: [(N, L, D_v)] * step_size
            input_masks: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids: [(N, L)] * step_size, with `0` on the first `max_v_len` bits, `1` on the last `max_t_len`
            input_labels: [(N, L)] * step_size, with `-1` on ignored positions
        """
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)  # (N, L, D)

        encoded_layer_outputs = self.encoder(
            embeddings, input_masks, output_all_encoded_layers=False)  # both outputs are list
        prediction_scores = self.decoder(encoded_layer_outputs[-1])  # (N, L, vocab_size)

        if input_labels is not None:
            caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                          input_labels.view(-1))
        else:
            caption_loss = None
        return caption_loss, prediction_scores

class MemoryMixer(nn.Module):
    def __init__(self, config):
        super(MemoryMixer, self).__init__()
        self.config = config
        self.W_v = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_v2t = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_t = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.W_t2v = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, video_ms, text_ms):
        mixed_video_ms, mixed_text_ms = [], []
        for video_m, text_m in zip(video_ms, text_ms):
            wv_video_m = gelu(self.W_v(video_m))
            wt_text_m = gelu(self.W_t(text_m))

            cp_video_m = gelu(self.W_t2v(wt_text_m)) * wv_video_m
            cp_text_m = gelu(self.W_v2t(wv_video_m)) * wt_text_m

            mixed_video_ms.append(cp_video_m)
            mixed_text_ms.append(cp_text_m)

        return mixed_video_ms, mixed_text_ms

class RecursiveTransformer(nn.Module):
    def __init__(self, config):
        super(RecursiveTransformer, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddingsTextUntied(config)  # text embedding (word embeddings)
        self.video_embeddings = BertEmbeddingsVideoUntied(config) # video embedding
        self.event_embeddings = BertEmbeddingsEventUntied(config) # event embedding
        self.event_abs_embeddings = BertEmbeddingsEventAbsoluteUntied(config) # event embedding
        self.encoder = BertEncoderWithVideoMemory(config) # video encoder + event-selector
        self.decoder = BertDecoderWithTextMemory(config)  # text decoder

        if config.joint:
            self.memory_mixer = MemoryMixer(config) # memory mixer
        
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder_classifier = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.event_loss_func = LabelSmoothingLoss(config.label_smoothing, config.max_v_len, ignore_index=-1)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(self, video_prev_ms, text_prev_ms, video_feature, 
                     video_mask, event_feature, event_abs_feature, event_labels, text_ids, 
                     text_mask, text_labels):
        """single step forward in the recursive structure"""
        # video encode w/ memory
        embeddings, embed_mask = self.compute_embeddings(video_feature, video_mask, 
                                                         event_feature, event_abs_feature)
        video_prev_ms, encoded_layer_outputs = self.encoder(
            video_prev_ms, embeddings, embed_mask, output_all_encoded_layers=False)  # both outputs are list
        # select events
        event_loss, selected_event_feats = self.select_event(video_prev_ms, encoded_layer_outputs[-1], event_labels)
        # text decode w/ memory
        text_prev_ms, caption_loss, prediction_scores = self.decode(text_prev_ms, selected_event_feats, 
                                                                    text_ids, text_mask, text_labels)
        # re-predict events
        return video_prev_ms, text_prev_ms, encoded_layer_outputs, prediction_scores, \
                caption_loss, event_loss

    def decode(self, prev_ms, selected_event_feats, text_ids, text_mask, text_labels):
        word_embeddings = self.embeddings(text_ids)
        selected_event_feats = selected_event_feats.unsqueeze(1)
        embeddings = selected_event_feats + word_embeddings
        prev_ms, decoder_outputs = self.decoder(prev_ms, embeddings, text_mask, selected_event_feats)
        prediction_scores = self.decoder_classifier(decoder_outputs[-1])
        caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                      text_labels.view(-1))
        return prev_ms, caption_loss, prediction_scores

    def select_event(self, prev_ms, encoded_layer_output, event_labels, inference=False):
        pooled_memory = torch.cat(prev_ms, dim=1).max(dim=1)[0]
        attn_energy = torch.einsum('ab,acb->ac', pooled_memory, encoded_layer_output) / math.sqrt(self.config.hidden_size)

        if inference:
            attn_max_indices = F.softmax(attn_energy, dim=-1).max(dim=1)[1]
            selected_event_feats = torch.zeros((attn_max_indices.shape[0], encoded_layer_output.shape[-1])).cuda()
            for i, attn_max_index in enumerate(attn_max_indices):
                selected_event_feats[i] = encoded_layer_output[i, attn_max_index]
            return selected_event_feats, attn_max_indices
        else:
            attn_logits = F.log_softmax(attn_energy, dim=-1)
            selected_events = F.gumbel_softmax(attn_logits, tau=self.config.tau, hard=True)
            selected_event_feats = torch.einsum('ab,abc->ac', selected_events, encoded_layer_output)

            # compute loss
            event_loss = self.event_loss_func(attn_energy, event_labels)
            return event_loss, selected_event_feats
    
    def compute_embeddings(self, video_features, video_masks, event_features, event_abs_features):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
        """
        video_embeddings = self.video_embeddings(video_features)  # (N, Lv, D)
        event_embeddings = self.event_embeddings(event_features)  # (N, Lv, D)
        event_abs_embeddings = self.event_abs_embeddings(event_abs_features) # (N, L_ea, D)
        input_embeddings = video_embeddings + event_embeddings + event_abs_embeddings
        input_masks = video_masks
        return input_embeddings, input_masks

    def forward(self, video_feature, video_mask, event_feature, event_abs_feature,
                event_labels_list, text_ids_list, text_mask_list, text_labels_list, 
                return_memory=False):
        # [(N, M, D)] * num_hidden_layers, initialized internally
        video_prev_ms = [None] * self.config.num_hidden_layers
        text_prev_ms = [None] * self.config.num_hidden_layers
        step_size = len(event_labels_list)
        memory_list = []  # [(N, M, D)] * num_hidden_layers * step_size
        encoded_outputs_list = []  # [(N, L, D)] * step_size
        prediction_scores_list = []  # [(N, L, vocab_size)] * step_size

        total_caption_loss = 0
        total_event_loss = 0

        for idx in range(step_size):
            video_prev_ms, text_prev_ms, encoded_layer_outputs,\
                prediction_scores, caption_loss, event_loss = self.forward_step(video_prev_ms, text_prev_ms, 
                      video_feature, video_mask, event_feature, event_abs_feature, 
                      event_labels_list[idx], text_ids_list[idx], 
                      text_mask_list[idx], text_labels_list[idx])

            if self.config.joint:
                video_prev_ms, text_prev_ms = self.memory_mixer(video_prev_ms, text_prev_ms)

            memory_list.append([video_prev_ms, text_prev_ms])
            encoded_outputs_list.append(encoded_layer_outputs)
            prediction_scores_list.append(prediction_scores)

            total_caption_loss += caption_loss
            total_event_loss += event_loss

        return total_caption_loss, total_event_loss, prediction_scores_list

# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    max_v_len=100,  # max length of the videos
    max_t_len=30,  # max length of the text
    n_memory_cells=10,  # memory size will be (n_memory_cells, D)
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)
