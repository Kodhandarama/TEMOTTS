"""ESPnet building blocks used by ESPnet2 FastSpeech2 (ported, same structure/defaults).

Sources: espnet/nets/pytorch_backend/transformer/{attention,embedding,encoder,encoder_layer,
layer_norm,multi_layer_conv}.py, espnet/nets/pytorch_backend/fastspeech/{duration_predictor,
length_regulator}.py, espnet2/tts/fastspeech2/variance_predictor.py,
espnet/nets/pytorch_backend/tacotron2/decoder.py (Postnet), espnet2/torch_utils/initialize.py.
"""
import math

import torch
from torch import nn


def make_pad_mask(lengths, maxlen=None):
    """True at padded positions. lengths: LongTensor (B,)."""
    maxlen = int(lengths.max()) if maxlen is None else maxlen
    return torch.arange(maxlen, device=lengths.device)[None, :] >= lengths[:, None]


def make_non_pad_mask(lengths, maxlen=None):
    return ~make_pad_mask(lengths, maxlen)


def pad_list(xs, pad_value=0.0):
    n_batch, max_len = len(xs), max(x.size(0) for x in xs)
    pad = xs[0].new_full((n_batch, max_len, *xs[0].size()[1:]), pad_value)
    for i, x in enumerate(xs):
        pad[i, : x.size(0)] = x
    return pad


class LayerNorm(nn.LayerNorm):
    """ESPnet LayerNorm (eps 1e-12), optionally over a non-last dim."""

    def __init__(self, nout, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(self.dim, -1)).transpose(self.dim, -1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (B, 1, *, T2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(self.attn), value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None and self.pe.size(1) >= x.size(1):
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        return self.dropout(x * self.xscale + self.pe[:, : x.size(1)])


class ScaledPositionalEncoding(PositionalEncoding):
    """x + alpha * PE (no sqrt(d) scaling), alpha trainable."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        self.extend_pe(x)
        return self.dropout(x + self.alpha * self.pe[:, : x.size(1)])


class MultiLayeredConv1d(nn.Module):
    """Conv1d position-wise FFN of FastSpeech."""

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        super().__init__()
        self.w_1 = nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = nn.Conv1d(hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, mask):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1))
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        return x, mask


class Encoder(nn.Module):
    """ESPnet transformer Encoder (also used as the FastSpeech2 decoder with input_layer=None)."""

    def __init__(self, attention_dim=384, attention_heads=4, linear_units=1536, num_blocks=6, dropout_rate=0.1,
                 positional_dropout_rate=0.1, attention_dropout_rate=0.0, input_layer=None,
                 pos_enc_class=ScaledPositionalEncoding, normalize_before=True, concat_after=False,
                 positionwise_conv_kernel_size=1):
        super().__init__()
        if isinstance(input_layer, nn.Module):
            self.embed = nn.Sequential(input_layer, pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            self.embed = nn.Sequential(pos_enc_class(attention_dim, positional_dropout_rate))
        self.normalize_before = normalize_before
        self.encoders = nn.ModuleList([
            EncoderLayer(attention_dim, MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                         MultiLayeredConv1d(attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate),
                         dropout_rate, normalize_before, concat_after)
            for _ in range(num_blocks)
        ])
        if normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        xs = self.embed(xs)
        for layer in self.encoders:
            xs, masks = layer(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


class DurationPredictor(nn.Module):
    """Predicts log(d + offset); inference returns integer durations."""

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super().__init__()
        self.offset = offset
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv1d(idim if i == 0 else n_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                          nn.ReLU(), LayerNorm(n_chans, dim=1), nn.Dropout(dropout_rate))
            for i in range(n_layers)
        ])
        self.linear = nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # log domain
        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0)
        return xs

    def forward(self, xs, x_masks=None):
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        return self._forward(xs, x_masks, True)


class VariancePredictor(nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, bias=True, dropout_rate=0.5):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv1d(idim if i == 0 else n_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
                          nn.ReLU(), LayerNorm(n_chans, dim=1), nn.Dropout(dropout_rate))
            for i in range(n_layers)
        ])
        self.linear = nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, 2))  # (B, T, 1)
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs


class LengthRegulator(nn.Module):
    def __init__(self, pad_value=0.0):
        super().__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, alpha=1.0):
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1
        return pad_list([torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)], self.pad_value)


class Postnet(nn.Module):
    def __init__(self, odim, n_layers=5, n_chans=512, n_filts=5, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        self.postnet = nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            mods = [nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False)]
            if use_batch_norm:
                mods.append(nn.BatchNorm1d(ochans))
            self.postnet.append(nn.Sequential(*mods, nn.Tanh(), nn.Dropout(dropout_rate)))
        ichans = n_chans if n_layers != 1 else odim
        mods = [nn.Conv1d(ichans, odim, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False)]
        if use_batch_norm:
            mods.append(nn.BatchNorm1d(odim))
        self.postnet.append(nn.Sequential(*mods, nn.Dropout(dropout_rate)))

    def forward(self, xs):
        for f in self.postnet:
            xs = f(xs)
        return xs


def initialize(model, init="xavier_uniform"):
    """espnet2.torch_utils.initialize (non-chainer branch)."""
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError(init)
    for name, p in model.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.zero_()
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.LayerNorm, nn.GroupNorm)):
            m.reset_parameters()
