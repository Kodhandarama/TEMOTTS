"""Global Style Tokens (port of espnet2/tts/gst/style_encoder.py) with access to
the token combination weights:

* forward(speech, lengths) -> (style_emb, weights)   weights: (B, heads, tokens)
* from_weights(weights)    -> style_emb              Stage II / text-only inference

The reference-encoder GRU is run on packed sequences (ESPnet runs it over the
zero padding), so an utterance's GST weights do not depend on its batch.
"""
import torch
from torch import nn

from temotts.models.layers import MultiHeadedAttention as _BaseMHA


class ReferenceEncoder(nn.Module):
    def __init__(self, idim=80, conv_layers=6, conv_chans_list=(32, 32, 64, 64, 128, 128), conv_kernel_size=3,
                 conv_stride=2, gru_layers=1, gru_units=128):
        super().__init__()
        assert conv_kernel_size % 2 == 1
        padding = (conv_kernel_size - 1) // 2
        convs = []
        for i in range(conv_layers):
            in_ch = 1 if i == 0 else conv_chans_list[i - 1]
            convs += [nn.Conv2d(in_ch, conv_chans_list[i], kernel_size=conv_kernel_size, stride=conv_stride,
                                padding=padding, bias=False),
                      nn.BatchNorm2d(conv_chans_list[i]), nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*convs)
        self.conv_layers, self.kernel_size, self.stride, self.padding = conv_layers, conv_kernel_size, conv_stride, padding
        gru_in_units = idim
        for _ in range(conv_layers):
            gru_in_units = (gru_in_units - conv_kernel_size + 2 * padding) // conv_stride + 1
        gru_in_units *= conv_chans_list[-1]
        self.gru = nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def _out_lengths(self, lengths):
        for _ in range(self.conv_layers):
            lengths = torch.div(lengths - self.kernel_size + 2 * self.padding, self.stride, rounding_mode="floor") + 1
        return lengths.clamp(min=1)

    def forward(self, speech, lengths=None):
        batch_size = speech.size(0)
        hs = self.convs(speech.unsqueeze(1)).transpose(1, 2)  # (B, T', C, F')
        hs = hs.contiguous().view(batch_size, hs.size(1), -1)
        self.gru.flatten_parameters()
        if lengths is not None:
            hs = nn.utils.rnn.pack_padded_sequence(hs, self._out_lengths(lengths).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, ref_embs = self.gru(hs)
        return ref_embs[-1]  # (B, gru_units)


class _MHA(_BaseMHA):
    """Multi-head attention with different q/k/v input dims (ESPnet GST)."""

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        nn.Module.__init__(self)
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(q_dim, n_feat)
        self.linear_k = nn.Linear(k_dim, n_feat)
        self.linear_v = nn.Linear(v_dim, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)


class StyleTokenLayer(nn.Module):
    def __init__(self, ref_embed_dim=128, gst_tokens=10, gst_token_dim=256, gst_heads=4, dropout_rate=0.0):
        super().__init__()
        self.register_parameter("gst_embs", nn.Parameter(torch.randn(gst_tokens, gst_token_dim // gst_heads)))
        self.mha = _MHA(q_dim=ref_embed_dim, k_dim=gst_token_dim // gst_heads, v_dim=gst_token_dim // gst_heads,
                        n_head=gst_heads, n_feat=gst_token_dim, dropout_rate=dropout_rate)
        self.n_tokens, self.n_heads = gst_tokens, gst_heads

    def forward(self, ref_embs):
        batch_size = ref_embs.size(0)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        style_embs = self.mha(ref_embs.unsqueeze(1), gst_embs, gst_embs, None)
        weights = self.mha.attn.squeeze(2)  # (B, heads, tokens)
        return style_embs.squeeze(1), weights

    def from_weights(self, weights):
        """weights: (B, tokens) shared by all heads, or (B, heads, tokens).
        Returns linear_out(concat_h sum_k w_hk V_h(tanh(token_k))) -- the
        attention output with the given combination weights."""
        if weights.dim() == 2:
            weights = weights.unsqueeze(1).expand(-1, self.n_heads, -1)
        batch_size = weights.size(0)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        v = self.mha.linear_v(gst_embs).view(batch_size, -1, self.mha.h, self.mha.d_k).transpose(1, 2)
        x = torch.matmul(weights.unsqueeze(2).to(v.dtype), v)  # (B, h, 1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.mha.h * self.mha.d_k)
        return self.mha.linear_out(x).squeeze(1)


class StyleEncoder(nn.Module):
    def __init__(self, idim=80, gst_tokens=10, gst_token_dim=256, gst_heads=4, conv_layers=6,
                 conv_chans_list=(32, 32, 64, 64, 128, 128), conv_kernel_size=3, conv_stride=2, gru_layers=1,
                 gru_units=128):
        super().__init__()
        self.ref_enc = ReferenceEncoder(idim=idim, conv_layers=conv_layers, conv_chans_list=conv_chans_list,
                                        conv_kernel_size=conv_kernel_size, conv_stride=conv_stride,
                                        gru_layers=gru_layers, gru_units=gru_units)
        self.stl = StyleTokenLayer(ref_embed_dim=gru_units, gst_tokens=gst_tokens, gst_token_dim=gst_token_dim,
                                   gst_heads=gst_heads)

    def forward(self, speech, lengths=None):
        return self.stl(self.ref_enc(speech, lengths))

    def from_weights(self, weights):
        return self.stl.from_weights(weights)
