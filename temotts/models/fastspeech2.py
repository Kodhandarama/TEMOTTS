"""FastSpeech2 + GST + speaker lookup table (port of espnet2/tts/fastspeech2/fastspeech2.py,
transformer encoder/decoder, reduction_factor 1).

Conditioning on the encoder output, as in ESPnet:  hs = enc(x) + gst(style) + sid_emb(spk)
"""
import torch
import torch.nn.functional as F
from torch import nn

from temotts.models.gst import StyleEncoder
from temotts.models.layers import (DurationPredictor, Encoder, LengthRegulator, Postnet, ScaledPositionalEncoding,
                                   PositionalEncoding, VariancePredictor, initialize, make_non_pad_mask,
                                   make_pad_mask)


class FastSpeech2Loss(nn.Module):
    """L1 (before+after postnet) + duration MSE (log domain) + pitch MSE + energy MSE, masked."""

    def __init__(self, use_masking=True):
        super().__init__()
        self.use_masking = use_masking
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, after_outs, before_outs, d_outs, p_outs, e_outs, ys, ds, ps, es, ilens, olens):
        if self.use_masking:
            out_masks = make_non_pad_mask(olens, ys.size(1)).unsqueeze(-1)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = after_outs.masked_select(out_masks) if after_outs is not None else None
            ys = ys.masked_select(out_masks)
            d_masks = make_non_pad_mask(ilens, ds.size(1))
            d_outs, ds = d_outs.masked_select(d_masks), ds.masked_select(d_masks)
            p_masks = d_masks.unsqueeze(-1)
            p_outs, ps = p_outs.masked_select(p_masks), ps.masked_select(p_masks)
            e_outs, es = e_outs.masked_select(p_masks), es.masked_select(p_masks)
        l1_before = self.l1(before_outs, ys)
        l1_after = self.l1(after_outs, ys) if after_outs is not None else torch.zeros_like(l1_before)
        duration_loss = self.mse(d_outs, torch.log(ds.float() + 1.0))
        pitch_loss = self.mse(p_outs, ps)
        energy_loss = self.mse(e_outs, es)
        return l1_before, l1_after, duration_loss, pitch_loss, energy_loss


class FastSpeech2(nn.Module):
    def __init__(self, idim, odim=80, adim=384, aheads=2, elayers=4, eunits=1536, dlayers=4, dunits=1536,
                 postnet_layers=5, postnet_chans=256, postnet_filts=5, postnet_dropout_rate=0.5,
                 positionwise_conv_kernel_size=3, use_scaled_pos_enc=True, use_batch_norm=True,
                 encoder_normalize_before=True, decoder_normalize_before=True,
                 encoder_concat_after=False, decoder_concat_after=False,
                 transformer_enc_dropout_rate=0.2, transformer_enc_positional_dropout_rate=0.2,
                 transformer_enc_attn_dropout_rate=0.2, transformer_dec_dropout_rate=0.2,
                 transformer_dec_positional_dropout_rate=0.2, transformer_dec_attn_dropout_rate=0.2,
                 duration_predictor_layers=2, duration_predictor_chans=256, duration_predictor_kernel_size=3,
                 duration_predictor_dropout_rate=0.1,
                 energy_predictor_layers=2, energy_predictor_chans=256, energy_predictor_kernel_size=3,
                 energy_predictor_dropout=0.5, energy_embed_kernel_size=1, energy_embed_dropout=0.0,
                 stop_gradient_from_energy_predictor=False,
                 pitch_predictor_layers=5, pitch_predictor_chans=256, pitch_predictor_kernel_size=5,
                 pitch_predictor_dropout=0.5, pitch_embed_kernel_size=1, pitch_embed_dropout=0.0,
                 stop_gradient_from_pitch_predictor=True,
                 spks=2,
                 use_gst=True, gst_tokens=16, gst_heads=1, gst_conv_layers=6,
                 gst_conv_chans_list=(32, 32, 64, 64, 128, 128), gst_conv_kernel_size=3, gst_conv_stride=2,
                 gst_gru_layers=1, gst_gru_units=128,
                 init_type="xavier_uniform", init_enc_alpha=1.0, init_dec_alpha=1.0, use_masking=True):
        super().__init__()
        self.idim, self.odim = idim, odim
        self.eos = idim - 1
        self.padding_idx = 0
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_gst = use_gst
        pos_enc_class = ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding

        self.encoder = Encoder(attention_dim=adim, attention_heads=aheads, linear_units=eunits, num_blocks=elayers,
                               input_layer=nn.Embedding(idim, adim, padding_idx=self.padding_idx),
                               dropout_rate=transformer_enc_dropout_rate,
                               positional_dropout_rate=transformer_enc_positional_dropout_rate,
                               attention_dropout_rate=transformer_enc_attn_dropout_rate, pos_enc_class=pos_enc_class,
                               normalize_before=encoder_normalize_before, concat_after=encoder_concat_after,
                               positionwise_conv_kernel_size=positionwise_conv_kernel_size)
        if use_gst:
            self.gst = StyleEncoder(idim=odim, gst_tokens=gst_tokens, gst_token_dim=adim, gst_heads=gst_heads,
                                    conv_layers=gst_conv_layers, conv_chans_list=gst_conv_chans_list,
                                    conv_kernel_size=gst_conv_kernel_size, conv_stride=gst_conv_stride,
                                    gru_layers=gst_gru_layers, gru_units=gst_gru_units)
        self.spks = spks
        if spks is not None and spks > 1:
            self.sid_emb = nn.Embedding(spks, adim)  # basic speaker lookup table

        self.duration_predictor = DurationPredictor(adim, duration_predictor_layers, duration_predictor_chans,
                                                    duration_predictor_kernel_size, duration_predictor_dropout_rate)
        self.pitch_predictor = VariancePredictor(adim, pitch_predictor_layers, pitch_predictor_chans,
                                                 pitch_predictor_kernel_size, dropout_rate=pitch_predictor_dropout)
        self.pitch_embed = nn.Sequential(nn.Conv1d(1, adim, pitch_embed_kernel_size,
                                                   padding=(pitch_embed_kernel_size - 1) // 2),
                                         nn.Dropout(pitch_embed_dropout))
        self.energy_predictor = VariancePredictor(adim, energy_predictor_layers, energy_predictor_chans,
                                                  energy_predictor_kernel_size, dropout_rate=energy_predictor_dropout)
        self.energy_embed = nn.Sequential(nn.Conv1d(1, adim, energy_embed_kernel_size,
                                                    padding=(energy_embed_kernel_size - 1) // 2),
                                          nn.Dropout(energy_embed_dropout))
        self.length_regulator = LengthRegulator()
        self.decoder = Encoder(attention_dim=adim, attention_heads=aheads, linear_units=dunits, num_blocks=dlayers,
                               input_layer=None, dropout_rate=transformer_dec_dropout_rate,
                               positional_dropout_rate=transformer_dec_positional_dropout_rate,
                               attention_dropout_rate=transformer_dec_attn_dropout_rate, pos_enc_class=pos_enc_class,
                               normalize_before=decoder_normalize_before, concat_after=decoder_concat_after,
                               positionwise_conv_kernel_size=positionwise_conv_kernel_size)
        self.feat_out = nn.Linear(adim, odim)
        self.postnet = None if postnet_layers == 0 else Postnet(odim, postnet_layers, postnet_chans, postnet_filts,
                                                                 postnet_dropout_rate, use_batch_norm)
        if init_type != "pytorch":
            initialize(self, init_type)
        if use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
        self.criterion = FastSpeech2Loss(use_masking=use_masking)

    @staticmethod
    def _source_mask(ilens):
        return make_non_pad_mask(ilens).unsqueeze(-2)

    def _add_eos(self, text, text_lengths):
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        xs[torch.arange(xs.size(0), device=xs.device), text_lengths] = self.eos
        return xs, text_lengths + 1

    def _forward(self, xs, ilens, ys=None, olens=None, ds=None, ps=None, es=None, sids=None, style_embs=None,
                 is_inference=False, alpha=1.0):
        hs, _ = self.encoder(xs, self._source_mask(ilens))
        gst_weights = None
        if self.use_gst:
            if style_embs is None:
                style_embs, gst_weights = self.gst(ys, olens)
            hs = hs + style_embs.unsqueeze(1)
        if self.spks is not None and self.spks > 1:
            hs = hs + self.sid_emb(sids.view(-1)).unsqueeze(1)

        d_masks = make_pad_mask(ilens, xs.size(1))
        p_in = hs.detach() if self.stop_gradient_from_pitch_predictor else hs
        e_in = hs.detach() if self.stop_gradient_from_energy_predictor else hs
        p_outs = self.pitch_predictor(p_in, d_masks.unsqueeze(-1))
        e_outs = self.energy_predictor(e_in, d_masks.unsqueeze(-1))
        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)
            p_src, e_src = (p_outs, e_outs) if ps is None else (ps, es)
            hs = hs + self.energy_embed(e_src.transpose(1, 2)).transpose(1, 2) \
                + self.pitch_embed(p_src.transpose(1, 2)).transpose(1, 2)
            hs = self.length_regulator(hs, d_outs if ds is None else ds, alpha)
            h_masks = None
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            hs = hs + self.energy_embed(es.transpose(1, 2)).transpose(1, 2) \
                + self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
            hs = self.length_regulator(hs, ds)
            h_masks = self._source_mask(olens)
        zs, _ = self.decoder(hs, h_masks)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)
        return dict(before_outs=before_outs, after_outs=after_outs, d_outs=d_outs, p_outs=p_outs, e_outs=e_outs,
                    gst_weights=gst_weights)

    def forward(self, text, text_lengths, feats, feats_lengths, durations, pitch, energy, sids=None):
        """text (B, N) w/o eos; durations/pitch/energy (B, N) token-level, normalised pitch/energy.
        Returns (loss, stats, outputs)."""
        xs, ilens = self._add_eos(text, text_lengths)
        ds = F.pad(durations, [0, 1])  # eos: duration 0
        ps = F.pad(pitch, [0, 1]).unsqueeze(-1)
        es = F.pad(energy, [0, 1]).unsqueeze(-1)
        out = self._forward(xs, ilens, feats, feats_lengths, ds, ps, es, sids=sids)
        l1_before, l1_after, d_loss, p_loss, e_loss = self.criterion(
            out["after_outs"] if self.postnet is not None else None, out["before_outs"], out["d_outs"],
            out["p_outs"], out["e_outs"], feats, ds, ps, es, ilens, feats_lengths)
        l1 = l1_before + l1_after
        loss = l1 + d_loss + p_loss + e_loss
        stats = dict(loss=loss.detach(), l1_loss=l1.detach(), l1_before=l1_before.detach(),
                     l1_after=l1_after.detach(), duration_loss=d_loss.detach(), pitch_loss=p_loss.detach(),
                     energy_loss=e_loss.detach())
        return loss, stats, out

    @torch.no_grad()
    def style_from_reference(self, feats, feats_lengths=None):
        """GST(ref mel) -> (style_emb (B, adim), weights (B, heads, tokens))."""
        return self.gst(feats, feats_lengths)

    @torch.no_grad()
    def style_from_weights(self, weights):
        return self.gst.from_weights(weights)

    @torch.no_grad()
    def inference(self, text, sids=None, style_embs=None, ref_feats=None, alpha=1.0, durations=None, pitch=None,
                  energy=None):
        """text: (N,) ids w/o eos. Style from `style_embs` (1, adim) or a reference mel (T, odim).
        Optional teacher forcing with durations/pitch/energy (N,)."""
        xs = F.pad(text, [0, 1], "constant", self.eos).unsqueeze(0)
        ilens = torch.tensor([xs.size(1)], dtype=torch.long, device=xs.device)
        if self.use_gst and style_embs is None:
            style_embs, _ = self.gst(ref_feats.unsqueeze(0))
        if sids is not None:
            sids = torch.as_tensor(sids, device=xs.device).view(1)
        ds = ps = es = None
        if durations is not None:
            ds = F.pad(durations, [0, 1]).unsqueeze(0)
            ps = F.pad(pitch, [0, 1]).view(1, -1, 1)
            es = F.pad(energy, [0, 1]).view(1, -1, 1)
        out = self._forward(xs, ilens, sids=sids, style_embs=style_embs, ds=ds, ps=ps, es=es, is_inference=True,
                            alpha=alpha)
        return dict(feat_gen=out["after_outs"][0], duration=out["d_outs"][0], pitch=out["p_outs"][0],
                    energy=out["e_outs"][0])
