"""Pretrained Parallel WaveGAN (LJSpeech, 22.05 kHz) vocoder, as used in the paper."""
import glob
import os

import numpy as np
import torch

PWG_TAG = "ljspeech_parallel_wavegan.v1"


def download_pwg(asset_dir):
    """Download the pretrained PWG once; returns the checkpoint path."""
    from parallel_wavegan.utils import download_pretrained_model

    out_dir = os.path.join(asset_dir, "vocoder")
    os.makedirs(out_dir, exist_ok=True)
    found = find_pwg(asset_dir)
    if found is not None:
        return found
    return download_pretrained_model(PWG_TAG, download_dir=out_dir)


def find_pwg(asset_dir):
    ckpts = sorted(glob.glob(os.path.join(asset_dir, "vocoder", "**", "checkpoint-*.pkl"), recursive=True))
    return ckpts[-1] if ckpts else None


class Vocoder:
    """Wraps PWG. Input: log10-mel (T, 80) *before* any TTS normalisation."""

    def __init__(self, asset_dir, device="cpu"):
        from parallel_wavegan.utils import load_model

        ckpt = find_pwg(asset_dir)
        if ckpt is None:
            raise FileNotFoundError(f"PWG checkpoint not found under {asset_dir}/vocoder "
                                    "(run python -m temotts.download_assets --asset_dir ...)")
        self.model = load_model(ckpt)  # registers stats.h5 found next to the checkpoint
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(device)
        self.device = device
        self.sample_rate = 22050

    @torch.no_grad()
    def __call__(self, logmel):
        if isinstance(logmel, np.ndarray):
            logmel = torch.from_numpy(logmel)
        wav = self.model.inference(logmel.float().to(self.device), normalize_before=True)
        return wav.view(-1).cpu().numpy()
