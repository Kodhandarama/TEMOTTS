"""TEMOTTS text-only inference (paper Fig. 3).

python synthesize.py --stage1_ckpt exp/stage1/best.pt --stage2_ckpt exp/stage2/best.pt \
    --asset_dir assets --text "I am so sad these days." --out_dir exp/samples
or --text_file sentences.txt (one per line). Optional --ref_wav to take the style
from reference audio instead (Stage I GST path) for comparison.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from temotts.audio import FeatureExtractor, load_wav
from temotts.pipeline import TEMOTTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", default=None)
    p.add_argument("--asset_dir", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--text_file", default=None)
    p.add_argument("--ref_wav", default=None)
    p.add_argument("--speaker", choices=["emotional", "ljspeech"], default="emotional")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()
    texts = [args.text] if args.text else [line.strip() for line in open(args.text_file) if line.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ref_wav is None and args.stage2_ckpt is None:
        raise SystemExit("need --stage2_ckpt for text-only inference (or --ref_wav)")

    tts = TEMOTTS(args.stage1_ckpt, args.stage2_ckpt if args.ref_wav is None else None, args.asset_dir, device)
    sid = {"emotional": 1, "ljspeech": 0}[args.speaker]
    ref_mel = None
    if args.ref_wav:
        mel, _, _ = FeatureExtractor()(load_wav(args.ref_wav, 22050))
        ref_mel = torch.from_numpy(tts.stats.norm_mel(mel).astype(np.float32))

    os.makedirs(args.out_dir, exist_ok=True)
    meta = []
    for k, text in enumerate(texts):
        o = tts.synthesize(text, ref_mel=ref_mel, sid=sid)
        path = os.path.join(args.out_dir, f"{k:03d}.wav")
        sf.write(path, o["wav"], 22050)
        entry = dict(wav=path, text=text, phones=" ".join(o["phones"]), gst_weights=np.round(o["weights"], 4).tolist())
        if o["lm_probs"] is not None:
            entry["lm_probs"] = dict(zip(tts.lm.labels, np.round(o["lm_probs"][0].numpy(), 4).tolist()))
        meta.append(entry)
        print(path, "|", text)
    with open(os.path.join(args.out_dir, "synthesis.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
