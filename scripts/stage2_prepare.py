"""Stage II data (paper Sec. 3.2), for every emotional-dataset utterance:
  * GST weights from the trained Stage I reference encoder (soft-label targets)
  * DistilRoBERTa emotion class probabilities -> emotion data pruning (keep if max prob > P_th)
  * emotion text embedding = mean pooling over the LM's last layer
Saved to --out as a torch dict; pruning is stored as a mask (applied in train_stage2.py).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temotts.data import read_index  # noqa: E402
from temotts.lm import LM_NAME, EmotionTextEncoder  # noqa: E402
from temotts.pipeline import load_stage1  # noqa: E402
from temotts.utils import load_yaml  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--feats_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    p_th = cfg["pruning"]["p_th"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in read_index(args.feats_dir) if r["corpus"] == "emotional"]
    model, stats, _, ck = load_stage1(args.stage1_ckpt, device)
    print(f"stage I checkpoint step {ck['step']}, {len(rows)} emotional utterances")

    gst = []
    with torch.no_grad():
        for r in rows:  # batch size 1: targets independent of batching
            mel = np.load(os.path.join(args.feats_dir, "feats", r["utt_id"] + ".npz"))["mel"]
            mel = torch.from_numpy(stats.norm_mel(mel).astype(np.float32)).to(device)
            _, w = model.style_from_reference(mel.unsqueeze(0))
            gst.append(w[0].float().cpu())
    gst = torch.stack(gst)  # (N, heads, tokens)

    lm = EmotionTextEncoder(device=device)
    texts = [r["raw_text"] for r in rows]
    emb, probs = lm(texts)
    keep = probs.max(1).values > p_th

    data = dict(utt_ids=[r["utt_id"] for r in rows], split=[r["split"] for r in rows],
                emotion=[r["emotion"] for r in rows], raw_text=texts, gst_weights=gst, targets=gst.mean(1),
                text_emb=emb, lm_probs=probs, lm_labels=lm.labels, keep=keep, p_th=p_th, lm_name=LM_NAME,
                stage1_ckpt=args.stage1_ckpt, stage1_step=ck["step"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(data, args.out)

    print(f"P_th = {p_th}: kept {int(keep.sum())} / {len(rows)}")
    for split in ("train", "val", "test"):
        idx = [i for i, s in enumerate(data["split"]) if s == split]
        print(f"  {split}: {int(keep[idx].sum())} / {len(idx)}")
    dom = probs.argmax(1)
    for emo in sorted(set(data["emotion"])):
        idx = [i for i, e in enumerate(data["emotion"]) if e == emo]
        c = Counter(lm.labels[int(dom[i])] for i in idx if keep[i])
        print(f"  folder {emo:11s} kept {int(keep[idx].sum()):4d}/{len(idx):4d}  LM labels of kept: {dict(c.most_common())}")
    print("mean target entropy:", float(-(data["targets"].clamp_min(1e-12).log() * data["targets"]).sum(-1).mean()))


if __name__ == "__main__":
    main()
