"""Fast CPU checks of the model / GST / TextGrid / feature code (no data prep needed).
python scripts/smoke_test.py [--wav some.wav]"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temotts.models.adaptation import AdaptationModule, soft_cross_entropy  # noqa: E402
from temotts.models.fastspeech2 import FastSpeech2  # noqa: E402
from temotts.optim import build_optimizer  # noqa: E402
from temotts.text.symbols import SYMBOLS  # noqa: E402
from temotts.textgrid import phones_and_durations  # noqa: E402
from temotts.utils import load_yaml  # noqa: E402

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "hi there"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 7
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "HH"
        intervals [3]:
            xmin = 0.2
            xmax = 0.35
            text = "AY1"
        intervals [4]:
            xmin = 0.35
            xmax = 0.5
            text = ""
        intervals [5]:
            xmin = 0.5
            xmax = 0.6
            text = "DH"
        intervals [6]:
            xmin = 0.6
            xmax = 0.8
            text = "EH1"
        intervals [7]:
            xmin = 0.8
            xmax = 1.0
            text = "sil"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_yaml(os.path.join(root, "configs/stage1.yaml"))

    # --- TextGrid
    with tempfile.NamedTemporaryFile("w", suffix=".TextGrid", delete=False) as f:
        f.write(TG)
    phones, durs, s, e = phones_and_durations(f.name, 22050, 256)
    assert phones == ["HH", "AY1", "sp", "DH", "EH1"], phones
    assert sum(durs) == e - s and s == 9, (durs, s, e)
    print("textgrid ok:", phones, durs, s, e)

    # --- model forward/backward
    model = FastSpeech2(idim=len(SYMBOLS), **cfg["model"])
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    B, N = 3, [7, 5, 9]
    text = torch.zeros(B, max(N), dtype=torch.long)
    durs = torch.zeros(B, max(N), dtype=torch.long)
    for i, n in enumerate(N):
        text[i, :n] = torch.randint(4, len(SYMBOLS) - 1, (n,))
        durs[i, :n] = torch.randint(1, 6, (n,))
    olens = durs.sum(1)
    feats = torch.randn(B, int(olens.max()), 80)
    for i in range(B):
        feats[i, olens[i]:] = 0
    pitch, energy = torch.randn(B, max(N)), torch.randn(B, max(N))
    for i, n in enumerate(N):
        pitch[i, n:] = 0
        energy[i, n:] = 0
    sids = torch.tensor([0, 1, 1])
    opt, sched = build_optimizer(model.parameters(), cfg["optim"])
    loss, stats, out = model(text, torch.tensor(N), feats, olens, durs, pitch, energy, sids)
    loss.backward()
    opt.step()
    sched.step()
    assert torch.isfinite(loss)
    assert out["gst_weights"].shape == (B, cfg["model"]["gst_heads"], cfg["model"]["gst_tokens"])
    print("forward ok:", {k: round(float(v), 4) for k, v in stats.items()}, "lr", sched.get_last_lr()[0])

    # --- GST: weights -> style embedding equals the attention output
    model.eval()
    with torch.no_grad():
        emb, w = model.style_from_reference(feats, olens)
        emb2 = model.style_from_weights(w if w.size(1) > 1 else w[:, 0])
        assert torch.allclose(emb, emb2, atol=1e-5), (emb - emb2).abs().max()
        emb_b1, w_b1 = model.style_from_reference(feats[1:2, : olens[1]])
        assert torch.allclose(w[1], w_b1[0], atol=1e-4), "GST weights depend on batch padding"
    print("gst from_weights ok; packed GRU batch-independent")

    # --- inference
    with torch.no_grad():
        o = model.inference(text[0, : N[0]], sids=1, style_embs=emb[:1])
        o2 = model.inference(text[0, : N[0]], sids=1, ref_feats=feats[0, : olens[0]])
    print("inference ok:", tuple(o["feat_gen"].shape), tuple(o2["feat_gen"].shape))

    # --- noam schedule peak
    lrs = []
    opt2, sch2 = build_optimizer([torch.nn.Parameter(torch.zeros(1))], cfg["optim"])
    for _ in range(8000):
        lrs.append(sch2.get_last_lr()[0])
        opt2.step()
        sch2.step()
    print(f"noam: lr@1 {lrs[0]:.2e} peak {max(lrs):.2e} at step {int(np.argmax(lrs)) + 1}, lr@8000 {lrs[-1]:.2e}")

    # --- adaptation module
    s2 = load_yaml(os.path.join(root, "configs/stage2.yaml"))
    am = AdaptationModule(s2["model"]["in_dim"], tuple(s2["model"]["hidden"]), 16)
    print(am)
    x, t = torch.randn(4, 768), torch.softmax(torch.randn(4, 16), -1)
    l = soft_cross_entropy(am(x), t)
    l.backward()
    print("adaptation ok:", float(l))

    # --- features
    if args.wav:
        from temotts.audio import FeatureExtractor, load_wav, average_by_duration

        x = load_wav(args.wav, 22050)
        mel, en, f0 = FeatureExtractor()(x)
        print("features:", mel.shape, en.shape, f0.shape, "mel range", mel.min(), mel.max(),
              "f0 Hz range", np.exp(f0[f0 > 0]).min(), np.exp(f0[f0 > 0]).max())
        print("avg", average_by_duration(f0, np.array([10, 0, len(f0) - 10])))
    print("ALL OK")


if __name__ == "__main__":
    main()
