"""TEMOTTS Stage II: train the emotional text adaptation module on the P_th-pruned
emotional data (text embedding -> GST weights, soft-label cross-entropy), then
run text-only synthesis for W&B.

python train_stage2.py --config configs/stage2.yaml --data $EXP_DIR/stage2/stage2_data.pt \
    --stage1_ckpt $EXP_DIR/stage1/best.pt --exp_dir $EXP_DIR/stage2 --asset_dir $ASSET_DIR
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from temotts.models.adaptation import AdaptationModule, soft_cross_entropy
from temotts.optim import build_optimizer
from temotts.utils import apply_overrides, init_wandb, load_yaml, plot_pitch, plot_weights, seed_everything

FIG2_FOLDERS = {"angry": "Anger", "happy": "Happy", "base": "Neutral", "sad": "Sad"}


def tsne_figure(weights, embs, folders, title_suffix=""):
    """Paper Fig. 2: t-SNE of the emotional style space (GST weights) next to the
    emotion text embedding space, coloured by source folder (labels only for display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    idx = [i for i, f in enumerate(folders) if f in FIG2_FOLDERS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X, title in zip(axes, (weights, embs), ("Emotional style space (GST weights)", "Emotion text embedding space")):
        X = np.asarray(X)[idx]
        Y = TSNE(n_components=2, perplexity=min(30, max(5, len(idx) // 4)), init="pca", random_state=0).fit_transform(X)
        for f, name in FIG2_FOLDERS.items():
            m = [k for k, i in enumerate(idx) if folders[i] == f]
            ax.scatter(Y[m, 0], Y[m, 1], s=8, label=name)
        ax.set_title(title + title_suffix, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend()
    fig.tight_layout()
    return fig


@torch.no_grad()
def evaluate(model, x, t):
    model.eval()
    logits = model(x)
    ce = float(soft_cross_entropy(logits, t))
    h = float(-(t * t.clamp_min(1e-12).log()).sum(-1).mean())
    top1 = float((logits.argmax(-1) == t.argmax(-1)).float().mean())
    l1 = float((logits.softmax(-1) - t).abs().sum(-1).mean())
    model.train()
    return dict(soft_ce=ce, kl=ce - h, top1_agreement=top1, weights_l1=l1)


def f0_hz(wav, sr=22050):
    import pyworld

    x = wav.astype(np.float64)
    f0, t = pyworld.dio(x, sr, f0_floor=80, f0_ceil=400, frame_period=1000 * 256 / sr)
    f0 = pyworld.stonemask(x, f0, t, sr)
    return np.where(f0 > 0, f0, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--feats_dir", default=None, help="for GT reference samples")
    p.add_argument("--asset_dir", default=None)
    p.add_argument("--run_name", default="stage2")
    p.add_argument("--wandb_group", default="stage2")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.max_steps=200")
    args = p.parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.set)
    tcfg = cfg["train"]
    os.makedirs(args.exp_dir, exist_ok=True)
    seed_everything(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(args.data, weights_only=False)
    # emotion data pruning: keep if the LM's dominant-emotion probability > P_th (recomputed from the
    # config; identical to stage2_prepare's mask for the same P_th)
    p_th = float(cfg["pruning"]["p_th"])
    keep, split = data["lm_probs"].max(1).values > p_th, np.array(data["split"])
    tr = np.where(keep.numpy() & (split == "train"))[0]
    va = np.where(keep.numpy() & (split == "val"))[0]
    X, T = data["text_emb"].float(), data["targets"].float()
    x_tr, t_tr, x_va, t_va = X[tr].to(device), T[tr].to(device), X[va].to(device), T[va].to(device)
    n_tokens = T.size(1)
    print(f"stage II: {len(tr)} train / {len(va)} val pruned utterances (P_th {p_th}), {n_tokens} tokens")

    margs = dict(in_dim=cfg["model"]["in_dim"], hidden=list(cfg["model"]["hidden"]), n_tokens=n_tokens)
    model = AdaptationModule(margs["in_dim"], tuple(margs["hidden"]), n_tokens).to(device)
    opt, sched = build_optimizer(model.parameters(), cfg["optim"])

    import wandb

    run = init_wandb(args.exp_dir, args.run_name, dict(cfg, n_train=len(tr), n_val=len(va), p_th=p_th,
                                                      stage1_ckpt=data["stage1_ckpt"], stage1_step=data["stage1_step"]),
                     group=args.wandb_group, job_type="train")
    folders = data["emotion"]
    kept_table = wandb.Table(columns=["folder", "total", "kept_train", "kept_val", "kept_test"])
    for f in sorted(set(folders)):
        idx = [i for i, e in enumerate(folders) if e == f]
        kept_table.add_data(f, len(idx), *[int(sum(1 for i in idx if keep[i] and split[i] == s)) for s in ("train", "val", "test")])
    pruned = np.where(keep.numpy() & (split != "test"))[0]
    fig = tsne_figure(T[pruned].numpy(), X[pruned].numpy(), [folders[i] for i in pruned], " (pruned train+val)")
    wandb.log({"data/pruning": kept_table, "data/tsne_fig2": wandb.Image(fig)}, step=0)
    import matplotlib.pyplot as plt

    plt.close(fig)

    best, step, t0 = math.inf, 0, time.time()
    best_path = os.path.join(args.exp_dir, "best.pt")
    bs = tcfg["batch_size"]
    g = torch.Generator().manual_seed(tcfg["seed"])
    model.train()
    while step < tcfg["max_steps"]:
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs].to(device)
            loss = soft_cross_entropy(model(x_tr[b]), t_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if step % tcfg["log_interval"] == 0:
                wandb.log({"train/soft_ce": float(loss), "train/grad_norm": float(gn), "train/lr": sched.get_last_lr()[0]},
                          step=step)
            if step % tcfg["val_interval"] == 0:
                v = evaluate(model, x_va, t_va)
                log = {f"val/{k}": val for k, val in v.items()}
                if v["soft_ce"] < best:
                    best = v["soft_ce"]
                    torch.save(dict(model=model.state_dict(), model_args=margs, step=step, val=v, p_th=p_th,
                                    seed=tcfg["seed"], n_train=len(tr), lm_name=data["lm_name"],
                                    stage1_ckpt=data["stage1_ckpt"]), best_path)
                log["val/best_soft_ce"] = best
                log["train_eval/soft_ce"] = evaluate(model, x_tr, t_tr)["soft_ce"]
                wandb.log(log, step=step)
                print(f"step {step} train {float(loss):.4f} val {v['soft_ce']:.4f} kl {v['kl']:.4f} "
                      f"top1 {v['top1_agreement']:.3f} best {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step >= tcfg["max_steps"]:
                break

    # ---------------- final analysis + text-only synthesis with the best module ----------------
    ck = torch.load(best_path, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pred = model.predict_weights(X.to(device)).cpu()
    test = np.where(split == "test")[0]
    vt = evaluate(model, X[test].to(device), T[test].to(device))
    summary = {f"test_all/{k}": v for k, v in vt.items()}
    tk = np.where(keep.numpy() & (split == "test"))[0]
    if len(tk):
        summary.update({f"test_pruned/{k}": v for k, v in evaluate(model, X[tk].to(device), T[tk].to(device)).items()})
    fig = tsne_figure(pred[test].numpy(), X[test].numpy(), [folders[i] for i in test], " (test, predicted)")
    summary["final/tsne_test_predicted"] = wandb.Image(fig)
    plt.close(fig)
    emos = sorted(set(folders))
    fig = plot_weights(np.stack([pred[[i for i in test if folders[i] == e]].mean(0).numpy() for e in emos]), emos,
                       "mean predicted GST weights per source folder (test, text only)")
    summary["final/pred_weights_by_folder"] = wandb.Image(fig)
    plt.close(fig)
    wandb.log(summary, step=step)
    print("best val soft-CE", ck["val"], "\n", {k: v for k, v in summary.items() if not k.startswith("final")})

    if args.asset_dir:
        from temotts.lm import EmotionTextEncoder
        from temotts.pipeline import TEMOTTS
        import soundfile as sf

        tts = TEMOTTS(args.stage1_ckpt, best_path, asset_dir=args.asset_dir, device=device,
                      lm=EmotionTextEncoder(device=device))
        out_dir = os.path.join(args.exp_dir, "samples")
        os.makedirs(out_dir, exist_ok=True)
        logs, ws, names = {}, [], []
        sentences = [s.strip() for s in open(cfg["demo_sentences"]) if s.strip()]
        for k, s in enumerate(sentences):
            o = tts.synthesize(s)
            labels = tts.lm.labels
            top = labels[int(o["lm_probs"][0].argmax())]
            sf.write(os.path.join(out_dir, f"demo_{k:02d}.wav"), o["wav"], 22050)
            logs[f"demo/{k:02d}"] = wandb.Audio(o["wav"], sample_rate=22050, caption=f"[LM: {top}] {s}")
            ws.append(o["weights"])
            names.append(f"{k:02d} [{top}] {s[:40]}")
        fig = plot_weights(np.stack(ws), names, "predicted GST weights (demo sentences, text only)")
        logs["demo/weights"] = wandb.Image(fig)
        plt.close(fig)

        # emotional test sentences: TEMOTTS (text only) vs. GT copy-synthesis, with F0 contours (cf. Fig. 5)
        if args.feats_dir:
            for f in ("angry", "happy", "sad", "base"):
                idx = [i for i in test if folders[i] == f][:2]
                for i in idx:
                    o = tts.synthesize(data["raw_text"][i])
                    mel = np.load(os.path.join(args.feats_dir, "feats", data["utt_ids"][i] + ".npz"))["mel"]
                    gt = tts.vocoder(mel)
                    cap = data["raw_text"][i][:120]
                    logs[f"test_text_only/{f}/{data['utt_ids'][i]}"] = wandb.Audio(o["wav"], sample_rate=22050, caption=cap)
                    logs[f"test_gt/{f}/{data['utt_ids'][i]}"] = wandb.Audio(gt, sample_rate=22050, caption=cap)
                    fig = plot_pitch([f0_hz(gt), f0_hz(o["wav"])], ["Ground truth (PWG copy-synthesis)", "TEMOTTS"],
                                     f"{f}: {cap[:60]}", 256 / 22050)
                    logs[f"test_f0/{f}/{data['utt_ids'][i]}"] = wandb.Image(fig)
                    plt.close(fig)
        wandb.log(logs, step=step)
    run.finish()


if __name__ == "__main__":
    main()
