"""TEMOTTS Stage I: train FastSpeech2 + GST (+ speaker table) on LJSpeech + emotional data.

python train_stage1.py --config configs/stage1.yaml --feats_dir $DATA_DIR/feats \
    --lexicon $DATA_DIR/corpus/lexicon.txt --exp_dir $EXP_DIR/stage1 --asset_dir $ASSET_DIR
Resumes automatically from <exp_dir>/last.pt. Keeps <exp_dir>/best.pt (lowest validation loss).
"""
import argparse
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from temotts.data import FrameBudgetBatchSampler, Stats, TTSDataset, collate
from temotts.models.fastspeech2 import FastSpeech2
from temotts.optim import build_optimizer
from temotts.text.frontend import load_lexicon
from temotts.text.symbols import SYMBOLS
from temotts.utils import apply_overrides, init_wandb, load_yaml, plot_mels, plot_weights, seed_everything


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def model_inputs(b):
    return dict(text=b["text"], text_lengths=b["text_lengths"], feats=b["feats"], feats_lengths=b["feats_lengths"],
                durations=b["durations"], pitch=b["pitch"], energy=b["energy"], sids=b["sids"])


@torch.no_grad()
def validate(model, loader, device, amp_dtype):
    model.eval()
    sums, n = defaultdict(float), 0
    weights, emotions = [], []
    for b in loader:
        b = to_device(b, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _, stats, out = model(**model_inputs(b))
        bs = b["text"].size(0)
        for k, v in stats.items():
            sums[k] += float(v) * bs
        n += bs
        if out["gst_weights"] is not None:
            weights.append(out["gst_weights"].float().mean(1).cpu())
            emotions += b["emotions"]
    model.train()
    return {k: v / n for k, v in sums.items()}, (torch.cat(weights).numpy() if weights else None), emotions


def entropy(w):
    w = np.clip(w, 1e-12, 1.0)
    return float(-(w * np.log(w)).sum(-1).mean())


@torch.no_grad()
def log_samples(model, ds, sample_rows, stats, vocoder, device, step, wandb, with_gt_audio):
    model.eval()
    logs, mels, titles = {}, [], []
    for r in sample_rows:
        item = ds.load(r)
        out = model.inference(item["text"].to(device), sids=item["spk_id"], ref_feats=item["mel"].to(device))
        pred = out["feat_gen"].float().cpu().numpy()
        mels += [item["mel"].numpy(), pred]
        titles += [f"{r['utt_id']} ({r['emotion']}) GT", f"{r['utt_id']} predicted (GST from GT ref, pred. durations)"]
        if vocoder is not None:
            cap = f"{r['emotion']}: {r['raw_text'][:120]}"
            logs[f"samples/{r['emotion']}/pred"] = wandb.Audio(vocoder(stats.denorm_mel(pred)), sample_rate=22050,
                                                                caption=cap)
            if with_gt_audio:
                logs[f"samples/{r['emotion']}/gt_copysynth"] = wandb.Audio(
                    vocoder(stats.denorm_mel(item["mel"].numpy())), sample_rate=22050, caption=cap)
    fig = plot_mels(mels, titles)
    logs["samples/mels"] = wandb.Image(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)
    wandb.log(logs, step=step)
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1.yaml")
    p.add_argument("--feats_dir", required=True)
    p.add_argument("--lexicon", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--asset_dir", default=None, help="PWG vocoder location (audio samples in W&B)")
    p.add_argument("--run_name", default="stage1")
    p.add_argument("--max_steps", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.val_interval=500")
    p.add_argument("--smoke", action="store_true", help="tiny train/val subsets for a quick end-to-end check")
    args = p.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.set)
    tcfg, dcfg = cfg["train"], cfg["data"]
    if args.max_steps is not None:
        tcfg["max_steps"] = args.max_steps
    os.makedirs(args.exp_dir, exist_ok=True)
    seed_everything(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if tcfg.get("amp") and device == "cuda" else None

    stats = Stats(os.path.join(args.feats_dir, "stats.npz"))
    train_ds = TTSDataset(args.feats_dir, stats, ("train",), dcfg["max_frames"])
    val_ds = TTSDataset(args.feats_dir, stats, ("val",), dcfg["max_frames"])
    if args.smoke:
        train_ds.rows = train_ds.rows[:: max(1, len(train_ds.rows) // 64)]
        val_ds.rows = val_ds.rows[:: max(1, len(val_ds.rows) // 32)]
    sampler = FrameBudgetBatchSampler([r["n_frames"] for r in train_ds.rows], dcfg["batch_max_frames"],
                                      dcfg["max_batch_size"], shuffle=True, seed=tcfg["seed"])
    val_sampler = FrameBudgetBatchSampler([r["n_frames"] for r in val_ds.rows], dcfg["batch_max_frames"],
                                          dcfg["max_batch_size"], shuffle=False)
    nw = dcfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate, num_workers=min(nw, 2))
    print(f"train {len(train_ds)} utts / {len(sampler)} batches per epoch, val {len(val_ds)} utts")

    model = FastSpeech2(idim=len(SYMBOLS), **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FastSpeech2+GST parameters: {n_params / 1e6:.2f}M")
    opt, sched = build_optimizer(model.parameters(), cfg["optim"])

    step, epoch, best_val = 0, 0, math.inf
    last_path, best_path = os.path.join(args.exp_dir, "last.pt"), os.path.join(args.exp_dir, "best.pt")
    if os.path.isfile(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, epoch, best_val = ck["step"], ck["epoch"], ck["best_val"]
        print(f"resumed from step {step}")

    lexicon = load_lexicon(args.lexicon)
    meta = dict(config=cfg, stats=stats.state_dict(), symbols=SYMBOLS, lexicon=lexicon,
                speakers={"ljspeech": 0, "emotional": 1})

    import wandb

    run = init_wandb(args.exp_dir, args.run_name, dict(cfg, n_params=n_params, n_train=len(train_ds),
                                                      n_val=len(val_ds)), group="stage1", job_type="train")
    wandb.define_metric("val/*", summary="min")

    vocoder = None
    if args.asset_dir:
        try:
            from temotts.vocoder import Vocoder

            vocoder = Vocoder(args.asset_dir, device=device)
        except Exception as e:  # samples are a convenience, never block training
            print("vocoder unavailable:", e)
    groups = defaultdict(list)
    for r in sorted(val_ds.rows, key=lambda r: r["utt_id"]):
        groups[r["emotion"]].append(r)
    sample_rows = [r for g in sorted(groups) for r in groups[g][: tcfg["n_samples_per_group"]]]

    model.train()
    acc, t0, n_acc = defaultdict(float), time.time(), 0
    done = step >= tcfg["max_steps"]
    while not done:
        sampler.set_epoch(epoch)
        for b in train_loader:
            b = to_device(b, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, st, out = model(**model_inputs(b))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            if torch.isfinite(grad_norm):
                opt.step()
            else:
                print(f"step {step}: non-finite grad norm, update skipped")
            sched.step()
            step += 1

            for k, v in st.items():
                acc[k] += float(v)
            acc["grad_norm"] += float(grad_norm)
            if out["gst_weights"] is not None:
                acc["gst_entropy"] += entropy(out["gst_weights"].detach().float().mean(1).cpu().numpy())
            n_acc += 1
            if step % tcfg["log_interval"] == 0:
                dt = time.time() - t0
                log = {f"train/{k}": v / n_acc for k, v in acc.items()}
                log.update({"train/lr": sched.get_last_lr()[0], "train/steps_per_sec": n_acc / dt, "epoch": epoch,
                            "train/batch_size": b["text"].size(0)})
                wandb.log(log, step=step)
                print(f"step {step} ep {epoch} loss {log['train/loss']:.4f} l1 {log['train/l1_loss']:.4f} "
                      f"dur {log['train/duration_loss']:.4f} pitch {log['train/pitch_loss']:.4f} "
                      f"energy {log['train/energy_loss']:.4f} lr {log['train/lr']:.2e} {n_acc / dt:.2f} it/s",
                      flush=True)
                acc, t0, n_acc = defaultdict(float), time.time(), 0

            if step % tcfg["val_interval"] == 0:
                vstats, vw, vemo = validate(model, val_loader, device, amp_dtype)
                log = {f"val/{k}": v for k, v in vstats.items()}
                if vw is not None:
                    log["val/gst_entropy"] = entropy(vw)
                    emos = sorted(set(vemo))
                    means = np.stack([vw[[i for i, e in enumerate(vemo) if e == em]].mean(0) for em in emos])
                    fig = plot_weights(means, emos, f"mean GST weights per source folder (val, step {step})")
                    log["val/gst_weights_by_folder"] = wandb.Image(fig)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                is_best = vstats["loss"] < best_val
                if is_best:
                    best_val = vstats["loss"]
                    save_ckpt(best_path, model=model.state_dict(), step=step, val=vstats, **meta)
                log["val/best_loss"] = best_val
                wandb.log(log, step=step)
                print(f"[val] step {step} loss {vstats['loss']:.4f} (best {best_val:.4f}){' *' if is_best else ''}",
                      flush=True)

            if step % tcfg["sample_interval"] == 0 or step == tcfg["val_interval"]:
                try:
                    log_samples(model, val_ds, sample_rows, stats, vocoder, device, step, wandb,
                                with_gt_audio=(step == tcfg["val_interval"]))
                except Exception as e:
                    print("sample logging failed:", repr(e))

            if step % tcfg["save_interval"] == 0 or step >= tcfg["max_steps"]:
                state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                             step=step, epoch=epoch, best_val=best_val, **meta)
                save_ckpt(last_path, **state)
                if step % tcfg["keep_every"] == 0:
                    save_ckpt(os.path.join(args.exp_dir, f"step{step}.pt"), **state)
            if step >= tcfg["max_steps"]:
                done = True
                break
        epoch += 1

    print(f"finished at step {step}; best val loss {best_val:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
