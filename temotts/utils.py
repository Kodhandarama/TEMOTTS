import os
import random
import uuid

import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, items):
    """--set a.b=value (value parsed as YAML)."""
    for item in items or []:
        key, val = item.split("=", 1)
        d = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            d = d[k]
        d[leaf] = yaml.safe_load(val)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(exp_dir, name, config, group, job_type):
    """W&B run that survives SLURM requeues/resubmits (run id stored in exp_dir)."""
    import wandb

    id_file = os.path.join(exp_dir, "wandb_run_id")
    if os.path.isfile(id_file):
        run_id = open(id_file).read().strip()
    else:
        run_id = uuid.uuid4().hex[:12]
        with open(id_file, "w") as f:
            f.write(run_id)
    return wandb.init(project=os.environ.get("WANDB_PROJECT", "temotts"), entity=os.environ.get("WANDB_ENTITY"),
                      name=name, id=run_id, resume="allow", config=config, group=group, job_type=job_type,
                      dir=os.environ.get("WANDB_DIR", exp_dir))


def plot_mels(mels, titles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(mels), 1, figsize=(10, 2.6 * len(mels)), squeeze=False)
    for ax, mel, title in zip(axes[:, 0], mels, titles):
        im = ax.imshow(np.asarray(mel).T, aspect="auto", origin="lower", interpolation="none")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_weights(weights, ylabels, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = np.asarray(weights)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(ylabels) + 1.5))
    im = ax.imshow(weights, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-6, weights.max()))
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(range(weights.shape[1]))
    ax.set_xlabel("GST token")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_pitch(curves, labels, title, hop_s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    for c, l in zip(curves, labels):
        ax.plot(np.arange(len(c)) * hop_s, c, label=l)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(title, fontsize=9)
    ax.legend()
    fig.tight_layout()
    return fig
