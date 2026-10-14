"""Dataset, length-bucketed dynamic batching and collation for Stage I."""
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def read_index(feats_dir):
    with open(os.path.join(feats_dir, "index.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
    for r in rows:
        r["n_frames"], r["n_phones"], r["spk_id"] = int(r["n_frames"]), int(r["n_phones"]), int(r["spk_id"])
    return rows


class Stats:
    def __init__(self, path):
        s = np.load(path)
        self.mel_mean, self.mel_std = s["mel_mean"], s["mel_std"]
        self.pitch_mean, self.pitch_std = float(s["pitch_mean"]), float(s["pitch_std"])
        self.energy_mean, self.energy_std = float(s["energy_mean"]), float(s["energy_std"])

    def state_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, d):
        obj = cls.__new__(cls)
        for k, v in d.items():
            setattr(obj, k, np.asarray(v, dtype=np.float32) if isinstance(v, list) else v)
        return obj

    def norm_mel(self, mel):
        return (mel - self.mel_mean) / self.mel_std

    def denorm_mel(self, mel):
        return mel * self.mel_std + self.mel_mean

    def norm_token(self, x, durations, mean, std):
        # zero-duration tokens carry no frames; give them the mean (0 after normalisation)
        return np.where(durations > 0, (x - mean) / std, 0.0).astype(np.float32)


class TTSDataset(Dataset):
    def __init__(self, feats_dir, stats, splits=("train",), max_frames=None, corpora=None, rows=None):
        rows = read_index(feats_dir) if rows is None else rows
        self.rows = [r for r in rows if r["split"] in splits
                     and (max_frames is None or r["n_frames"] <= max_frames)
                     and (corpora is None or r["corpus"] in corpora)]
        self.feats_dir, self.stats = feats_dir, stats

    def __len__(self):
        return len(self.rows)

    def load(self, r):
        d = np.load(os.path.join(self.feats_dir, "feats", r["utt_id"] + ".npz"))
        s = self.stats
        return dict(
            utt_id=r["utt_id"], spk_id=r["spk_id"], emotion=r["emotion"],
            text=torch.from_numpy(d["phones"]),
            durations=torch.from_numpy(d["durations"]),
            pitch=torch.from_numpy(s.norm_token(d["pitch"], d["durations"], s.pitch_mean, s.pitch_std)),
            energy=torch.from_numpy(s.norm_token(d["energy"], d["durations"], s.energy_mean, s.energy_std)),
            mel=torch.from_numpy(s.norm_mel(d["mel"]).astype(np.float32)),
        )

    def __getitem__(self, i):
        return self.load(self.rows[i])


def _pad(seqs, value=0):
    out = seqs[0].new_full((len(seqs), max(len(s) for s in seqs), *seqs[0].shape[1:]), value)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate(batch):
    return dict(
        utt_ids=[b["utt_id"] for b in batch],
        emotions=[b["emotion"] for b in batch],
        text=_pad([b["text"] for b in batch]),
        text_lengths=torch.tensor([len(b["text"]) for b in batch]),
        durations=_pad([b["durations"] for b in batch]),
        pitch=_pad([b["pitch"] for b in batch]),
        energy=_pad([b["energy"] for b in batch]),
        feats=_pad([b["mel"] for b in batch]),
        feats_lengths=torch.tensor([len(b["mel"]) for b in batch]),
        sids=torch.tensor([b["spk_id"] for b in batch]),
    )


class FrameBudgetBatchSampler(Sampler):
    """Sort by length, cut into batches with (batch size x longest utt) <= max_frames,
    shuffle batch order every epoch (ESPnet 'length'/'numel'-style bucketing)."""

    def __init__(self, lengths, max_frames, max_batch_size=64, shuffle=True, seed=0):
        order = np.argsort(lengths, kind="stable")
        self.batches, cur, cur_max = [], [], 0
        for idx in order:
            new_max = max(cur_max, lengths[idx])
            if cur and (new_max * (len(cur) + 1) > max_frames or len(cur) >= max_batch_size):
                self.batches.append(cur)
                cur, new_max = [], lengths[idx]
            cur.append(int(idx))
            cur_max = new_max
        if cur:
            self.batches.append(cur)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.batches)
