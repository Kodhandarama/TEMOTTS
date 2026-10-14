"""MFA TextGrids + wavs -> per-utterance features for FastSpeech2.

For each aligned utterance writes <out_dir>/feats/<utt>.npz with
  mel [T, 80] (raw log10-mel), phones [N] (ids), durations [N] (frames, sum == T),
  pitch [N], energy [N] (token-averaged raw log-F0 / energy)
plus <out_dir>/index.tsv and train-split normalisation stats <out_dir>/stats.npz.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temotts.audio import FeatureExtractor, average_by_duration, load_wav  # noqa: E402
from temotts.text.symbols import SPN, UNK, phones_to_ids  # noqa: E402
from temotts.textgrid import phones_and_durations  # noqa: E402

FEAT = None
CFG = None


def _init(cfg):
    global FEAT, CFG
    CFG = cfg
    FEAT = FeatureExtractor(sample_rate=cfg["sample_rate"], hop_length=cfg["hop_length"])


def process(row):
    utt = row["utt_id"]
    tg = os.path.join(CFG["textgrid_dir"], row["corpus"], utt + ".TextGrid")
    if not os.path.isfile(tg):
        return utt, "no_alignment", None
    try:
        phones, durs, start, end = phones_and_durations(tg, CFG["sample_rate"], CFG["hop_length"])
    except ValueError as e:
        return utt, f"bad_textgrid:{e}", None
    if SPN in phones or UNK in phones:
        return utt, "oov_phone", None
    x = load_wav(row["wav"], CFG["sample_rate"])
    mel, energy, f0 = FEAT(x)
    if end > len(mel):  # rounding at the very end of the file
        durs[-1] -= end - len(mel)
        end = len(mel)
        if durs[-1] < 0:
            return utt, "bad_durations", None
    mel, energy, f0 = mel[start:end], energy[start:end], f0[start:end]
    durs = np.asarray(durs, dtype=np.int64)
    assert durs.sum() == len(mel), (utt, durs.sum(), len(mel))
    pitch = average_by_duration(f0, durs)
    tok_energy = average_by_duration(energy, durs)
    np.savez(os.path.join(CFG["out_dir"], "feats", utt + ".npz"), mel=mel,
             phones=np.asarray(phones_to_ids(phones), dtype=np.int64), durations=durs,
             pitch=pitch, energy=tok_energy)
    voiced = durs > 0
    acc = dict(mel_sum=mel.astype(np.float64).sum(0), mel_sq=(mel.astype(np.float64) ** 2).sum(0), n_frames=len(mel),
               pitch=pitch[voiced & (pitch > 0)].astype(np.float64), energy=tok_energy[voiced].astype(np.float64))
    return utt, "ok", dict(n_frames=len(mel), n_phones=len(phones), acc=acc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", required=True)
    p.add_argument("--textgrid_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--nj", type=int, default=8)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out_dir, "feats"), exist_ok=True)

    with open(os.path.join(args.corpus_dir, "metadata.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
    cfg = dict(textgrid_dir=args.textgrid_dir, out_dir=args.out_dir, sample_rate=args.sample_rate,
               hop_length=args.hop_length)
    by_id = {r["utt_id"]: r for r in rows}

    results, failures = {}, {}
    with Pool(args.nj, initializer=_init, initargs=(cfg,)) as pool:
        for utt, status, info in tqdm(pool.imap_unordered(process, rows, chunksize=8), total=len(rows), mininterval=30):
            if status == "ok":
                results[utt] = info
            else:
                failures[utt] = status

    # train-split statistics (global MVN, as in ESPnet)
    mel_sum = np.zeros(80)
    mel_sq = np.zeros(80)
    n = 0
    pitch, energy = [], []
    for utt, info in results.items():
        if by_id[utt]["split"] != "train":
            continue
        a = info["acc"]
        mel_sum += a["mel_sum"]
        mel_sq += a["mel_sq"]
        n += a["n_frames"]
        pitch.append(a["pitch"])
        energy.append(a["energy"])
    mel_mean = mel_sum / n
    mel_std = np.sqrt(np.maximum(mel_sq / n - mel_mean ** 2, 1e-12))
    pitch, energy = np.concatenate(pitch), np.concatenate(energy)
    np.savez(os.path.join(args.out_dir, "stats.npz"), mel_mean=mel_mean.astype(np.float32),
             mel_std=mel_std.astype(np.float32), pitch_mean=np.float32(pitch.mean()), pitch_std=np.float32(pitch.std()),
             energy_mean=np.float32(energy.mean()), energy_std=np.float32(energy.std()))

    with open(os.path.join(args.out_dir, "index.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["utt_id", "corpus", "spk_id", "emotion", "split", "n_frames", "n_phones", "raw_text", "text"])
        for r in rows:
            if r["utt_id"] in results:
                info = results[r["utt_id"]]
                w.writerow([r["utt_id"], r["corpus"], r["spk_id"], r["emotion"], r["split"], info["n_frames"],
                            info["n_phones"], r["raw_text"], r["text"]])
    with open(os.path.join(args.out_dir, "failures.tsv"), "w") as f:
        for utt, status in sorted(failures.items()):
            f.write(f"{utt}\t{status}\n")

    print(f"ok: {len(results)}  failed: {len(failures)}")
    from collections import Counter

    print("failure reasons:", Counter(s.split(":")[0] for s in failures.values()))
    print(f"pitch(log f0) mean {pitch.mean():.3f} std {pitch.std():.3f}; energy mean {energy.mean():.3f} std {energy.std():.3f}")


if __name__ == "__main__":
    main()
