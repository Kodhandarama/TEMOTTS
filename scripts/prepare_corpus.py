"""Build the Stage I corpus: LJSpeech + emotional TTS dataset.

Outputs (in --out_dir):
  metadata.tsv  utt_id, corpus, spk_id, emotion*, split, wav, raw_text, text
  lexicon.txt   word -> stress-marked ARPAbet (g2p_en), given to MFA and reused at inference
and an MFA corpus (in --mfa_dir/corpus/<speaker>/<utt>.{wav,lab}).

*emotion = source folder name. It is never used as a training signal; it is only
used to stratify the 18:1:1 split and to colour plots / group logged samples.
"""
import argparse
import csv
import os
import random
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temotts.text.cleaners import english_cleaners  # noqa: E402
from temotts.text.frontend import Frontend, save_lexicon, words_of  # noqa: E402

SPEAKERS = {"ljspeech": 0, "emotional": 1}


def read_ljspeech(root):
    items = []
    with open(os.path.join(root, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            uid, raw, norm = line.rstrip("\n").split("|")
            items.append(dict(utt_id=uid, corpus="ljspeech", emotion="ljspeech",
                              wav=os.path.join(root, "wavs", uid + ".wav"), raw_text=raw, norm_text=norm))
    return items


def read_emotional(root):
    items = []
    for emo in sorted(os.listdir(root)):
        meta = os.path.join(root, emo, "metadata.csv")
        if not os.path.isfile(meta):
            continue
        n_missing = 0
        with open(meta, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                name, text = line.split("|", 1)
                name, text = name.strip(), text.strip()
                wav = os.path.join(root, emo, "wavs", name)
                if not os.path.isfile(wav):
                    n_missing += 1
                    continue
                items.append(dict(utt_id=f"emo_{emo}_{os.path.splitext(name)[0]}", corpus="emotional", emotion=emo,
                                  wav=wav, raw_text=text, norm_text=text))
        if n_missing:
            print(f"[emotional] {emo}: {n_missing} metadata rows without audio -> skipped")
    return items


def split_18_1_1(items, seed):
    """Stratified (per corpus / emotion folder) train:val:test = 18:1:1."""
    groups = {}
    for it in items:
        groups.setdefault(it["emotion"], []).append(it)
    rng = random.Random(seed)
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda x: x["utt_id"])
        rng.shuffle(g)
        n_eval = max(1, round(len(g) / 20))
        for i, it in enumerate(g):
            it["split"] = "val" if i < n_eval else ("test" if i < 2 * n_eval else "train")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="contains LJSpeech-1.1.tar.bz2 (or LJSpeech-1.1/) and emotions/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mfa_dir", required=True)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    lj_root = os.path.join(args.raw_dir, "LJSpeech-1.1")
    if not os.path.isfile(os.path.join(lj_root, "metadata.csv")):
        print("extracting LJSpeech ...")
        with tarfile.open(os.path.join(args.raw_dir, "LJSpeech-1.1.tar.bz2")) as tar:
            tar.extractall(args.raw_dir)

    items = read_ljspeech(lj_root) + read_emotional(os.path.join(args.raw_dir, "emotions"))
    frontend = Frontend()
    kept = []
    for it in items:
        it["text"] = english_cleaners(it["norm_text"])
        words = words_of(it["text"])
        if not words:
            print("no words, skipped:", it["utt_id"])
            continue
        it["words"] = words
        kept.append(it)
    split_18_1_1(kept, args.seed)

    vocab = sorted({w for it in kept for w in it["words"]})
    print(f"{len(kept)} utterances, {len(vocab)} unique words -> g2p_en lexicon")
    lexicon = {w: frontend.word_to_phones(w) for w in vocab}
    empty = [w for w, ph in lexicon.items() if not ph]
    if empty:
        print("words without pronunciation (utterances dropped):", empty)
        kept = [it for it in kept if not any(w in empty for w in it["words"])]
        lexicon = {w: ph for w, ph in lexicon.items() if ph}

    os.makedirs(args.out_dir, exist_ok=True)
    save_lexicon(lexicon, os.path.join(args.out_dir, "lexicon.txt"))
    with open(os.path.join(args.out_dir, "metadata.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["utt_id", "corpus", "spk_id", "emotion", "split", "wav", "raw_text", "text"])
        for it in kept:
            w.writerow([it["utt_id"], it["corpus"], SPEAKERS[it["corpus"]], it["emotion"], it["split"], it["wav"],
                        it["raw_text"].replace("\t", " "), it["text"]])

    for it in kept:
        d = os.path.join(args.mfa_dir, "corpus", it["corpus"])
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, it["utt_id"] + ".wav")
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(it["wav"]), link)
        with open(os.path.join(d, it["utt_id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(it["words"]) + "\n")

    for split in ("train", "val", "test"):
        for corpus in ("ljspeech", "emotional"):
            n = sum(1 for it in kept if it["split"] == split and it["corpus"] == corpus)
            print(f"{split:5s} {corpus:9s} {n}")


if __name__ == "__main__":
    main()
