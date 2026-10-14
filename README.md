# TEMOTTS

PyTorch implementation of **TEMOTTS**, from *Exploring speech style spaces with language
models: Emotional TTS without emotion labels* (Shreeram Suresh Chandra, Zongyang Du,
Berrak Sisman; [arXiv:2405.11413](https://arxiv.org/abs/2405.11413)).
Audio samples: [demo page](https://kodhandarama.github.io/TEMOTTSdemo/).

TEMOTTS learns emotional speaking styles without emotion labels, and needs only the text
at inference:

- **Stage I** trains FastSpeech2 with Global Style Tokens (GST) on LJSpeech plus an
  emotional speech dataset. The GST reference encoder learns a style space from audio alone.
- **Stage II** trains a small MLP that maps a sentence's embedding from an emotion language
  model (DistilRoBERTa) to the GST weights of its recording.
- **Inference:** text → language model → MLP → GST weights → FastSpeech2 → Parallel WaveGAN.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for how each part of the paper maps to the code
and for the details the paper leaves open.

## Setup

Linux with a CUDA GPU, Python 3.10.

```bash
conda create -n temotts python=3.10 -y && conda activate temotts
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "setuptools<70" wheel
pip install --no-deps --no-build-isolation parallel_wavegan==0.6.1
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']]"
python -m temotts.download_assets --asset_dir assets   # PWG vocoder + emotion LM

# Montreal Forced Aligner, in its own environment
conda create -n mfa -c conda-forge "montreal-forced-aligner>=3.1" python=3.11 -y
conda run -n mfa mfa model download acoustic english_us_arpa
```

`python scripts/smoke_test.py` runs quick CPU checks of the model code without any data.

## Data

```
data/raw/
  LJSpeech-1.1.tar.bz2              # or the extracted LJSpeech-1.1/
  emotions/<emotion>/metadata.csv   # one "<file>.wav | <text>" per line
  emotions/<emotion>/wavs/*.wav     # 22.05 kHz, one speaker
```

The paper uses the skit-ai emotional TTS dataset (one female speaker, one folder per
emotion). Any single-speaker dataset in this layout works. Folder names are never used as
training labels; they only stratify the 18:1:1 train/val/test split and group the plots.

## Training and synthesis

Run from the repository root:

```bash
# 1. corpus, split and lexicon -> forced alignment -> mel / pitch / energy / durations
python scripts/prepare_corpus.py --raw_dir data/raw --out_dir data/corpus --mfa_dir data/mfa
conda run -n mfa mfa align data/mfa/corpus data/corpus/lexicon.txt english_us_arpa data/mfa/aligned \
    --config_path configs/mfa_align.yaml --clean --overwrite --num_jobs 8 --output_format long_textgrid
python scripts/extract_features.py --corpus_dir data/corpus --textgrid_dir data/mfa/aligned \
    --out_dir data/feats --nj 8

# 2. Stage I: FastSpeech2 + GST (resumes from exp/stage1/last.pt)
python train_stage1.py --config configs/stage1.yaml --feats_dir data/feats \
    --lexicon data/corpus/lexicon.txt --exp_dir exp/stage1 --asset_dir assets

# 3. Stage II: GST-weight targets + LM embeddings, then the adaptation MLP
python scripts/stage2_prepare.py --config configs/stage2.yaml --stage1_ckpt exp/stage1/best.pt \
    --feats_dir data/feats --out exp/stage2/stage2_data.pt
python train_stage2.py --config configs/stage2.yaml --data exp/stage2/stage2_data.pt \
    --stage1_ckpt exp/stage1/best.pt --exp_dir exp/stage2 --feats_dir data/feats --asset_dir assets

# 4. text-only synthesis
python synthesize.py --stage1_ckpt exp/stage1/best.pt --stage2_ckpt exp/stage2/best.pt \
    --asset_dir assets --text "I am so sad these days." --out_dir outputs
```

- **Training time:** the Stage I default is 300k steps, about 8.5 h on one L40S (48 GB).
  `best.pt` is the checkpoint with the lowest validation loss (in our run, at 64k steps).
  Stage II takes a few minutes.
- **Out of memory:** lower `data.batch_max_frames`.
- **Config overrides:** any config value can be changed from the command line, e.g.
  `--set train.max_steps=100000`.
- **Other synthesis modes:** `synthesize.py` also takes `--text_file` (one sentence per
  line), `--ref_wav` (copy the style of a recording instead of predicting it from text) and
  `--speaker ljspeech`.
- **Logging:** training logs to Weights & Biases (project `temotts`). Set `WANDB_ENTITY` to
  choose the account, or `WANDB_MODE=offline` to log locally.

## Code layout

```
temotts/models/   fastspeech2.py (FastSpeech2 + GST + speaker table), gst.py, adaptation.py, layers.py
temotts/text/     text cleaners, ARPAbet symbols, g2p_en frontend
temotts/          audio.py (features), lm.py (emotion LM), vocoder.py (PWG), pipeline.py (inference)
scripts/          data preparation, Stage II targets, smoke test
configs/          stage1.yaml, stage2.yaml, MFA config, demo sentences
```

## Citation

```bibtex
@article{chandra2024temotts,
  title   = {Exploring speech style spaces with language models: Emotional TTS without emotion labels},
  author  = {Suresh Chandra, Shreeram and Du, Zongyang and Sisman, Berrak},
  journal = {arXiv preprint arXiv:2405.11413},
  year    = {2024}
}
```

## Acknowledgements

The FastSpeech2 and GST modules are ported from [ESPnet](https://github.com/espnet/espnet).
The vocoder is the pretrained LJSpeech model from
[ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN). The emotion language
model is
[michellejieli/emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier).
