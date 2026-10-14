"""Acoustic features.

* log-mel: identical to Parallel WaveGAN's `logmelfilterbank` (librosa STFT,
  magnitude, slaney mel, log10, eps 1e-10) so the pretrained LJSpeech PWG can
  vocode the outputs directly.
* energy: ESPnet2 `Energy` (L2 norm of the STFT frame magnitude).
* pitch: ESPnet2 `Dio` (pyworld dio + stonemask, continuous, log F0).
All three share one frame grid (hop 256, centered frames).
"""
import numpy as np
import soundfile as sf


def load_wav(path, sr):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        import librosa

        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x.astype(np.float32)


def continuous_f0(f0):
    """ESPnet Dio._convert_to_continuous_f0."""
    if (f0 == 0).all():
        return f0
    from scipy.interpolate import interp1d

    nz = np.where(f0 != 0)[0]
    f0 = f0.copy()
    f0[: nz[0]] = f0[nz[0]]
    f0[nz[-1]:] = f0[nz[-1]]
    nz = np.where(f0 != 0)[0]
    return interp1d(nz, f0[nz])(np.arange(f0.shape[0]))


def average_by_duration(x, durations):
    """ESPnet token averaging: mean of the positive frame values of each token
    (0 for tokens with zero duration or no positive frames)."""
    out = np.zeros(len(durations), dtype=np.float32)
    bounds = np.concatenate([[0], np.cumsum(durations)])
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg = x[s:e]
        seg = seg[seg > 0]
        if len(seg):
            out[i] = seg.mean()
    return out


class FeatureExtractor:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=None, window="hann",
                 n_mels=80, fmin=80, fmax=7600, f0min=80, f0max=400):
        import librosa

        self.sr, self.n_fft, self.hop, self.win_length, self.window = sample_rate, n_fft, hop_length, win_length, window
        self.f0min, self.f0max = f0min, f0max
        self.mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, x):
        """x: float waveform -> (logmel [T, n_mels], energy [T], log-f0 [T])."""
        import librosa

        spc = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                                  window=self.window, pad_mode="reflect")).T
        logmel = np.log10(np.maximum(1e-10, spc @ self.mel_basis.T)).astype(np.float32)
        energy = np.sqrt(np.maximum((spc ** 2).sum(axis=1), 1e-10)).astype(np.float32)
        f0 = self.log_f0(x, len(logmel))
        return logmel, energy, f0

    def log_f0(self, x, n_frames):
        import pyworld

        x = x.astype(np.float64)
        f0, t = pyworld.dio(x, self.sr, f0_floor=self.f0min, f0_ceil=self.f0max, channels_in_octave=2,
                            frame_period=1000 * self.hop / self.sr)
        f0 = continuous_f0(pyworld.stonemask(x, f0, t, self.sr))
        nz = f0 > 0
        f0[nz] = np.log(f0[nz])
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return f0[:n_frames].astype(np.float32)
