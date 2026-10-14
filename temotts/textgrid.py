"""MFA TextGrid (long format) -> phones + frame durations."""
import re

import numpy as np

from temotts.text.symbols import SP, SPN

_TIER_RE = re.compile(r'class = "IntervalTier"\s*name = "([^"]*)"(.*?)(?=class = "|\Z)', re.S)
_IV_RE = re.compile(r'xmin = ([-\d.eE+]+)\s*xmax = ([-\d.eE+]+)\s*text = "((?:[^"]|"")*)"')
SILENCES = {"", "sil", "sp", "silence", "<eps>"}


def read_tiers(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return {
        name: [(float(a), float(b), t.replace('""', '"')) for a, b, t in _IV_RE.findall(body)]
        for name, body in _TIER_RE.findall(content)
    }


def phones_and_durations(path, sample_rate, hop_length):
    """Returns (phones, durations, start_frame, end_frame).

    Leading/trailing silences are trimmed; inner silences become 'sp';
    'spn' (MFA OOV) is kept. Frame boundaries are rounded once per interval
    edge, so durations telescope and sum(durations) == end_frame - start_frame.
    """
    tiers = read_tiers(path)
    ivs = tiers.get("phones")
    if ivs is None:
        raise ValueError(f"no phones tier in {path}")
    labels = [SP if t.strip().lower() in SILENCES else (SPN if t.strip().lower() == "spn" else t.strip()) for _, _, t in ivs]
    speech = [i for i, p in enumerate(labels) if p != SP]
    if not speech:
        raise ValueError(f"no speech in {path}")
    first, last = speech[0], speech[-1]
    phones, durations = [], []
    to_frame = lambda sec: int(np.round(sec * sample_rate / hop_length))
    for (s, e, _), p in zip(ivs[first : last + 1], labels[first : last + 1]):
        d = to_frame(e) - to_frame(s)
        if p == SP and phones and phones[-1] == SP:
            durations[-1] += d
            continue
        phones.append(p)
        durations.append(d)
    return phones, durations, to_frame(ivs[first][0]), to_frame(ivs[last][1])
