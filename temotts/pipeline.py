"""Load trained stages and run TEMOTTS text-only inference (paper Fig. 3):
text -> LM emotion embedding -> adaptation module -> GST weights -> style embedding
     -> FastSpeech2 (phones from the same lexicon/G2P) -> PWG vocoder."""
import torch

from temotts.data import Stats
from temotts.models.adaptation import AdaptationModule
from temotts.models.fastspeech2 import FastSpeech2
from temotts.text.frontend import Frontend
from temotts.text.symbols import phones_to_ids


def load_stage1(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FastSpeech2(idim=len(ck["symbols"]), **ck["config"]["model"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model, Stats.from_state_dict(ck["stats"]), Frontend(ck["lexicon"]), ck


def load_stage2(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["model_args"]
    module = AdaptationModule(m["in_dim"], tuple(m["hidden"]), m["n_tokens"])
    module.load_state_dict(ck["model"])
    return module.eval().to(device), ck


class TEMOTTS:
    def __init__(self, stage1_ckpt, stage2_ckpt=None, asset_dir=None, device="cpu", lm=None):
        self.device = device
        self.fs2, self.stats, self.frontend, ck1 = load_stage1(stage1_ckpt, device)
        self.emotional_sid = ck1["speakers"]["emotional"]
        self.adapter = load_stage2(stage2_ckpt, device)[0] if stage2_ckpt else None
        self.lm = lm
        if self.adapter is not None and self.lm is None:
            from temotts.lm import EmotionTextEncoder

            self.lm = EmotionTextEncoder(device=device)
        self.vocoder = None
        if asset_dir is not None:
            from temotts.vocoder import Vocoder

            self.vocoder = Vocoder(asset_dir, device=device)

    @torch.no_grad()
    def predict_weights(self, texts):
        emb, probs = self.lm(texts)
        return self.adapter.predict_weights(emb.to(self.device)), probs

    @torch.no_grad()
    def synthesize(self, text, weights=None, ref_mel=None, sid=None, alpha=1.0):
        """Text-only by default. Optionally pass GST `weights` (tokens,) or a
        normalised reference mel (T, 80) instead. Returns dict(wav, mel, weights, phones)."""
        sid = self.emotional_sid if sid is None else sid
        lm_probs = None
        if ref_mel is not None:
            style, w = self.fs2.style_from_reference(ref_mel.unsqueeze(0).to(self.device))
            weights = w.mean(1)[0]
        else:
            if weights is None:
                w, lm_probs = self.predict_weights([text])
                weights = w[0]
            style = self.fs2.style_from_weights(weights.view(1, -1).to(self.device))
        phones = self.frontend(text)
        ids = torch.tensor(phones_to_ids(phones), dtype=torch.long, device=self.device)
        out = self.fs2.inference(ids, sids=sid, style_embs=style, alpha=alpha)
        mel = self.stats.denorm_mel(out["feat_gen"].float().cpu().numpy())
        wav = self.vocoder(mel) if self.vocoder is not None else None
        return dict(wav=wav, mel=mel, weights=weights.float().cpu().numpy(), phones=phones, lm_probs=lm_probs)
