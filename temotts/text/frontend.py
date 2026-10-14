"""Text -> phones. The same lexicon (g2p_en) is given to MFA for alignment and
used at inference, so training and inference phone sequences agree."""
import re

from temotts.text.cleaners import english_cleaners
from temotts.text.symbols import ARPABET_SET, SP

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[^\sa-z']")
_PAUSE_PUNCT = set(",.!?;:()\"")


def tokenize(clean_text):
    """Cleaned text -> list of words and punctuation tokens."""
    return _TOKEN_RE.findall(clean_text)


def words_of(clean_text):
    return [t for t in tokenize(clean_text) if t[0].isalpha()]


def load_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] not in lex:
                lex[parts[0]] = parts[1].split()
    return lex


def save_lexicon(lex, path):
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted(lex):
            f.write(f"{w}\t{' '.join(lex[w])}\n")


class Frontend:
    def __init__(self, lexicon=None):
        self.lexicon = dict(lexicon or {})
        self._g2p = None

    def word_to_phones(self, word):
        """CMUdict (first pronunciation) for the whole word, else g2p_en's neural G2P.
        Whole-word lookup first because g2p_en tokenises contractions (don't -> do n't)."""
        if word not in self.lexicon:
            if self._g2p is None:
                from g2p_en import G2p

                self._g2p = G2p()
            prons = self._g2p.cmu.get(word)
            phones = prons[0] if prons else self._g2p(word)
            self.lexicon[word] = [p for p in phones if p in ARPABET_SET]
        return self.lexicon[word]

    def __call__(self, text):
        """Raw text -> phone list; pause punctuation -> 'sp' (no leading/trailing sp,
        matching the MFA-trimmed training data)."""
        phones = []
        for tok in tokenize(english_cleaners(text)):
            if tok[0].isalpha():
                phones.extend(self.word_to_phones(tok))
            elif tok in _PAUSE_PUNCT and phones and phones[-1] != SP:
                phones.append(SP)
        while phones and phones[-1] == SP:
            phones.pop()
        return phones
