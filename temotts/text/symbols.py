"""Phone inventory: stress-marked ARPAbet (MFA english_us_arpa / g2p_en) + specials.

Layout follows ESPnet: index 0 = padding, last index = <sos/eos>.
"""
_VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
_CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S",
               "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
ARPABET = sorted([v + s for v in _VOWELS for s in "012"] + _CONSONANTS)
ARPABET_SET = frozenset(ARPABET)

PAD, UNK, SP, SPN, EOS = "<pad>", "<unk>", "sp", "spn", "<sos/eos>"
SYMBOLS = [PAD, UNK, SP, SPN] + ARPABET + [EOS]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def phones_to_ids(phones):
    return [SYMBOL_TO_ID.get(p, SYMBOL_TO_ID[UNK]) for p in phones]
