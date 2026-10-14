"""English text normalisation (keithito/tacotron `english_cleaners`)."""
import re

import inflect
from unidecode import unidecode

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
        ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
        ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
        ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ]
]


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return "%s %s, %s %s" % (dollars, "dollar" if dollars == 1 else "dollars", cents, "cent" if cents == 1 else "cents")
    if dollars:
        return "%s %s" % (dollars, "dollar" if dollars == 1 else "dollars")
    if cents:
        return "%s %s" % (cents, "cent" if cents == 1 else "cents")
    return "zero dollars"


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        if num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return _inflect.number_to_words(num, andword="")


def expand_numbers(text):
    text = re.sub(_comma_number_re, lambda m: m.group(1).replace(",", ""), text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, lambda m: m.group(1).replace(".", " point "), text)
    text = re.sub(_ordinal_re, lambda m: _inflect.number_to_words(m.group(0)), text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def english_cleaners(text):
    text = unidecode(text)
    text = text.replace("%", " percent ").replace("&", " and ")
    text = expand_numbers(text)
    text = text.lower()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  # well-known -> well known
    return re.sub(r"\s+", " ", text).strip()
