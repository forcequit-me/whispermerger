"""Portugal runs (#23): CAMOES's words and spelling, large-v3's capitalization and punctuation.

CAMOES hears European speech better and spells it the European way, but writes lowercase with
almost no punctuation. The two word sequences are aligned ignoring case, punctuation and accents;
where a word lines up, it takes large-v3's capitalization and the punctuation around it. Where
they disagree, CAMOES's word stays as it is.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)   # leading punct, word, trailing punct


def _key(token: str) -> str:
    t = unicodedata.normalize("NFD", token.lower())
    return re.sub(r"[^\w']+", "", "".join(c for c in t if not unicodedata.combining(c)))


def _transfer(camoes: str, v3: str) -> str:
    _, word, _ = _SPLIT.match(camoes).groups()
    lead, v3_word, trail = _SPLIT.match(v3).groups()
    if len(v3_word) > 1 and v3_word.isupper():
        word = word.upper()
    elif v3_word[:1].isupper():
        word = word[:1].upper() + word[1:]
    return lead + word + trail


def merge_punctuation(camoes_segments: list[str], v3_text: str) -> list[str]:
    """Each CAMOES segment's text with large-v3's capitals and punctuation on the matching words.
    Word count per segment is unchanged, so segment boundaries and timings stay CAMOES's."""
    seg_tokens = [s.split() for s in camoes_segments]
    c = [t for toks in seg_tokens for t in toks]
    v = v3_text.split()
    out = list(c)
    sm = difflib.SequenceMatcher(None, [_key(t) for t in c], [_key(t) for t in v], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if _key(c[i1 + k]):                       # skip bare punctuation tokens
                    out[i1 + k] = _transfer(c[i1 + k], v[j1 + k])
    merged, i = [], 0
    for toks in seg_tokens:
        merged.append(" ".join(out[i:i + len(toks)]))
        i += len(toks)
    return merged
