# 글자수, 하이라이트, 중복탐지(선택)

import re
from collections import Counter

def count_chars(text: str) -> int:
    return len(text.replace(" ", "").replace("\n", ""))

def detect_repeated_keywords(texts: list[str], top_n: int = 10):
    tokens = []
    for t in texts:
        t = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
        tokens += [w for w in t.split() if len(w) >= 2]
    freq = Counter(tokens)
    return freq.most_common(top_n)
