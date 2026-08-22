import re
from typing import List


TOKEN_REGEX = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = frozenset("""
a an the is are was were be been being of in on at to for from by with
as and or not do does did what which who how why when where this that
these those you your it its can could should would will may might use
used using
""".split())


def tokenize(text: str) -> List[str]:
    """Tokenize the input text into a list of lowercase tokens,
    excluding stopwords."""
    tokens: List[str] = []
    for t in TOKEN_REGEX.findall(text):
        low = t.lower()
        if low not in STOPWORDS:
            tokens.append(low)
    return tokens
