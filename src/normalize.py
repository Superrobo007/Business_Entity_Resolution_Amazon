"""
Text normalization for business names and addresses.

Design goal: everything here is generic string/token processing, not a
lookup table keyed on a fixed set of countries. This is important because
the test set includes France, which never appears in training — a
normalizer that only knows US/India suffix lists will silently degrade
on French records. Country-specific abbreviation maps below are additive
(they help, they don't gate whether normalization runs).
"""

import re
import unicodedata

# --- Legal-suffix / abbreviation expansion -----------------------------
# Applied to name tokens. Keys are already-lowercased tokens (after
# punctuation stripping). This list is intentionally broad-strokes; it is
# meant to collapse the MOST common legal-entity noise, not to be
# exhaustive per country. Anything not in this map passes through
# unchanged, so unfamiliar (e.g. French) suffixes are simply left as-is
# and still get compared via fuzzy string similarity downstream.
NAME_SUFFIX_MAP = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl", "sa": "sa",
}

STOPWORD_CONNECTORS = {"and", "&", "the", "of"}

ADDRESS_ABBREV_MAP = {
    "street": "st", "st": "st", "str": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "near": "near", "nr": "near",
    "post": "post", "pin": "pin", "zip": "zip",
}

LANDMARK_PATTERN = re.compile(r"\b(near|nr\.?|opp\.?|opposite|behind|beside)\b\s+(.*)", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    """Fold transliteration/diacritic variants to plain ASCII where possible."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _basic_clean(text: str) -> str:
    text = text.lower().strip()
    text = _strip_accents(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(raw_name: str) -> str:
    """Return a canonicalized business name string for comparison."""
    if not raw_name:
        return ""
    text = _basic_clean(raw_name)
    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        if tok in STOPWORD_CONNECTORS:
            continue
        tok = NAME_SUFFIX_MAP.get(tok, tok)
        out_tokens.append(tok)
    return " ".join(out_tokens)


def name_tokens_sorted(raw_name: str) -> str:
    """Word-order-invariant version of the normalized name (handles transpositions)."""
    norm = normalize_name(raw_name)
    return " ".join(sorted(norm.split()))


def extract_landmark(raw_address: str) -> str:
    """Pull out a landmark reference ('Near SBI ATM') if present, else ''."""
    if not raw_address:
        return ""
    m = LANDMARK_PATTERN.search(raw_address)
    return _basic_clean(m.group(2)) if m else ""


def normalize_address(raw_address: str) -> str:
    """Return a canonicalized address string for comparison."""
    if not raw_address:
        return ""
    text = _basic_clean(raw_address)
    tokens = text.split()
    out_tokens = [ADDRESS_ABBREV_MAP.get(tok, tok) for tok in tokens]
    return " ".join(out_tokens)


def extract_numeric_tokens(text: str) -> set:
    """House/PIN/ZIP-like numeric tokens — strong signal when they match."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def add_normalized_columns(df, name_col="business_name", addr_col="business_address"):
    """Attach normalized_name, name_sorted, normalized_address, address_numbers columns."""
    df = df.copy()
    # Vectorized operations are faster than apply for large datasets
    names = df[name_col].values
    addrs = df[addr_col].values
    
    df["normalized_name"] = [normalize_name(n) for n in names]
    df["name_sorted"] = [name_tokens_sorted(n) for n in names]
    df["normalized_address"] = [normalize_address(a) for a in addrs]
    df["address_numbers"] = [extract_numeric_tokens(a) for a in addrs]
    df["landmark"] = [extract_landmark(a) for a in addrs]
    return df
