"""Celebrity identity constants for UnlearnDiffAtk-style celebrity erasure data.

Must stay importable under the GCD conda env (Python 3.6).
"""

from typing import Any, Dict, List

CELEBRITIES = {
    "elon_musk": {
        "slug": "elon_musk",
        "display_name": "Elon Musk",
        "gcd_label": "Elon_Musk_[m.03nzf1]",
        "gcd_name": "Elon Musk",
    },
    "adam_lambert": {
        "slug": "adam_lambert",
        "display_name": "Adam Lambert",
        "gcd_label": "Adam_Lambert_[m.07sbny7]",
        "gcd_name": "Adam Lambert",
    },
    "taylor_swift": {
        "slug": "taylor_swift",
        "display_name": "Taylor Swift",
        "gcd_label": "Taylor_Swift_[m.0dl567]",
        "gcd_name": "Taylor Swift",
    },
}  # type: Dict[str, Dict[str, Any]]

def list_slugs():
    # type: () -> List[str]
    return list(CELEBRITIES.keys())


def get_celebrity(slug):
    # type: (str) -> Dict[str, Any]
    if slug not in CELEBRITIES:
        raise KeyError(f"unknown celebrity slug: {slug}; choose from {list_slugs()}")
    return CELEBRITIES[slug]


def normalize_gcd_name(label_or_name):
    # type: (str) -> str
    """Convert GCD raw label or display name to comparable plain name."""
    text = str(label_or_name).strip()
    if "_[ " in text or "_[" in text:
        text = text.split("_[", 1)[0]
    return text.replace("_", " ").strip().lower()
