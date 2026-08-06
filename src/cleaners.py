"""
Domain-aware post-processing for raw OCR strings.

clean_naive  -> just strips whitespace, no assumptions
clean_smart  -> strip -> keep only digits/decimal point/sign -> canonicalise
                to a float. This is the accuracy multiplier: raw OCR output
                is often noisy ("O" vs "0", stray characters, doubled
                decimal points), and a numeric display only ever needs
                a number out the other end.
"""

import re


def clean_naive(text):
    return text.strip()


def clean_smart(text, decimal_places=None):
    if text is None:
        return None

    t = text.strip().upper()
    t = t.replace("O", "0").replace("S", "5").replace("B", "8")
    t = re.sub(r"[^0-9.\-]", "", t)

    if t.count(".") > 1:
        head, *rest = t.split(".")
        t = head + "." + "".join(rest)

    if t in ("", ".", "-", "-."):
        return None

    try:
        value = float(t)
    except ValueError:
        return None

    if decimal_places is not None:
        value = round(value, decimal_places)

    return value
