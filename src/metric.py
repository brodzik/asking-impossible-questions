import collections
import string


def normalize_text(text):
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = " ".join(text.split())
    text = text.strip()
    return text


def get_tokens(text):
    return normalize_text(text).split() if text else []


def compute_em(gold_text, pred_text):
    return int(normalize_text(gold_text) == normalize_text(pred_text))


def compute_f1(gold_text, pred_text):
    gold_tokens = get_tokens(gold_text)
    pred_tokens = get_tokens(pred_text)

    intersection = sum((collections.Counter(gold_tokens) & collections.Counter(pred_tokens)).values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)

    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_tokens)
    recall = intersection / len(gold_tokens)
    f1 = float(2 * precision * recall / (precision + recall))

    return f1
