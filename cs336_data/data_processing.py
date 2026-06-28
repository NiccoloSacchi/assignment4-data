from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
import nltk
import numpy as np
from enum import Enum
from cs336_data.common import get_shared_assets_path


class ModelType(Enum):
    LANG_CLASSIFIER = 1
    NSFW_CLASSIFIER = 2
    HATE_SPEECH_CLASSIFIER = 3


_FASTTEXT_MODELS = {}


def get_fasttext_model(model_type: ModelType):
    global _FASTTEXT_MODELS

    if model_type in _FASTTEXT_MODELS:
        return _FASTTEXT_MODELS[model_type]

    model_path = get_shared_assets_path() / "classifiers"
    match model_type:
        case ModelType.LANG_CLASSIFIER:
            model_path = model_path / "lid.176.bin"
        case ModelType.NSFW_CLASSIFIER:
            model_path = model_path / "dolma_fasttext_nsfw_jigsaw_model.bin"
        case ModelType.HATE_SPEECH_CLASSIFIER:
            model_path = model_path / "dolma_fasttext_hatespeech_jigsaw_model.bin"
        case _:
            # Fallback wildcard for safety
            raise ValueError(f"Unknown model type: {model_type}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"fastText model {model_type} not found at {model_path}"
        )
    _FASTTEXT_MODELS[model_type] = fasttext.load_model(str(model_path))
    return _FASTTEXT_MODELS[model_type]


def identify_language(text: str) -> tuple[str, float]:
    return compute_label_and_score(text.replace("\n", " "), ModelType.LANG_CLASSIFIER)


def extract_text_from_warc(warc_file_path, n=10):
    texts = []
    with open(warc_file_path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
            # Use record.http_content_type to look INSIDE the HTTP response payload
            # (Checks for 'text/html', 'application/xhtml+xml', etc.).
            if (
                not record.http_content_type
                or "html" not in record.http_content_type.lower()
            ):
                continue

            # Read the raw byte string from the record body.
            html_bytes = record.reader.read()
            if not html_bytes:
                continue

            texts.append(
                {
                    "record_uri": record.headers.get("WARC-Target-URI"),
                    "text": extract_text_from_html_bytes(html_bytes),
                }
            )
            if len(texts) == n:
                break
    return texts


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    # Detect the encoding of the input byte string and decode.
    encoding = detect_encoding(html_bytes) or "utf-8"
    html_str = html_bytes.decode(encoding, errors="ignore")

    # Resiliparse extracts plain text directly from the byte string.
    return extract_plain_text(html_str)


def mask_emails(text: str) -> tuple[str, int]:
    # Matches typical email formats
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    new_text, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return new_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    # Matches common US phone formats: e.g., 2831823829, (283)-182-3829, (283) 182 3829, 283-182-3829
    pattern = r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b"
    new_text, count = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return new_text, count


def mask_ips(text: str) -> tuple[str, int]:
    # Matches 4 numbers separated by dots
    pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    count = 0

    def repl(match):
        nonlocal count
        octets = match.group(0).split(".")
        if all(0 <= int(octet) <= 255 for octet in octets):
            count += 1
            return "|||IP_ADDRESS|||"
        return match.group(0)

    new_text = re.sub(pattern, repl, text)
    return new_text, count


def classify_nsfw(text: str) -> tuple[str, float]:
    return compute_label_and_score(text, ModelType.NSFW_CLASSIFIER)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    return compute_label_and_score(text, ModelType.HATE_SPEECH_CLASSIFIER)


def compute_label_and_score(text: str, model_type: ModelType):
    model = get_fasttext_model(model_type)
    labels, probabilities = model.predict(text)

    if not labels:
        raise ValueError(f"model.predict({text}) returned no prediction")
    if len(labels) != 1:
        raise ValueError(
            f"model.predict({text}) unexpectedly did not return exactly one prediction: {labels}, {probabilities}"
        )
    label = labels[0]
    score = float(probabilities[0])
    if label.startswith("__label__"):
        return label[len("__label__") :], score
    return label, score


def gopher_quality_filter(text: str) -> bool:
    # Download the required tokenization models (only needed once).
    nltk.download("punkt_tab", quiet=True)

    lines = text.splitlines()
    tokens = nltk.word_tokenize(text)
    words = [t for t in tokens if any(char.isalpha() for char in t)]
    if not words:
        return False

    # Contain less than 50 or more than 100,000 words.
    words_count = len(words)
    if words_count < 50 or words_count > 100_000:
        return False

    # Have a mean word length outside the range of 3 to 10 characters.
    word_len_mean = np.mean([len(w) for w in words])
    if word_len_mean < 3 or word_len_mean > 10:
        return False

    # Have more than 30% of lines ending with an ellipsis (“...”).
    valid_lines = [line.strip() for line in lines if line.strip()]
    if valid_lines:
        ellipsis_lines = sum(
            1 for line in valid_lines if line.endswith("...") or line.endswith("…")
        )
        if (ellipsis_lines / len(valid_lines)) > 0.30:
            return False

    # Contain less than 80% of words with at least one alphabetic character.
    token_count = len(tokens)
    if (words_count / token_count) < 0.80:
        return False

    return True
