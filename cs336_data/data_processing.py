from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
from pathlib import Path
from cs336_data.common import get_shared_assets_path

_FASTTEXT_MODEL = None


def get_fasttext_model():
    global _FASTTEXT_MODEL

    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL

    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            f"fastText language identification model not found at {model_path}"
        )
    _FASTTEXT_MODEL = fasttext.load_model(str(model_path))
    return _FASTTEXT_MODEL


def identify_language(text: str) -> tuple[str, float]:
    cleaned_text = text.replace("\n", " ")
    model = get_fasttext_model()
    labels, probabilities = model.predict(cleaned_text, k=1)
    if not labels:
        raise ValueError(f"model.predict({cleaned_text}) returned no prediction")
    if len(labels) != 1:
        raise ValueError(
            f"model.predict({cleaned_text}) unexpectedly did not return exactly one prediction: {labels}, {probabilities}"
        )
    label = labels[0]
    score = float(probabilities[0])
    if label.startswith("__label__"):
        lang = label[len("__label__") :]
    else:
        lang = label
    return lang, score


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
