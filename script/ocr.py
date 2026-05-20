import re
import cv2
from paddleocr import PaddleOCR

# Paddle 3.3+ on Windows CPU: oneDNN can crash with NotImplementedError.
# See: https://github.com/PaddlePaddle/Paddle/issues/77340
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def _sort_lines_by_position(result):
    texts = result.get("rec_texts") or []
    boxes = result.get("rec_polys") or result.get("rec_boxes")
    if not texts:
        return []

    if boxes is None or len(boxes) != len(texts):
        return list(texts)

    indexed = []
    for text, box in zip(texts, boxes):
        arr = box if hasattr(box, "shape") else box
        y = float(arr[:, 1].mean()) if getattr(arr, "ndim", 0) == 2 else float(arr[1])
        indexed.append((y, text))

    indexed.sort(key=lambda item: item[0])
    return [text for _, text in indexed]


def normalize_plate_text(text):
    text = re.sub(r"[^0-9A-Z ]", "", text.upper())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_plate(cropped, debug=False):
    if cropped is None or cropped.size == 0:
        return ""

    results = ocr.predict(cropped)
    if not results:
        return ""

    page = results[0]
    lines = _sort_lines_by_position(page)

    if debug:
        for line in lines:
            print(f"  line -> '{line}'")

    plate = normalize_plate_text(" ".join(lines))
    return plate
