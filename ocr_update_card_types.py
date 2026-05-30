#!/usr/bin/env python3
"""
Update missing card types in a tcg-arena CardList.json using OCR.

Install:
    pip install easyocr opencv-python pillow requests numpy

Example:
    python ocr_update_card_types.py --input CardList.json --output CardList.updated.json

Optional review log:
    python ocr_update_card_types.py --input CardList.json --output CardList.updated.json --review ocr_review.json
"""

import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


KNOWN_CARD_TYPES = [
    "Character",
    "Action",
    "Item",
    "Location",
    "Attachment",
    "Event",
    "Spell",
    "Trap",
    "Resource",
    "Token",
]

TYPE_ALIASES = {
    "characters": "Character",
    "character": "Character",
    "action": "Action",
    "actions": "Action",
    "item": "Item",
    "items": "Item",
    "location": "Location",
    "locations": "Location",
    "attachment": "Attachment",
    "attachments": "Attachment",
    "event": "Event",
    "events": "Event",
    "spell": "Spell",
    "spells": "Spell",
    "trap": "Trap",
    "traps": "Trap",
    "resource": "Resource",
    "resources": "Resource",
    "token": "Token",
    "tokens": "Token",
}

MISSING_TYPE_VALUES = {"", "unknown", "none", "null", "n/a", "na"}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_front(card: Dict[str, Any]) -> Dict[str, Any]:
    return card.setdefault("face", {}).setdefault("front", {})


def is_missing_type(card: Dict[str, Any]) -> bool:
    top_type = str(card.get("type", "") or "").strip()
    front_type = str(get_front(card).get("type", "") or "").strip()

    return (
        top_type.lower() in MISSING_TYPE_VALUES
        or front_type.lower() in MISSING_TYPE_VALUES
        or "type" not in card
        or "type" not in get_front(card)
    )


def get_image_url(card: Dict[str, Any]) -> str:
    front = get_front(card)
    return str(front.get("image", "") or card.get("image", "") or "").strip()


def download_image(url: str, timeout: int = 30) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; VibesTCG-OCR-Script/1.0)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """
    General preprocessing. Upscales image to help OCR read small card text.
    """
    w, h = image.size
    image = image.resize((w * 2, h * 2))
    arr = np.array(image)

    try:
        import cv2

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    except Exception:
        return arr


def run_easyocr(image_array: np.ndarray, gpu: bool = False) -> List[Tuple[str, float]]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=gpu)
    results = reader.readtext(image_array, detail=1, paragraph=False)

    extracted: List[Tuple[str, float]] = []
    for item in results:
        if len(item) >= 3:
            text = str(item[1]).strip()
            confidence = float(item[2])
            if text:
                extracted.append((text, confidence))
    return extracted


def normalize_text(text: str) -> str:
    text = text.replace("|", "I")
    text = re.sub(r"[^A-Za-z0-9 /'-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_type(ocr_texts: List[Tuple[str, float]]) -> Tuple[Optional[str], float, List[str]]:
    candidates: List[Tuple[str, float, str]] = []

    for raw_text, confidence in ocr_texts:
        cleaned = normalize_text(raw_text)
        words = re.findall(r"[A-Za-z]+", cleaned.lower())

        for word in words:
            if word in TYPE_ALIASES:
                candidates.append((TYPE_ALIASES[word], confidence, raw_text))

        lower_line = cleaned.lower()
        for alias, canonical in TYPE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower_line):
                candidates.append((canonical, confidence, raw_text))

    if not candidates:
        return None, 0.0, []

    type_rank = {t: i for i, t in enumerate(KNOWN_CARD_TYPES)}
    candidates.sort(key=lambda x: (x[1], -type_rank.get(x[0], 999)), reverse=True)

    best_type = candidates[0][0]
    best_conf = candidates[0][1]
    matched_lines = [line for t, _, line in candidates if t == best_type]

    return best_type, best_conf, matched_lines[:5]


def update_card_type(card: Dict[str, Any], detected_type: str) -> None:
    card["type"] = detected_type
    get_front(card)["type"] = detected_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CardList.json")
    parser.add_argument("--output", required=True, help="Path to write updated CardList JSON")
    parser.add_argument("--review", default="ocr_type_review.json", help="Path to write OCR review log")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available")
    parser.add_argument("--overwrite-existing", action="store_true", help="OCR all cards, not just missing types")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    review_path = Path(args.review)

    cardlist = load_json(input_path)

    updated_count = 0
    skipped_count = 0
    failed_count = 0
    review_rows: List[Dict[str, Any]] = []

    for card_id, card in cardlist.items():
        if not isinstance(card, dict):
            skipped_count += 1
            continue

        if not args.overwrite_existing and not is_missing_type(card):
            skipped_count += 1
            continue

        name = card.get("name") or get_front(card).get("name") or f"Card {card_id}"
        image_url = get_image_url(card)

        row: Dict[str, Any] = {
            "id": card_id,
            "name": name,
            "image": image_url,
            "previous_type": {
                "top_level": card.get("type"),
                "front": get_front(card).get("type"),
            },
            "detected_type": None,
            "ocr_confidence": 0.0,
            "matched_lines": [],
            "all_ocr_text": [],
            "status": "pending",
            "error": None,
        }

        if not image_url:
            row["status"] = "failed"
            row["error"] = "No image URL found at face.front.image"
            failed_count += 1
            review_rows.append(row)
            continue

        try:
            image = download_image(image_url)
            processed = preprocess_for_ocr(image)
            ocr_texts = run_easyocr(processed, gpu=args.gpu)

            detected_type, confidence, matched_lines = detect_type(ocr_texts)

            row["all_ocr_text"] = [
                {"text": text, "confidence": round(conf, 4)}
                for text, conf in ocr_texts
            ]
            row["detected_type"] = detected_type
            row["ocr_confidence"] = round(confidence, 4)
            row["matched_lines"] = matched_lines

            if detected_type:
                update_card_type(card, detected_type)
                row["status"] = "updated"
                updated_count += 1
            else:
                row["status"] = "needs_manual_review"
                failed_count += 1

        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            failed_count += 1

        review_rows.append(row)

    save_json(output_path, cardlist)
    save_json(review_path, review_rows)

    print("Done.")
    print(f"Updated: {updated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Needs review / failed: {failed_count}")
    print(f"Updated file: {output_path}")
    print(f"Review log: {review_path}")


if __name__ == "__main__":
    main()
