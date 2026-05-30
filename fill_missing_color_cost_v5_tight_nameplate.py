#!/usr/bin/env python3
import argparse, json, re
from io import BytesIO
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import requests
from PIL import Image

POSSIBLE_COLORS = ["Red", "Blue", "Yellow", "Purple", "Green", "Colorless"]
HSV_RANGES = {
    "Red": [((0, 60, 50), (12, 255, 255)), ((165, 60, 50), (179, 255, 255))],
    "Blue": [((90, 50, 50), (135, 255, 255))],
    "Yellow": [((18, 50, 70), (40, 255, 255))],
    "Purple": [((130, 40, 40), (164, 255, 255))],
    "Green": [((45, 40, 40), (89, 255, 255))],
}
COLORLESS_TYPES = {"relic", "fit"}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def get_front(card): return card.setdefault("face", {}).setdefault("front", {})
def get_card_type(card): return str(card.get("type") or get_front(card).get("type") or "").strip()
def get_card_name(card_id, card): return str(card.get("name") or get_front(card).get("name") or f"Card {card_id}").strip()
def get_image_url(card): return str(get_front(card).get("image", "") or card.get("image", "") or "").strip()

def is_missing_color(card):
    c = card.get("Color")
    if c is None: return True
    if isinstance(c, list): return len(c) == 0
    if isinstance(c, str): return c.strip() == ""
    return False

def download_image(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def make_easyocr_reader(gpu=False):
    import easyocr
    return easyocr.Reader(["en"], gpu=gpu)

def preprocess_for_ocr(image, scale=3):
    w, h = image.size
    image = image.resize((w * scale, h * scale))
    arr = np.array(image)
    try:
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    except Exception:
        return arr

def run_easyocr_raw(reader, image, scale=3):
    arr = preprocess_for_ocr(image, scale)
    results = reader.readtext(arr, detail=1, paragraph=False)
    out = []
    for item in results:
        if len(item) >= 3:
            bbox, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if text:
                scaled_bbox = [[float(x) / scale, float(y) / scale] for x, y in bbox]
                out.append((scaled_bbox, text, conf))
    return out

def norm(text): return re.sub(r"[^a-z0-9]+", "", text.lower())

def bbox_to_rect(bbox):
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def find_name_bbox(ocr_results, expected_name, min_similarity=0.50):
    expected = norm(expected_name)
    best, candidates = None, []
    for bbox, text, conf in ocr_results:
        n = norm(text)
        if not n: continue
        sim = SequenceMatcher(None, expected, n).ratio()
        x1, y1, x2, y2 = bbox_to_rect(bbox)
        width = x2 - x1
        # Prefer similar, confident, wide, top-center text.
        score = sim * 0.75 + conf * 0.20 + min(width / 500, 1) * 0.10 - (y1 / 1000)
        candidates.append({"text": text, "similarity": round(sim,4), "confidence": round(conf,4), "score": round(score,4), "bbox": [x1,y1,x2,y2]})
        if best is None or score > best[0]: best = (score, sim, conf, text, (x1,y1,x2,y2))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if best and best[1] >= min_similarity:
        return best[4], {"method":"ocr_name_match", "matched_text":best[3], "similarity":round(best[1],4), "confidence":round(best[2],4), "candidates":candidates[:8]}
    return None, {"method":"name_not_found", "candidates":candidates[:8]}

def detect_cost_from_ocr(ocr_results):
    candidates = []
    for bbox, raw, conf in ocr_results:
        x1, y1, x2, y2 = bbox_to_rect(bbox)
        cleaned = raw.replace("O","0").replace("o","0").replace("I","1").replace("l","1")
        for n in re.findall(r"\b\d{1,2}\b", cleaned):
            v = int(n)
            if 0 <= v <= 99:
                left_bonus = 1.0 if x1 < 250 else 0.0
                candidates.append((v, conf + left_bonus, conf, raw))
    if not candidates: return None, 0.0, []
    candidates.sort(key=lambda x: (x[1], -len(x[3])), reverse=True)
    best_v, _score, best_conf, _ = candidates[0]
    return str(best_v), best_conf, [line for v, _, _, line in candidates if v == best_v][:5]

def crop_tight_nameplate(image, name_rect):
    # Key fix: stay tightly around the name text, so blue artwork outside the rounded nameplate is excluded.
    w, h = image.size
    x1, y1, x2, y2 = name_rect
    text_w, text_h = max(1, x2-x1), max(1, y2-y1)
    left = max(0, x1 - int(text_w * 0.22))
    right = min(w, x2 + int(text_w * 0.22))
    top = max(0, y1 - int(text_h * 0.45))
    bottom = min(h, y2 + int(text_h * 0.45))
    rect = (left, top, right, bottom)
    return image.crop(rect), rect

def fallback_nameplate_crop(image):
    w, h = image.size
    rect = (int(w * 0.25), int(h * 0.065), int(w * 0.78), int(h * 0.135))
    return image.crop(rect), rect

def classify_nameplate_color(crop, card_type, dual_similarity_ratio=0.75):
    if card_type.lower() in COLORLESS_TYPES: return ["Colorless"], {}, "type_forced_colorless"
    try:
        import cv2
    except Exception:
        return [], {}, "opencv_missing"
    arr = np.array(crop)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    # Remove black text, white glare, grey border, low saturation pixels.
    valid = ((hsv[:,:,1] >= 45) & (hsv[:,:,2] >= 80)).astype(np.uint8)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0: return ["Colorless"], {}, "no_valid_pixels_default_colorless"
    coverage = {}
    for color, ranges in HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask_total = cv2.bitwise_or(mask_total, mask)
        mask_total = cv2.bitwise_and(mask_total, mask_total, mask=valid)
        coverage[color] = int(np.count_nonzero(mask_total)) / valid_pixels
    sorted_colors = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    top_color, top_ratio = sorted_colors[0]
    second_color, second_ratio = sorted_colors[1]
    if top_ratio < 0.20:
        return ["Colorless"], {k:round(v,5) for k,v in coverage.items()}, "low_confidence_default_colorless"
    detected = [top_color]
    reason = "single_tight_nameplate_color"
    if second_ratio >= 0.12 and top_ratio > 0 and (second_ratio / top_ratio) >= dual_similarity_ratio:
        detected.append(second_color); reason = "dual_similar_tight_nameplate_color"
    detected = [c for c in POSSIBLE_COLORS if c in detected]
    return detected, {k:round(v,5) for k,v in coverage.items()}, reason

def update_card(card, colors, cost):
    card["Color"] = colors
    if cost is not None:
        card["cost"] = cost
        card["Cost"] = cost
        get_front(card)["cost"] = cost

def safe_filename(text): return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:80] or "card"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--review", default="color_cost_review.json")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--top-percent", type=float, default=0.22)
    p.add_argument("--debug-crops", default=None)
    args = p.parse_args()

    cards = load_json(Path(args.input))
    reader = make_easyocr_reader(gpu=args.gpu)
    debug_dir = Path(args.debug_crops) if args.debug_crops else None
    if debug_dir: debug_dir.mkdir(parents=True, exist_ok=True)
    review = []
    updated = skipped = failed = 0

    for card_id, card in cards.items():
        if not isinstance(card, dict): skipped += 1; continue
        if not args.overwrite_existing and not is_missing_color(card): skipped += 1; continue
        name = get_card_name(card_id, card)
        card_type = get_card_type(card)
        image_url = get_image_url(card)
        row = {"id":card_id,"name":name,"type":card_type,"image":image_url,"previous_color":card.get("Color"),"detected_colors":[],"color_coverage":{},"color_reason":None,"name_match":{},"nameplate_rect":None,"detected_cost":None,"cost_confidence":0.0,"cost_matched_lines":[],"status":"pending","error":None}
        if not image_url:
            row["status"]="failed"; row["error"]="No image URL"; failed += 1; review.append(row); continue
        try:
            image = download_image(image_url)
            w,h = image.size
            top_crop = image.crop((0,0,w,int(h*args.top_percent)))
            ocr = run_easyocr_raw(reader, top_crop)
            cost, cost_conf, cost_lines = detect_cost_from_ocr(ocr)
            name_rect, match_debug = find_name_bbox(ocr, name)
            row["name_match"] = match_debug
            if name_rect:
                crop, rect = crop_tight_nameplate(image, name_rect)
            else:
                crop, rect = fallback_nameplate_crop(image)
                row["color_reason"] = "fallback_fixed_nameplate"
            colors, coverage, reason = classify_nameplate_color(crop, card_type)
            if row["color_reason"]: reason = row["color_reason"] + "__" + reason
            update_card(card, colors, cost)
            row.update({"detected_colors":colors,"color_coverage":coverage,"color_reason":reason,"nameplate_rect":list(rect),"detected_cost":cost,"cost_confidence":round(cost_conf,4),"cost_matched_lines":cost_lines,"status":"updated" if cost is not None else "updated_color_only_needs_cost_review"})
            if debug_dir: crop.save(debug_dir / f"{card_id}_{safe_filename(name)}_nameplate.png")
            updated += 1
        except Exception as e:
            row["status"]="failed"; row["error"]=str(e); failed += 1
        review.append(row)

    save_json(Path(args.output), cards)
    save_json(Path(args.review), review)
    print("Done.")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Updated file: {args.output}")
    print(f"Review log: {args.review}")
    if debug_dir: print(f"Debug crops folder: {debug_dir}")

if __name__ == "__main__": main()
