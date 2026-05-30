#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        cards = json.load(f)

    updated = 0

    for card_id, card in cards.items():
        if not isinstance(card, dict):
            continue

        if str(card.get("type", "")).strip() == "Action":
            card["cost"] = "None"
            card["Cost"] = "None"

            if "face" in card and "front" in card["face"]:
                card["face"]["front"]["cost"] = "None"

            updated += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

    print(f"Updated {updated} Action cards.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
