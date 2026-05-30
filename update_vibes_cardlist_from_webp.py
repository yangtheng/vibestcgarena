import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_PAGE_URL = "https://www.vibes-tracker.com/set3-spoilers"


def slug_to_card_name(slug: str) -> str:
    """
    Convert 'unidentified-flying-birb' -> 'Unidentified Flying Birb'.
    Keeps common TCG/crypto abbreviations in uppercase where useful.
    """
    special_words = {
        "nft": "NFT",
        "tcg": "TCG",
        "glhf": "GLHF",
        "irl": "IRL",
        "gm": "GM",
        "xp": "XP",
        "hp": "HP",
        "atk": "ATK",
    }

    words = slug.replace("_", "-").split("-")
    return " ".join(special_words.get(word.lower(), word.capitalize()) for word in words if word)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def get_script_urls(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    script_urls = []

    for script in soup.find_all("script", src=True):
        script_urls.append(urljoin(page_url, script["src"]))

    return script_urls


def extract_webp_urls(text: str, base_url: str) -> set[str]:
    """
    Extracts .webp URLs from HTML/JS text.

    Handles:
    - absolute URLs: https://site/path/card.webp
    - root-relative URLs: /set3-spoilers/card.webp
    - relative URLs: ./card.webp or card.webp
    """
    webp_urls = set()

    patterns = [
        r"https?://[^\"'()\\\s]+?\.webp",
        r"/[^\"'()\\\s]+?\.webp",
        r"(?:\./)?[^\"'()\\\s]+?\.webp",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            # Remove common JS/CSS trailing escapes or punctuation.
            cleaned = match.replace("\\/", "/").rstrip(";,")
            webp_urls.add(urljoin(base_url, cleaned))

    return webp_urls


def scrape_webp_urls(page_url: str) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; VibesTCGCardScraper/1.0)"
    }

    response = requests.get(page_url, headers=headers, timeout=30)
    response.raise_for_status()

    html = response.text
    all_texts = [html]

    for script_url in get_script_urls(page_url, html):
        try:
            script_response = requests.get(script_url, headers=headers, timeout=30)
            script_response.raise_for_status()
            all_texts.append(script_response.text)
        except requests.RequestException as exc:
            print(f"Warning: could not fetch JS bundle {script_url}: {exc}")

    webp_urls = set()
    for text in all_texts:
        webp_urls.update(extract_webp_urls(text, page_url))

    # Keep likely card art URLs from this page path.
    parsed_page = urlparse(page_url)
    likely_urls = []
    for url in webp_urls:
        parsed = urlparse(url)
        if parsed.netloc == parsed_page.netloc and "/set3-spoilers/" in parsed.path:
            likely_urls.append(url)

    return sorted(set(likely_urls))


def webp_url_to_card(url: str, card_id: str) -> dict:
    filename = Path(urlparse(url).path).name
    slug = filename.removesuffix(".webp")
    name = slug_to_card_name(slug)

    return {
        "id": card_id,
        "isToken": False,
        "face": {
            "front": {
                "name": name,
                "type": "",
                "cost": "",
                "image": url,
                "isHorizontal": False
            }
        },
        "name": name,
        "type": "",
        "cost": "",
        "Color": [],
        "Cost": ""
    }


def merge_cards(existing_cards: dict, webp_urls: list[str], start_id: int | None = None) -> dict:
    output = dict(existing_cards)

    existing_names = {
        normalize_name(card.get("name", "")): card_id
        for card_id, card in output.items()
        if isinstance(card, dict)
    }

    numeric_ids = [
        int(card_id)
        for card_id in output.keys()
        if str(card_id).isdigit()
    ]

    next_id = start_id if start_id is not None else (max(numeric_ids) + 1 if numeric_ids else 1)

    added = 0
    skipped = 0

    for url in webp_urls:
        filename = Path(urlparse(url).path).name
        slug = filename.removesuffix(".webp")
        name = slug_to_card_name(slug)

        key = normalize_name(name)
        if key in existing_names:
            # Update missing image if the card already exists.
            existing_id = existing_names[key]
            card = output[existing_id]
            card.setdefault("face", {}).setdefault("front", {})
            if not card["face"]["front"].get("image"):
                card["face"]["front"]["image"] = url
            skipped += 1
            continue

        card_id = str(next_id)
        output[card_id] = webp_url_to_card(url, card_id)
        existing_names[key] = card_id
        next_id += 1
        added += 1

    print(f"Added {added} new cards.")
    print(f"Skipped/updated {skipped} existing cards.")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape .webp card image URLs from VibesTracker and merge them into a tcg-arena CardList.json file."
    )
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--input", default="CardList.json")
    parser.add_argument("--output", default="CardList.updated.json")
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="Optional first ID to use for newly added cards. Defaults to max existing numeric ID + 1."
    )
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Only print discovered .webp URLs. Do not update JSON."
    )

    args = parser.parse_args()

    webp_urls = scrape_webp_urls(args.page_url)

    if args.urls_only:
        for url in webp_urls:
            print(url)
        return

    input_path = Path(args.input)
    if input_path.exists():
        existing_cards = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        existing_cards = {}

    updated_cards = merge_cards(existing_cards, webp_urls, start_id=args.start_id)

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(updated_cards, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Found {len(webp_urls)} .webp URLs.")
    print(f"Wrote updated card list to {output_path}")


if __name__ == "__main__":
    main()
