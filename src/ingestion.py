import asyncio
import aiohttp
import json
import os
import re
import sys
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


BASE_URL = "https://www.shl.com"
START_URL = "https://www.shl.com/solutions/products/product-catalog/?type=1"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_catalog.json")

MAX_CONCURRENT_REQUESTS = 10
RETRY_LIMIT = 3


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    noise = [
        r"Home\s+Products.*?",
        r"Product Fact Sheet.*",
        r"English \(USA\)",
        r"Test Type:.*"
    ]
    for p in noise:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def get_headers():
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    ]
    return {"User-Agent": random.choice(agents)}


async def fetch_url(session, url: str) -> str:
    for _ in range(RETRY_LIMIT):
        try:
            async with session.get(url, headers=get_headers(), timeout=15) as response:
                if response.status == 200:
                    return await response.text()
                if response.status == 404:
                    return ""
        except Exception:
            await asyncio.sleep(1)
    return ""


async def parse_test_details(session, url: str, semaphore) -> dict:
    async with semaphore:
        html = await fetch_url(session, url)
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        duration = 45
        match = re.search(r"Approximate Completion Time.*?(\d+)", text, re.IGNORECASE)
        if match:
            duration = int(match.group(1))

        job_level = "All Levels"
        match = re.search(r"Job levels\s+(.*?)(?:Individual|Languages|$)", text, re.IGNORECASE)
        if match:
            job_level = clean_text(match.group(1))

        desc = ""
        desc_div = soup.select_one(".product-detail__content")
        if desc_div:
            desc = clean_text(desc_div.get_text())
        else:
            match = re.search(r"Description(.*?)(?:Job levels|Product Fact)", text, re.IGNORECASE)
            if match:
                desc = clean_text(match.group(1))

        return {
            "description": desc[:1000],
            "job_levels": job_level,
            "duration": duration,
        }


async def scrape_catalog():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_tests = []
    page_url = START_URL
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        test_links = []

        while page_url:
            html = await fetch_url(session, page_url)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            rows = soup.select("table tr")

            for tr in rows:
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue

                link = tds[0].find("a")
                if not link:
                    continue

                test_links.append({
                    "name": clean_text(link.get_text()),
                    "url": urljoin(BASE_URL, link["href"]),
                    "remote_support": "Yes" if tds[1].select_one(".-yes") else "No",
                    "adaptive_support": "Yes" if tds[2].select_one(".-yes") else "No",
                    "test_type": [
                        t.get_text(strip=True)
                        for t in tds[3].select(".product-catalogue__key")
                    ],
                })

            next_btn = soup.select_one(".pagination .next a")
            page_url = urljoin(page_url, next_btn["href"]) if next_btn else None

        tasks = [
            parse_test_details(session, t["url"], semaphore)
            for t in test_links
        ]

        details = []
        for f in tqdm.as_completed(tasks, total=len(tasks)):
            details.append(await f)

        for base, detail in zip(test_links, details):
            base.update(detail)
            all_tests.append(base)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_tests, f, indent=2)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(scrape_catalog())
