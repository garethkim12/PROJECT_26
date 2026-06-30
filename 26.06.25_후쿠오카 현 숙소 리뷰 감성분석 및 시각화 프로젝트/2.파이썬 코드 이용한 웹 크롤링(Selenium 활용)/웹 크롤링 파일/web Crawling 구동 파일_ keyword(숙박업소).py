# -*- coding: utf-8 -*-
"""
구글 맵스 리뷰 크롤러 (후쿠오카 현 텍스트데이터분석 조별과제용)

담당 주제별 사용 예 (목표: 1인당 10만 건):
  # 김동건 — 관광 (관광지/쇼핑/오락/온천)
  python main.py --topic 관광 --max-places 40 --max-reviews 5000

  # 김근하 — 숙박 (호텔/료칸/게스트하우스)
  python main.py --topic 숙박 --max-places 80 --max-reviews 5000

  # 남종희 — 음식점 (맛집/라멘/이자카야/카페)
  python main.py --topic 음식점 --max-places 80 --max-reviews 5000

  # 주제 안의 카테고리 하나만
  python main.py --topic 관광 --category 온천

  # 프리셋에 없는 검색어로 직접 수집
  python main.py --keyword "후쿠오카 현 야타이" --max-places 20

중간 저장/이어하기:
  - 장소 하나가 끝날 때마다 output/reviews_<주제>.csv 에 바로 추가 저장됩니다.
  - 진행 상황은 output/progress_<주제>.json 에 기록되어, 같은 명령을 다시
    실행하면 이미 수집한 장소는 건너뛰고 이어서 수집합니다.
  - 처음부터 다시 하려면 --restart 를 붙이세요 (기존 CSV는 지우지 않음).

컬럼: category, place_name, reviewer, rating, date, text
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# 주제별 검색 키워드 프리셋 — 담당 주제에 맞게 자유롭게 수정/추가하세요
REGION = "후쿠오카 현"
TOPICS = {
    "관광": {
        "관광": f"{REGION} 관광지",
        "쇼핑": f"{REGION} 쇼핑",
        "오락": f"{REGION} 오락",
        "온천": f"{REGION} 온천",
    },
    "숙박": {
        "호텔": f"{REGION} 호텔",
        "료칸": f"{REGION} 료칸",
        "게스트하우스": f"{REGION} 게스트하우스",
    },
    "음식점": {
        "맛집": f"{REGION} 맛집",
        "라멘": f"{REGION} 라멘",
        "이자카야": f"{REGION} 이자카야",
        "카페": f"{REGION} 카페",
    },
}

OUTPUT_DIR = Path(__file__).parent / "output"
COLUMNS = ["category", "place_name", "reviewer", "rating", "date", "text"]

# 로드된 리뷰 카드를 브라우저 안에서 한 번에 파싱 (Selenium 개별 조회보다 수십 배 빠름)
JS_EXTRACT = """
const start = arguments[0];
const cards = document.querySelectorAll('div.jftiEf');
const out = [];
for (let i = start; i < cards.length; i++) {
  const c = cards[i];
  const q = s => { const e = c.querySelector(s); return e ? e.textContent.trim() : ''; };
  let rating = null;
  const star = c.querySelector('span.kvMYJc');
  if (star) {
    const m = (star.getAttribute('aria-label') || '').match(/\\d+/);
    if (m) rating = parseInt(m[0]);
  } else {
    const f = c.querySelector('span.fzvQIb');
    if (f) { const m = f.textContent.match(/(\\d+)\\s*\\/\\s*5/); if (m) rating = parseInt(m[1]); }
  }
  let date = q('span.rsqaWe') || q('span.xRkPPb');
  date = date.split('\\n')[0].trim();
  out.push({
    id: c.getAttribute('data-review-id') || '',
    reviewer: q('div.d4r55'),
    rating: rating,
    date: date,
    text: q('span.wiI7pd') || q('div.OA1nbd'),
  });
}
return out;
"""

JS_EXPAND = "document.querySelectorAll('button.w8nwRe').forEach(b => b.click());"


def make_driver(headless: bool = False) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,950")
    opts.add_argument("--lang=ko-KR")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(3)
    return driver


def collect_place_links(driver, keyword: str, max_places: int) -> list[str]:
    """검색 결과 패널을 스크롤하며 장소 상세 페이지 링크를 수집."""
    url = f"https://www.google.com/maps/search/{keyword}?hl=ko"
    driver.get(url)
    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.hfpxzc")))
    except TimeoutException:
        if "/maps/place/" in driver.current_url:
            return [driver.current_url]
        print(f"  [경고] '{keyword}' 검색 결과를 찾지 못했습니다.")
        return []

    try:
        panel = driver.find_element(By.CSS_SELECTOR, "div[role='feed']")
    except NoSuchElementException:
        panel = None

    links: list[str] = []
    seen = set()
    stagnant = 0
    while len(links) < max_places and stagnant < 5:
        cards = driver.find_elements(By.CSS_SELECTOR, "a.hfpxzc")
        before = len(links)
        for a in cards:
            href = a.get_attribute("href")
            if href and href not in seen:
                seen.add(href)
                links.append(href)
                if len(links) >= max_places:
                    break
        stagnant = stagnant + 1 if len(links) == before else 0
        if panel is not None and len(links) < max_places:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
            time.sleep(1.5)
        else:
            break
    print(f"  장소 {len(links)}곳 수집 완료")
    return links[:max_places]


def open_reviews_tab(driver) -> str | None:
    """'리뷰' 탭을 클릭하고 aria-label('<장소명> 리뷰')에서 장소명을 추출.

    실패하면 None, 성공하면 장소명(없으면 빈 문자열)을 반환.
    """
    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
    except TimeoutException:
        return None
    deadline = time.time() + 20
    while time.time() < deadline:
        for tab in driver.find_elements(By.CSS_SELECTOR, "button[role='tab']"):
            label = (tab.get_attribute("aria-label") or "") or (tab.text or "")
            if "리뷰" in label or "Reviews" in label:
                name = re.sub(r"\s*(리뷰|Reviews)\s*$", "", label).strip()
                driver.execute_script("arguments[0].click()", tab)
                time.sleep(2)
                if not name:
                    name = get_place_name(driver)
                return name
        time.sleep(1.5)
    return None


def get_place_name(driver) -> str:
    # 상세 패널의 장소명 (검색 결과 패널의 '검색 결과' h1과 구분)
    for sel in ("h1.DUwDvf", "h1"):
        try:
            name = driver.find_element(By.CSS_SELECTOR, sel).text.strip()
            if name and name != "검색 결과":
                return name
        except NoSuchElementException:
            continue
    return ""


def scrape_reviews(driver, place_url: str, category: str, max_reviews: int) -> list[dict]:
    # 구글이 로드마다 레이아웃을 달리 주는 경우가 있어 리뷰 탭이 없으면 재로드
    place_name = None
    for attempt in range(3):
        driver.get(place_url)
        place_name = open_reviews_tab(driver)
        if place_name is not None:
            break
        time.sleep(2)
    if place_name is None:
        print("  [경고] 리뷰 탭을 찾지 못했습니다:", place_url[:80])
        return []
    print(f"  ▶ {place_name}", flush=True)

    # 리뷰 목록 스크롤 영역
    try:
        panel = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'m6QErb') and contains(@class,'DxyBCb') and contains(@class,'dS8AEf')]",
        )
    except NoSuchElementException:
        panel = None

    rows: list[dict] = []
    seen_ids: set[str] = set()
    processed = 0  # 이미 파싱한 카드 수 — 새로 로드된 카드만 파싱해서 속도 유지
    stagnant = 0
    last_log = time.time()
    while len(rows) < max_reviews and stagnant < 8:
        driver.execute_script(JS_EXPAND)  # '자세히' 펼치기
        time.sleep(0.3)
        new_cards = driver.execute_script(JS_EXTRACT, processed)
        for c in new_cards:
            processed += 1
            rid = c.get("id") or ""
            if rid and rid in seen_ids:
                continue
            seen_ids.add(rid)
            if not (c.get("reviewer") or c.get("text")):
                continue
            rows.append({
                "category": category, "place_name": place_name,
                "reviewer": c.get("reviewer", ""), "rating": c.get("rating"),
                "date": c.get("date", ""), "text": c.get("text", ""),
            })
            if len(rows) >= max_reviews:
                break
        stagnant = stagnant + 1 if not new_cards else 0
        if time.time() - last_log > 30:
            print(f"    ... {len(rows)}건", flush=True)
            last_log = time.time()
        if panel is not None and len(rows) < max_reviews:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
            time.sleep(1.2)
        elif panel is None:
            break
    print(f"    리뷰 {len(rows)}건 수집", flush=True)
    return rows


def append_csv(rows: list[dict], path: Path):
    if not rows:
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_progress(path: Path, progress: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=1), encoding="utf-8")


def crawl_keywords(driver, keywords: dict[str, str], max_places: int, max_reviews: int,
                   csv_path: Path, progress_path: Path, restart: bool) -> int:
    progress = {} if restart else load_progress(progress_path)
    total = sum(progress.values())
    if total:
        print(f"[이어하기] 기존 수집 {total}건 — 완료한 장소 {len(progress)}곳은 건너뜁니다.")
    for cat, kw in keywords.items():
        print(f"\n=== [{cat}] '{kw}' 검색 ===", flush=True)
        links = collect_place_links(driver, kw, max_places)
        for link in links:
            key = link.split("?")[0]  # 같은 장소가 다른 쿼리스트링으로 나와도 한 번만
            if key in progress:
                continue
            try:
                rows = scrape_reviews(driver, link, cat, max_reviews)
            except Exception as e:
                print(f"  [오류] 장소 수집 실패: {e}")
                continue
            append_csv(rows, csv_path)  # 장소 단위로 즉시 저장
            progress[key] = len(rows)
            save_progress(progress_path, progress)
            total += len(rows)
            print(f"    (누적 {total}건 → {csv_path.name})", flush=True)
            time.sleep(1)
    return total


def main():
    ap = argparse.ArgumentParser(description="구글 맵스 리뷰 크롤러 (후쿠오카 현 조별과제)")
    ap.add_argument("--topic", choices=list(TOPICS), help="담당 주제: " + ", ".join(TOPICS))
    ap.add_argument("--category", help="주제 안의 카테고리 하나만 (예: --topic 관광 --category 온천)")
    ap.add_argument("--keyword", help="프리셋 대신 직접 검색어 지정 (예: '후쿠오카 현 야타이')")
    ap.add_argument("--url", help="단일 장소 URL")
    ap.add_argument("--max-places", type=int, default=40, help="카테고리당 장소 수 (기본 40)")
    ap.add_argument("--max-reviews", type=int, default=5000, help="장소당 리뷰 수 (기본 5000)")
    ap.add_argument("--restart", action="store_true", help="이어하기 무시하고 처음부터")
    ap.add_argument("--headless", action="store_true", help="브라우저 창 숨김")
    args = ap.parse_args()

    if not (args.topic or args.keyword or args.url):
        ap.error("--topic, --keyword, --url 중 하나는 필요합니다.")

    if args.url:
        tag = "single"
        keywords = None
    elif args.keyword:
        tag = args.keyword.replace(" ", "_")
        keywords = {args.keyword: args.keyword}
    else:
        keywords = TOPICS[args.topic]
        if args.category:
            if args.category not in keywords:
                ap.error(f"'{args.topic}' 주제에 '{args.category}' 카테고리가 없습니다. "
                         f"가능: {', '.join(keywords)}")
            keywords = {args.category: keywords[args.category]}
        tag = args.topic  # 카테고리를 나눠 돌려도 같은 주제면 같은 CSV에 누적

    csv_path = OUTPUT_DIR / f"reviews_{tag}.csv"
    progress_path = OUTPUT_DIR / f"progress_{tag}.json"

    driver = make_driver(headless=args.headless)
    try:
        if args.url:
            rows = scrape_reviews(driver, args.url, args.category or "단일장소", args.max_reviews)
            append_csv(rows, csv_path)
            total = len(rows)
        else:
            total = crawl_keywords(driver, keywords, args.max_places, args.max_reviews,
                                   csv_path, progress_path, args.restart)
    finally:
        driver.quit()

    print(f"\n저장 위치: {csv_path}  (이번 실행까지 누적 {total}건)")


if __name__ == "__main__":
    main()
