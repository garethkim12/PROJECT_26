"""
전처리 모듈 — 분석설계서 v1.1 기준 (2단계 전처리 규칙 확정안)

처리 규칙 요약:
  0. 빈 행/완전 중복 행 제거 (엑셀에서 셀 내용만 지운 경우 행이 남는 문제 처리)
  1. rating       : >5 인 값은 NaN 처리. 원본은 rating_raw 보존. platform 컬럼 신설.
  2. date         : platform 추출(Google/Trip.com/Tripadvisor/기타).
                    'N개월/년/주/일 전' 패턴을 기준일(CRAWL_DATE) 기준으로
                    review_month(YYYY-MM) 컬럼으로 환산. '수정일:' 접두사 제거 후 동일 처리.
  3. text         : 결측 제외. 리터럴 줄바꿈(\n) → 공백. URL/이메일 제거.
                    5자 이하는 제외하지 않고 is_short_text 플래그만 부여.
                    말줄임표(…)로 끝나는 절단 텍스트는 has_truncated_text 플래그로 보존.
  4. reviewer     : 분석에서 제외하되, 재식별 방지를 위해 reviewer_id로 해시화하여 보존.
  5. 중복         : (place_name, reviewer, text) 완전 일치 시 제거.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

# ── 기준일 (크롤링 시작일) ──────────────────────────────────────
CRAWL_DATE = date(2026, 6, 12)

PLATFORMS = ["Google", "Trip.com", "Tripadvisor"]

# 'N개월 전', 'N년 전', 'N주 전', 'N일 전', 'N시간 전', 'N분 전' 패턴
RELATIVE_TIME_RE = re.compile(r"(\d+)\s*(년|개월|주|일|시간|분)\s*전")


def extract_platform(date_str: str) -> str:
    """
    date 텍스트에서 출처 플랫폼 추출.
    Google/Trip.com/Tripadvisor 외(IHG, Accor 등 호텔 체인 자체 리뷰 플랫폼 등)는
    '기타'로 묶는다. rating 척도는 1~5로 정상 범위이므로 분석에 지장 없음.
    """
    if not isinstance(date_str, str):
        return "기타"
    for p in PLATFORMS:
        if p in date_str:
            return p
    return "기타"


def relative_to_month(date_str: str, base: date = CRAWL_DATE) -> str | None:
    """
    'N개월 전 Google에 게시됨' / '수정일: N년 전 ...' 형태의 상대시간을
    기준일(base) 대비 절대 연-월(YYYY-MM) 문자열로 환산.
    매칭 실패 시 None.
    """
    if not isinstance(date_str, str):
        return None
    # '수정일:' 등의 접두사는 제거하고 동일 패턴 탐색
    cleaned = date_str.replace("수정일:", "").strip()
    m = RELATIVE_TIME_RE.search(cleaned)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)

    if unit == "년":
        target = base - relativedelta(years=n)
    elif unit == "개월":
        target = base - relativedelta(months=n)
    elif unit == "주":
        target = base - relativedelta(weeks=n)
    elif unit in ("일", "시간", "분"):
        target = base  # 같은 달로 처리
    else:
        return None

    return f"{target.year:04d}-{target.month:02d}"


def hash_reviewer(name: str) -> str:
    """reviewer를 재식별 불가능한 해시 ID로 변환"""
    if not isinstance(name, str) or not name.strip():
        name = "unknown"
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def clean_text(text: str) -> str:
    """텍스트 정제: 리터럴 줄바꿈/URL/이메일 제거, 공백 정리"""
    if not isinstance(text, str):
        return ""
    t = text.replace("\\n", " ").replace("\n", " ")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\S+@\S+\.\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 크롤링 DataFrame(category, place_name, reviewer, rating, date, text)을
    분석설계서 규칙에 따라 전처리하여 반환.
    """
    df = df.copy()
    before_n = len(df)

    # ── 0. 빈 행 / 완전 중복 행 제거 ─────────────────────────────
    # 엑셀에서 "셀 내용 지우기"로 처리한 경우, 행 자체는 안 지워지고
    # 모든 컬럼이 NaN인 빈 행이 남는다. 이를 먼저 제거한다.
    blank_mask = df.isna().all(axis=1)
    n_blank = int(blank_mask.sum())
    df = df[~blank_mask].copy()

    full_dup_mask = df.duplicated()
    n_full_dup = int(full_dup_mask.sum())
    df = df[~full_dup_mask].copy()

    # ── 1. platform 추출 + rating 정규화 ────────────────────────
    df["platform"] = df["date"].apply(extract_platform)
    df["rating_raw"] = df["rating"]
    df["rating"] = df["rating"].where(df["rating"] <= 5, np.nan)

    # ── 2. date → review_month 환산 ─────────────────────────────
    df["review_month"] = df["date"].apply(relative_to_month)

    # ── 3. text 정제 ─────────────────────────────────────────────
    df["has_truncated_text"] = (
        df["text"].fillna("").astype(str).str.strip().str.endswith("…")
    )
    df["text_clean"] = df["text"].apply(clean_text)
    df["is_short_text"] = df["text_clean"].str.len() <= 5

    # 결측 텍스트(빈 문자열) 제외
    missing_mask = df["text_clean"].str.len() == 0
    n_missing = int(missing_mask.sum())
    df = df[~missing_mask].copy()

    # ── 4. reviewer 해시화 ───────────────────────────────────────
    df["reviewer_id"] = df["reviewer"].apply(hash_reviewer)
    df = df.drop(columns=["reviewer"])

    # ── 5. 중복 제거 (place_name, reviewer_id, text_clean) ───────
    dup_mask = df.duplicated(subset=["place_name", "reviewer_id", "text_clean"])
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    after_n = len(df)

    # 최종 컬럼 순서 정리
    cols = [
        "category", "place_name", "reviewer_id",
        "rating", "rating_raw", "platform",
        "date", "review_month",
        "text_clean", "is_short_text", "has_truncated_text",
    ]
    df = df[cols].rename(columns={"text_clean": "text"})

    report = {
        "원본 건수": before_n,
        "빈 행 제거": n_blank,
        "완전 중복 행 제거": n_full_dup,
        "결측 텍스트 제외": n_missing,
        "중복 제거(부분키 기준)": n_dup,
        "최종 건수": after_n,
        "rating 결측 처리(>5)": int(df["rating"].isna().sum()) if "rating" in df else 0,
    }
    preprocess.last_report = report  # 간단한 리포트 보관
    return df


# ── 단독 실행 시: 샘플 데이터로 검증 ────────────────────────────
if __name__ == "__main__":
    raw = pd.read_excel("sample_raw.xlsx", sheet_name="크롤링 셈플")
    cleaned = preprocess(raw)

    print("=== 전처리 리포트 ===")
    for k, v in preprocess.last_report.items():
        print(f"{k}: {v}")

    print("\n=== 결과 미리보기 ===")
    print(cleaned.head(5).to_string())

    print("\n=== platform별 건수 ===")
    print(cleaned["platform"].value_counts())

    print("\n=== review_month 범위 ===")
    print(cleaned["review_month"].min(), "~", cleaned["review_month"].max())
    print("review_month 결측:", cleaned["review_month"].isna().sum())

    cleaned.to_csv("sample_cleaned.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: sample_cleaned.csv")
