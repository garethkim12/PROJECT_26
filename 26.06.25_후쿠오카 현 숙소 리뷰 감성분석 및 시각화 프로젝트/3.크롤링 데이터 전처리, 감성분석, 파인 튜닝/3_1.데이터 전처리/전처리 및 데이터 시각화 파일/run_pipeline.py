"""
후쿠오카 숙박업소 리뷰 — 전체 파이프라인 실행 스크립트
(전처리 → 감성분석 → 결과 CSV 저장)

사용자 실행용 스크립트입니다. 로컬 PC 또는 Google Colab에서 실행하세요.

[모델 변경 안내]
1차로 사용한 snunlp/KR-FinBert-SC 는 금융 도메인 모델이라 호텔 리뷰에서
98.9%가 "중립"으로 잘못 분류되어 폐기했습니다. validate_model.py로 검증한 결과
alsgyu/sentiment-analysis-fine-tuned-model 로 교체했습니다.
(라벨 의미가 모델 카드에 명시되어 있지 않아, 직접 16개 예문으로 검증함:
 LABEL_0=부정, LABEL_1=중립, LABEL_2=긍정. 긍정/부정 판별 정확도 9/10,
 중립 판별은 다소 불안정 — 보고서에 한계로 명시 권장)

[실행 방법 — 이미 전처리된 파일(full_cleaned.csv)을 쓰는 경우] (기본값)
1) 같은 폴더에 preprocess.py 를 둡니다 (SKIP_PREPROCESS=True면 실제로 호출되지 않지만,
   import 에러를 피하기 위해 같은 폴더에 두는 것을 권장합니다).
2) Claude가 전처리해서 전달한 full_cleaned.csv 를 같은 폴더에 둡니다.
3) INPUT_PATH = "full_cleaned.csv", SKIP_PREPROCESS = True (기본값 그대로) 확인.
4) 터미널에서: python run_pipeline.py

[실행 방법 — 원본 파일(전처리 전)을 바로 쓰는 경우]
1) INPUT_PATH 를 원본 파일명으로 변경
2) SKIP_PREPROCESS = False 로 변경
3) 터미널에서: python run_pipeline.py

[필요 패키지]
pip install pandas openpyxl transformers torch python-dateutil tqdm --break-system-packages
(Colab이면 --break-system-packages 없이 그냥 pip install 만 하면 됩니다)

[Windows에서 OpenMP 충돌 경고가 뜨는 경우]
실행 전에 터미널에서: $env:KMP_DUPLICATE_LIB_OK="TRUE"

[GPU 권장]
8만 건을 CPU로 돌리면 수 시간 이상 걸릴 수 있습니다.
Google Colab 사용 시 메뉴 > 런타임 > 런타임 유형 변경 > GPU(T4) 선택을 권장합니다.
"""

import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# ── 사용자가 수정할 부분 ────────────────────────────────────────
INPUT_PATH = "full_cleaned.csv"      # Claude가 전처리한 파일 (기본값)
SHEET_NAME = 0                       # 엑셀이면 시트명/인덱스, CSV면 무시됨
OUTPUT_PATH = "final_result.csv"     # 최종 결과 저장 파일명
BATCH_SIZE = 32                      # GPU 메모리 부족하면 16으로 낮추세요
MODEL_NAME = "alsgyu/sentiment-analysis-fine-tuned-model"
SKIP_PREPROCESS = True               # True: 이미 정제된 파일이라 전처리 건너뜀
                                      # False: 원본 파일이라 preprocess.py 규칙 적용
# ───────────────────────────────────────────────────────────────

# validate_model.py 로 직접 검증해 확정한 라벨 매핑 (모델 카드에 명시 안 되어 있었음)
LABEL_MAP = {"LABEL_0": "부정", "LABEL_1": "중립", "LABEL_2": "긍정"}
POSITIVE_KEY, NEGATIVE_KEY = "LABEL_2", "LABEL_0"


def load_raw(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp949")
    return pd.read_excel(path, sheet_name=SHEET_NAME)


def run_sentiment(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    print(f"[모델 로드] {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[디바이스] {device}  (cpu일 경우 8만 건은 수 시간 걸릴 수 있습니다)")
    print(f"[라벨 매핑] {LABEL_MAP}")

    id2label = model.config.id2label
    # id2label 예: {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}
    # LABEL_0=부정, LABEL_1=중립, LABEL_2=긍정 (validate_model.py 검증 결과)
    label_keys = [id2label[i] for i in range(len(id2label))]

    sentiments, scores, score_pos, score_neg = [], [], [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="감성 분석 중"):
        batch = texts[i:i + batch_size]
        batch = [t if isinstance(t, str) and t.strip() else "내용 없음" for t in batch]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for prob in probs:
            idx = int(np.argmax(prob))
            label_key = id2label[idx]
            sentiments.append(LABEL_MAP.get(label_key, label_key))
            scores.append(float(prob[idx]))
            # 긍정/부정 점수는 검증된 키로 직접 참조
            pos_idx = label_keys.index(POSITIVE_KEY) if POSITIVE_KEY in label_keys else None
            neg_idx = label_keys.index(NEGATIVE_KEY) if NEGATIVE_KEY in label_keys else None
            score_pos.append(float(prob[pos_idx]) if pos_idx is not None else None)
            score_neg.append(float(prob[neg_idx]) if neg_idx is not None else None)

    return pd.DataFrame({
        "sentiment": sentiments,
        "sentiment_score": scores,
        "score_positive": score_pos,
        "score_negative": score_neg,
    })


def main():
    t0 = time.time()

    print("=" * 60)
    print("1/2 단계: 데이터 로드")
    print("=" * 60)
    raw = load_raw(INPUT_PATH)
    print(f"로드 건수: {len(raw):,}")

    if SKIP_PREPROCESS:
        print("\n[안내] SKIP_PREPROCESS=True — 이미 전처리된 파일로 간주하고 전처리를 건너뜁니다.")
        cleaned = raw
        if "text" not in cleaned.columns:
            raise ValueError(
                "'text' 컬럼이 없습니다. SKIP_PREPROCESS=True는 Claude가 전처리한 "
                "full_cleaned.csv 처럼 이미 text 컬럼이 정리된 파일에만 사용하세요."
            )
    else:
        print("\n" + "=" * 60)
        print("1.5/2 단계: 전처리 (preprocess.py 규칙 적용)")
        print("=" * 60)
        from preprocess import preprocess  # 원본 파일일 때만 import
        cleaned = preprocess(raw)
        for k, v in preprocess.last_report.items():
            print(f"  {k}: {v}")
    print(f"감성분석 대상 건수: {len(cleaned):,}")

    print("\n" + "=" * 60)
    print("2/2 단계: 감성분석 (alsgyu/sentiment-analysis-fine-tuned-model)")
    print("=" * 60)
    sentiment_df = run_sentiment(cleaned["text"].tolist(), batch_size=BATCH_SIZE)
    final = pd.concat([cleaned.reset_index(drop=True), sentiment_df], axis=1)

    final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"완료! 저장 위치: {OUTPUT_PATH}")
    print(f"총 소요 시간: {elapsed/60:.1f}분")
    print("=" * 60)
    print("\n감성 분포:")
    print(final["sentiment"].value_counts())


if __name__ == "__main__":
    main()
