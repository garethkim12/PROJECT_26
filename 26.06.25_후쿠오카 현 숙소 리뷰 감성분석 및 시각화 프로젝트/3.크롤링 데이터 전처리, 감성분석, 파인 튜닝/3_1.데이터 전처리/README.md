# 🎭 한국어 감성 분석 대시보드

KR-FinBert-SC 기반 한국어 감성 분석 + Streamlit 시각화 프로젝트

---

## 📁 파일 구조

```
sentiment_project/
├── app.py                  # Streamlit 메인 앱
├── sentiment_analyzer.py   # 감성 분석 모듈 (KR-FinBert-SC)
├── visualizer.py           # 워드클라우드 & 차트 모듈
├── requirements.txt        # 의존성 패키지
└── README.md
```

---

## ⚡ 빠른 시작

### 1. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

> **한글 형태소 분석기(KoNLPy) 설치** (워드클라우드 품질 향상)
> ```bash
> # Ubuntu/macOS
> pip install konlpy
>
> # Windows: Java 설치 후
> pip install konlpy
> ```
> KoNLPy 없이도 기본 명사 추출로 동작합니다.

### 3. 한글 폰트 설치 (Ubuntu)
```bash
sudo apt-get install -y fonts-nanum
fc-cache -fv
```

### 4. Streamlit 앱 실행
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 📊 주요 기능

| 기능 | 설명 |
|------|------|
| **감성 분석** | KR-FinBert-SC로 긍정/부정/중립 자동 분류 |
| **워드클라우드** | 전체 & 감성별 핵심 단어 시각화 |
| **히스토그램** | 신뢰도 점수 분포 확인 |
| **파이/바 차트** | 감성 비율 인터랙티브 차트 |
| **박스플롯** | 감성별 점수 분포 비교 |
| **산점도** | 긍정 vs 부정 점수 관계 |
| **결과 다운로드** | 분석 결과 CSV 저장 |

---

## 📝 CSV 파일 형식

최소한 `text` 컬럼이 필요합니다. 다른 컬럼이 있어도 무관합니다.

```csv
text,date,product
"이 제품 정말 최고예요!",2024-01-01,A
"배송이 너무 늦었어요.",2024-01-02,B
```

컬럼명이 다를 경우 사이드바에서 변경 가능합니다.

---

## ⚙️ 모델 정보

- **모델**: `snunlp/KR-FinBert-SC`
- **분류**: 긍정(positive) / 부정(negative) / 중립(neutral)
- **최초 실행 시** HuggingFace에서 모델 자동 다운로드 (~400MB)
- GPU 있을 경우 자동으로 CUDA 사용

---

## 🔧 자주 묻는 질문

**Q. 처음 실행이 느려요.**
A. 최초 1회 모델 다운로드(~400MB)가 필요합니다. 이후 캐시 사용.

**Q. 워드클라우드에 한글이 깨져요.**
A. 한글 폰트를 설치하세요. Ubuntu: `sudo apt-get install fonts-nanum`

**Q. 분석 속도가 느려요.**
A. GPU 환경 권장. CPU 사용 시 샘플링(1,000건 이하)을 권장합니다.

**Q. 컬럼명이 text가 아니에요.**
A. 사이드바의 "텍스트 컬럼명" 입력란에서 변경하세요.
