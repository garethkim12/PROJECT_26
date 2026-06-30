"""
Streamlit 감성 분석 대시보드
실행: streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px

# ── 페이지 설정 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="한국어 감성 분석 대시보드",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 스타일 ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid;
    }
    .positive { border-color: #4CAF50; }
    .negative { border-color: #F44336; }
    .neutral  { border-color: #2196F3; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 유틸 함수 ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 모델 로딩 중... (최초 1회)")
def get_model():
    from sentiment_analyzer import load_model
    return load_model()


@st.cache_data(show_spinner="⚙️ 감성 분석 중...")
def run_analysis(df_json: str, text_col: str) -> pd.DataFrame:
    from sentiment_analyzer import analyze_dataframe
    df = pd.read_json(df_json, orient="records")
    return analyze_dataframe(df, text_col=text_col)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎭 감성 분석 대시보드")
    st.markdown("---")

    uploaded = st.file_uploader(
        "📂 CSV / Excel 파일 업로드",
        type=["csv", "xlsx", "xls"],
        help="text 컬럼이 포함된 파일을 업로드하세요.",
    )

    st.markdown("#### ⚙️ 분석 설정")
    text_col = st.text_input("텍스트 컬럼명", value="text")
    batch_size = st.slider("배치 사이즈", 8, 64, 32, step=8,
                           help="클수록 빠르지만 메모리 사용량 증가")
    sample_n = st.number_input(
        "샘플 수 (0 = 전체)", min_value=0, max_value=10000, value=0, step=100
    )

    st.markdown("---")
    st.caption("모델: snunlp/KR-FinBert-SC\n긍정 / 부정 / 중립 분류")


# ── 메인 ──────────────────────────────────────────────────────────
st.title("🎭 한국어 감성 분석 대시보드")

if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 CSV 또는 Excel 파일을 업로드하세요.")

    # 데모 데이터 안내
    with st.expander("📋 샘플 데이터로 테스트하기"):
        demo = pd.DataFrame({
            "text": [
                "이 제품 정말 최고예요! 강력 추천합니다.",
                "배송이 너무 늦고 품질도 별로였어요.",
                "그냥 보통이에요. 특별하지 않네요.",
                "가격 대비 훌륭한 성능이에요. 만족합니다.",
                "다시는 구매하지 않겠습니다. 실망이에요.",
                "포장이 꼼꼼하고 배송도 빠르네요.",
                "생각보다 크기가 작아서 아쉬워요.",
                "색상이 사진과 달라요. 환불 원합니다.",
                "친절한 고객 서비스에 감동받았어요.",
                "무난하게 쓸 수 있는 제품입니다.",
            ]
        })
        st.dataframe(demo, use_container_width=True)
        csv_bytes = df_to_csv_bytes(demo)
        st.download_button("⬇️ 샘플 CSV 다운로드", csv_bytes, "sample.csv", "text/csv")
    st.stop()


# ── 데이터 로드 ───────────────────────────────────────────────────
@st.cache_data
def load_file(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        # 인코딩 자동 감지
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="cp949")
    else:
        return pd.read_excel(file)


df_raw = load_file(uploaded)

# 컬럼 검증
if text_col not in df_raw.columns:
    st.error(f"❌ '{text_col}' 컬럼을 찾을 수 없습니다.\n\n사용 가능한 컬럼: {list(df_raw.columns)}")
    st.stop()

# 샘플링
if sample_n > 0 and sample_n < len(df_raw):
    df_raw = df_raw.sample(n=sample_n, random_state=42).reset_index(drop=True)
    st.warning(f"⚠️ {sample_n}건 샘플링 적용")

st.success(f"✅ {len(df_raw):,}건 로드 완료 | 컬럼: {list(df_raw.columns)}")

# ── 분석 실행 ─────────────────────────────────────────────────────
if st.button("🚀 감성 분석 시작", type="primary", use_container_width=True):
    st.session_state["df_result"] = None  # 캐시 초기화
    with st.spinner("분석 중..."):
        df_result = run_analysis(df_raw.to_json(orient="records"), text_col)
    st.session_state["df_result"] = df_result
    st.success("✅ 분석 완료!")

df_result = st.session_state.get("df_result", None)

if df_result is None:
    st.info("위의 **감성 분석 시작** 버튼을 눌러 분석을 진행하세요.")
    st.stop()


# ── 탭 구성 ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 개요",
    "☁️ 워드클라우드",
    "📈 시각화",
    "📋 데이터",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 : 개요
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 감성 분석 결과 개요")

    total = len(df_result)
    cnt = df_result["sentiment"].value_counts()
    pos = cnt.get("긍정", 0)
    neg = cnt.get("부정", 0)
    neu = cnt.get("중립", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 전체 건수", f"{total:,}")
    c2.metric("😊 긍정", f"{pos:,}", f"{pos/total*100:.1f}%")
    c3.metric("😞 부정", f"{neg:,}", f"{neg/total*100:.1f}%")
    c4.metric("😐 중립", f"{neu:,}", f"{neu/total*100:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        from visualizer import plot_sentiment_distribution
        st.plotly_chart(plot_sentiment_distribution(df_result), use_container_width=True)
    with col_b:
        from visualizer import plot_sentiment_pie
        st.plotly_chart(plot_sentiment_pie(df_result), use_container_width=True)

    # 평균 신뢰도
    st.markdown("#### 📌 평균 신뢰도 점수")
    avg_scores = df_result.groupby("sentiment")["sentiment_score"].mean().round(3)
    st.dataframe(avg_scores.reset_index().rename(columns={"sentiment_score": "평균 점수"}),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 : 워드클라우드
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("☁️ 워드클라우드")
    from visualizer import generate_all_wordclouds, fig_to_bytes

    wc_option = st.radio(
        "표시할 워드클라우드",
        ["전체", "긍정", "부정", "중립"],
        horizontal=True,
    )

    with st.spinner("워드클라우드 생성 중..."):
        wc_figs = generate_all_wordclouds(df_result, text_col)

    if wc_option in wc_figs:
        fig = wc_figs[wc_option]
        st.image(fig_to_bytes(fig), use_container_width=True)
        st.download_button(
            f"⬇️ {wc_option} 워드클라우드 저장",
            data=fig_to_bytes(fig),
            file_name=f"wordcloud_{wc_option}.png",
            mime="image/png",
        )
    else:
        st.warning(f"'{wc_option}' 감성의 데이터가 없습니다.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 : 시각화
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 상세 시각화")
    from visualizer import plot_score_histogram, plot_score_boxplot

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_score_histogram(df_result), use_container_width=True)
    with col2:
        st.plotly_chart(plot_score_boxplot(df_result), use_container_width=True)

    # 긍정 vs 부정 점수 산점도 (해당 컬럼 있을 때)
    if "score_positive" in df_result.columns and df_result["score_positive"].notna().any():
        st.markdown("#### 긍정 vs 부정 점수 산점도")
        fig_scatter = px.scatter(
            df_result,
            x="score_positive",
            y="score_negative",
            color="sentiment",
            color_discrete_map={"긍정": "#4CAF50", "부정": "#F44336", "중립": "#2196F3"},
            opacity=0.6,
            title="긍정 점수 vs 부정 점수",
            labels={
                "score_positive": "긍정 점수",
                "score_negative": "부정 점수",
                "sentiment": "감성",
            },
        )
        fig_scatter.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 : 데이터
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 분석 결과 데이터")

    # 필터
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_sent = st.multiselect(
            "감성 필터",
            ["긍정", "부정", "중립"],
            default=["긍정", "부정", "중립"],
        )
    with col_f2:
        min_score = st.slider("최소 신뢰도 점수", 0.0, 1.0, 0.0, 0.05)

    filtered = df_result[
        df_result["sentiment"].isin(filter_sent)
        & (df_result["sentiment_score"] >= min_score)
    ]

    st.caption(f"표시 건수: {len(filtered):,} / {len(df_result):,}")
    st.dataframe(
        filtered[[text_col, "sentiment", "sentiment_score"]].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    # 다운로드
    st.download_button(
        "⬇️ 결과 CSV 다운로드 (UTF-8)",
        data=df_to_csv_bytes(filtered),
        file_name="sentiment_result.csv",
        mime="text/csv",
    )
