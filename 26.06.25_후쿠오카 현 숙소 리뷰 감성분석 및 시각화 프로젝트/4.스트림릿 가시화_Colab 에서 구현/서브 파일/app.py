"""
후쿠오카 숙박업소 리뷰 감성분석 대시보드 — 최종판 (v3 기준)
실행: streamlit run app.py
사이드바에서 final_result_v3.csv 업로드
"""

import re, io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="후쿠오카 숙박업소 리뷰 분석",
    page_icon="🏨", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#f7f8fa; }
.stTabs [data-baseweb="tab"] { font-size:15px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── 색상 ───────────────────────────────────────────────────────
C_POS, C_NEG, C_BLUE = "#1D9E75", "#E24B4A", "#378ADD"
COLOR_MAP = {"긍정": C_POS, "부정": C_NEG}

# ── 한글 폰트 ──────────────────────────────────────────────────
def _get_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    for f in fm.findSystemFonts():
        if any(k in f.lower() for k in ["nanum","malgun","gothic","noto"]):
            return f
    # Colab 환경이면 나눔폰트 자동 설치
    try:
        import subprocess
        subprocess.run(["apt-get","install","-y","-q","fonts-nanum"],
                       check=True, capture_output=True)
        fm._load_fontmanager(try_read_cache=False)
        nanum = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if Path(nanum).exists(): return nanum
    except: pass
    return None

FONT_PATH = _get_font()

# ── 유틸 ───────────────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    try:    df = pd.read_csv(file, encoding="utf-8-sig")
    except: df = pd.read_csv(file, encoding="cp949")
    df["year"] = df["review_month"].str[:4]
    df["text_len"] = df["text"].fillna("").astype(str).apply(len)
    return df

def csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def html_metric(label, value, delta="", color=None):
    dh = ""
    if delta:
        dc = color or ("#1D9E75" if not delta.startswith("-") else "#E24B4A")
        dh = f'<p style="font-size:13px;color:{dc};margin:4px 0 0">{delta}</p>'
    st.markdown(f"""
    <div style="background:#f7f8fa;border-radius:8px;padding:14px 16px;min-height:80px">
      <p style="font-size:13px;color:#666;margin:0 0 4px">{label}</p>
      <p style="font-size:22px;font-weight:600;margin:0">{value}</p>
      {dh}
    </div>""", unsafe_allow_html=True)

def make_wordcloud(texts, color):
    stopwords = {
        "이","가","을","를","은","는","에","의","도","로","으로","와","과","한",
        "나","만","에서","하고","하다","있다","없다","되다","이다","것","수","등",
        "및","더","또","그","저","제","때","아","어","네","요","거","좀","잘","다",
        "그냥","진짜","정말","너무","매우","같아요","같았어요","습니다","했습니다",
        "있습니다","이었습니다","였습니다","었습니다","이나","나","도","만","위",
    }
    joined = " ".join(
        w for t in texts
        for w in re.sub(r"[^가-힣a-zA-Z\s]", " ", str(t)).split()
        if len(w) >= 2 and w not in stopwords
    )
    if not joined.strip():
        return None
    wc_kw = dict(width=900, height=420, background_color="white",
                 max_words=150, prefer_horizontal=0.85, collocations=False)
    if FONT_PATH:
        wc_kw["font_path"] = FONT_PATH
    wc = WordCloud(**wc_kw).generate(joined)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", ["#cccccc", color])
    wc.recolor(colormap=cmap)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🏨 후쿠오카 리뷰 분석")
    st.markdown("---")
    uploaded = st.file_uploader(
        "📂 final_result_v3.csv 업로드",
        type=["csv"], help="v3 파인튜닝 모델 감성분석 결과 파일"
    )
    st.markdown("---")
    st.caption(
        "데이터: Google Maps 크롤링\n"
        "모델: alsgyu + 의사레이블 파인튜닝 (v3)\n"
        "Macro F1: 0.8691 (+17.5%)\n"
        "분류: 긍정 / 부정 (2분류)"
    )

# ── 메인 ──────────────────────────────────────────────────────
st.title("🏨 후쿠오카 현 숙박업소 리뷰 감성분석 대시보드")

if uploaded is None:
    st.info("👈 사이드바에서 **final_result_v3.csv** 파일을 업로드하세요.")
    st.stop()

df = load_data(uploaded)
st.success(f"✅ {len(df):,}건 로드 완료 (최종 파인튜닝 모델 v3 기준)")

# ── 공통 필터 ──────────────────────────────────────────────────
with st.expander("🔍 전체 필터 (선택)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_opts = sorted(df["category"].dropna().unique())
        sel_cat_opt = st.selectbox("숙소 유형", ["전체"] + cat_opts)
        sel_cat = cat_opts if sel_cat_opt == "전체" else [sel_cat_opt]
    with c2:
        yr_opts = sorted(df["year"].dropna().unique())
        sel_yr_s = st.selectbox("시작 연도", yr_opts,
                                index=yr_opts.index("2017") if "2017" in yr_opts else 0)
        sel_yr_e = st.selectbox("종료 연도", yr_opts, index=len(yr_opts)-1)
    with c3:
        sel_sent_opt = st.selectbox("감성", ["전체", "긍정", "부정"])
        sel_sent = ["긍정","부정"] if sel_sent_opt == "전체" else [sel_sent_opt]

mask = (
    df["category"].isin(sel_cat) &
    df["year"].between(sel_yr_s, sel_yr_e) &
    df["sentiment"].isin(sel_sent)
)
dff = df[mask].copy()
st.caption(f"필터 적용 후: {len(dff):,}건")

# ── 탭 ────────────────────────────────────────────────────────
tab_ov, tab_wc, tab_rq1, tab_rq2, tab_rq3, tab_rq4, tab_model, tab_data = st.tabs([
    "📊 개요",
    "☁️ 워드클라우드",
    "🔍 RQ1 · 평점-감성",
    "🏘️ RQ2 · 숙소 유형·업소",
    "📈 RQ3 · 시계열 트렌드",
    "📝 RQ4 · 보조 분석",
    "🤖 모델 성능 비교",
    "📋 데이터",
])

# ═══════════════════════════════════════════════════
# TAB 0 · 개요
# ═══════════════════════════════════════════════════
with tab_ov:
    total = len(dff)
    pos_n = (dff["sentiment"]=="긍정").sum()
    neg_n = (dff["sentiment"]=="부정").sum()
    avg_r = dff["rating"].mean()
    avg_s = dff["sentiment_score"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: html_metric("전체 리뷰", f"{total:,}")
    with c2: html_metric("긍정", f"{pos_n:,}", f"{pos_n/total*100:.1f}%", C_POS)
    with c3: html_metric("부정", f"{neg_n:,}", f"{neg_n/total*100:.1f}%", C_NEG)
    with c4: html_metric("평균 평점", f"{avg_r:.2f} / 5")
    with c5: html_metric("평균 신뢰도", f"{avg_s:.3f}")

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        fig_pie = go.Figure(go.Pie(
            labels=["긍정","부정"], values=[pos_n, neg_n],
            hole=0.45, marker=dict(colors=[C_POS, C_NEG]),
            textinfo="label+percent", textfont_size=15,
        ))
        fig_pie.update_layout(title="감성 비율", showlegend=False, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
    with cb:
        cnt = dff["rating"].value_counts().sort_index().reset_index()
        cnt.columns = ["rating","건수"]
        fig_rat = px.bar(cnt, x="rating", y="건수",
            color_discrete_sequence=[C_BLUE], title="평점(rating) 분포",
            labels={"rating":"평점"})
        fig_rat.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_rat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 프로젝트 데이터 파이프라인 요약")
    cols = st.columns(5)
    steps = [
        ("🕷️", "크롤링", "Google Maps\nSelenium\n130,000건"),
        ("🔧", "전처리", "빈행·중복 제거\nTrip.com 제외\n81,613건"),
        ("🔍", "EDA", "컬럼 분석\n이상치 탐지\n설계서 작성"),
        ("🤖", "감성분석", "5종 모델 검증\n파인튜닝 v3\nF1: 0.8691"),
        ("📊", "시각화", "Streamlit\nRQ1~RQ4\n대시보드"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:#f0f9fb;border-radius:10px;padding:12px;text-align:center;min-height:120px">
              <p style="font-size:24px;margin:0">{icon}</p>
              <p style="font-size:13px;font-weight:600;color:#1F3864;margin:4px 0">{title}</p>
              <p style="font-size:11px;color:#666;margin:0;white-space:pre-line">{desc}</p>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 1 · 워드클라우드
# ═══════════════════════════════════════════════════
with tab_wc:
    st.subheader("☁️ 워드클라우드")
    st.caption("감성별로 자주 등장하는 핵심 단어를 시각화합니다.")

    col_opt, col_dl = st.columns([3, 1])
    with col_opt:
        wc_sel = st.radio("감성 선택", ["전체","긍정","부정"], horizontal=True, key="wc_sel")

    if wc_sel == "전체":
        wc_texts = dff["text"].dropna().tolist()
        wc_color = C_BLUE
    elif wc_sel == "긍정":
        wc_texts = dff[dff["sentiment"]=="긍정"]["text"].dropna().tolist()
        wc_color = C_POS
    else:
        wc_texts = dff[dff["sentiment"]=="부정"]["text"].dropna().tolist()
        wc_color = C_NEG

    with st.spinner("워드클라우드 생성 중..."):
        wc_fig = make_wordcloud(wc_texts, wc_color)

    if wc_fig:
        st.image(fig_bytes(wc_fig), use_container_width=True)
        st.download_button(
            f"⬇️ {wc_sel} 워드클라우드 저장 (PNG)",
            data=fig_bytes(wc_fig),
            file_name=f"wordcloud_{wc_sel}.png", mime="image/png"
        )
    else:
        st.warning("해당 조건에 텍스트가 없습니다.")

    # 부정 워드클라우드 인사이트
    if wc_sel == "부정":
        st.info("💡 부정 리뷰에도 '좋았습니다'가 자주 등장합니다. "
                "이는 한국어 리뷰의 역접 구조('좋았지만... 아쉬웠다') 패턴으로, "
                "모델은 전체 문맥을 보고 부정으로 분류합니다.")

# ═══════════════════════════════════════════════════
# TAB 2 · RQ1 평점-감성 일치도
# ═══════════════════════════════════════════════════
with tab_rq1:
    st.subheader("RQ1. 평점과 감성분석 결과는 얼마나 일치하는가?")
    st.caption("별점은 후하게 주지만 텍스트엔 불만이 담긴 '불일치 케이스'를 식별합니다.")

    cross = dff[dff["rating"].notna()].groupby(
        ["rating","sentiment"]).size().unstack(fill_value=0).reset_index()
    cross["total"] = cross[["긍정","부정"]].sum(axis=1)
    cross["긍정비율"] = (cross["긍정"]/cross["total"]*100).round(1)
    cross["부정비율"] = (cross["부정"]/cross["total"]*100).round(1)

    ca, cb = st.columns(2)
    with ca:
        fig_cross = px.bar(cross, x="rating", y=["긍정","부정"],
            color_discrete_map=COLOR_MAP, barmode="stack",
            title="평점별 긍정·부정 건수", labels={"rating":"평점","value":"건수","variable":"감성"})
        fig_cross.update_layout(height=360)
        st.plotly_chart(fig_cross, use_container_width=True)
    with cb:
        fig_pct = go.Figure()
        fig_pct.add_bar(x=cross["rating"], y=cross["긍정비율"], name="긍정", marker_color=C_POS)
        fig_pct.add_bar(x=cross["rating"], y=cross["부정비율"], name="부정", marker_color=C_NEG)
        fig_pct.update_layout(barmode="stack", title="평점별 감성 비율(%)",
            xaxis_title="평점", yaxis_title="%", height=360)
        st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 불일치 케이스 — 포트폴리오 핵심 포인트")
    c1, c2 = st.columns(2)
    with c1:
        high_mis = dff[(dff["rating"]==5)&(dff["sentiment"]=="부정")]
        html_metric("5점인데 부정", f"{len(high_mis):,}건",
                    f"전체의 {len(high_mis)/len(dff)*100:.1f}%", C_NEG)
        if len(high_mis):
            sub = high_mis[["place_name","rating","sentiment_score","text"]]\
                .sort_values("sentiment_score", ascending=False).head(10).reset_index(drop=True)
            sub["text"] = sub["text"].str[:60] + "…"
            fig_t = go.Figure(go.Table(
                header=dict(values=["업소명","평점","신뢰도","리뷰(앞 60자)"],
                            fill_color="#f0f0f0", align="left", font=dict(size=12)),
                cells=dict(values=[sub["place_name"],sub["rating"],
                                   sub["sentiment_score"].round(3),sub["text"]],
                           align="left", font=dict(size=11))
            ))
            fig_t.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=280)
            st.plotly_chart(fig_t, use_container_width=True)
    with c2:
        low_mis = dff[(dff["rating"]==1)&(dff["sentiment"]=="긍정")]
        html_metric("1점인데 긍정", f"{len(low_mis):,}건",
                    f"전체의 {len(low_mis)/len(dff)*100:.1f}%", C_POS)
        if len(low_mis):
            sub2 = low_mis[["place_name","rating","sentiment_score","text"]]\
                .sort_values("sentiment_score", ascending=False).head(10).reset_index(drop=True)
            sub2["text"] = sub2["text"].str[:60] + "…"
            fig_t2 = go.Figure(go.Table(
                header=dict(values=["업소명","평점","신뢰도","리뷰(앞 60자)"],
                            fill_color="#f0f0f0", align="left", font=dict(size=12)),
                cells=dict(values=[sub2["place_name"],sub2["rating"],
                                   sub2["sentiment_score"].round(3),sub2["text"]],
                           align="left", font=dict(size=11))
            ))
            fig_t2.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=280)
            st.plotly_chart(fig_t2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 긍정 점수 vs 부정 점수 산점도")
    samp = dff.sample(min(3000, len(dff)), random_state=42)
    fig_sc = px.scatter(samp, x="score_positive", y="score_negative",
        color="sentiment", color_discrete_map=COLOR_MAP, opacity=0.4, height=380,
        labels={"score_positive":"긍정 점수","score_negative":"부정 점수","sentiment":"감성"},
        title=f"감성 점수 분포 (샘플 {len(samp):,}건)")
    fig_sc.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 3 · RQ2 숙소 유형·업소
# ═══════════════════════════════════════════════════
with tab_rq2:
    st.subheader("RQ2. 숙소 유형·개별 업소별 감성과 키워드는 어떻게 다른가?")

    cat_df = dff.groupby(["category","sentiment"]).size().reset_index(name="건수")
    fig_cat = px.bar(cat_df, x="category", y="건수", color="sentiment",
        color_discrete_map=COLOR_MAP, barmode="group",
        title="숙소 유형(category)별 감성 분포",
        labels={"category":"유형","건수":"리뷰 수"})
    fig_cat.update_layout(height=360)
    st.plotly_chart(fig_cat, use_container_width=True)

    cat_stat = dff[dff["rating"].notna()].groupby("category").agg(
        건수=("text","count"), 평균평점=("rating","mean"),
        부정률=("sentiment", lambda x: (x=="부정").mean()*100)
    ).round(2).reset_index()
    fig_ct = go.Figure(go.Table(
        header=dict(values=["유형","건수","평균평점","부정률(%)"],
                    fill_color="#f0f0f0", align="center", font=dict(size=13)),
        cells=dict(values=[cat_stat["category"],cat_stat["건수"],
                           cat_stat["평균평점"],cat_stat["부정률"]],
                   align="center", font=dict(size=12))
    ))
    fig_ct.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=140)
    st.plotly_chart(fig_ct, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 업소별 분석")
    cc1, cc2 = st.columns(2)
    with cc1:
        min_rev = st.selectbox("최소 리뷰 수", [30,50,100,200], index=2)
    with cc2:
        top_n = st.selectbox("표시 업소 수", [10,15,20], index=1)

    place_stat = dff.groupby("place_name").agg(
        건수=("text","count"),
        평균평점=("rating","mean"),
        긍정률=("sentiment", lambda x: (x=="긍정").mean()*100),
        부정률=("sentiment", lambda x: (x=="부정").mean()*100),
    ).round(2).reset_index()
    place_stat = place_stat[place_stat["건수"] >= min_rev]

    ca, cb = st.columns(2)
    with ca:
        top_pos = place_stat.nlargest(top_n, "긍정률")
        fig_pos = px.bar(top_pos, x="긍정률", y="place_name", orientation="h",
            color_discrete_sequence=[C_POS], title=f"긍정률 상위 {top_n}개 업소",
            labels={"긍정률":"%","place_name":"업소명"})
        fig_pos.update_layout(height=max(360, top_n*28), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_pos, use_container_width=True)
    with cb:
        top_neg = place_stat.nlargest(top_n, "부정률")
        fig_neg = px.bar(top_neg, x="부정률", y="place_name", orientation="h",
            color_discrete_sequence=[C_NEG], title=f"부정률 상위 {top_n}개 업소",
            labels={"부정률":"%","place_name":"업소명"})
        fig_neg.update_layout(height=max(360, top_n*28), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_neg, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 업소별 워드클라우드")
    sel_place = st.selectbox("업소 선택", sorted(dff["place_name"].dropna().unique()))
    wc_sent2 = st.radio("감성", ["긍정","부정"], horizontal=True, key="rq2_wc")
    place_texts = dff[(dff["place_name"]==sel_place)&(dff["sentiment"]==wc_sent2)]["text"].dropna().tolist()
    st.caption(f"{sel_place} — {wc_sent2} 리뷰 {len(place_texts):,}건")
    with st.spinner("워드클라우드 생성 중..."):
        wc2 = make_wordcloud(place_texts, C_POS if wc_sent2=="긍정" else C_NEG)
    if wc2:
        st.image(fig_bytes(wc2), use_container_width=True)
    else:
        st.warning("해당 조건에 텍스트가 없습니다.")

# ═══════════════════════════════════════════════════
# TAB 4 · RQ3 시계열 트렌드
# ═══════════════════════════════════════════════════
with tab_rq3:
    st.subheader("RQ3. 시간 흐름에 따라 감성과 평점은 어떻게 변화했는가?")
    st.caption("코로나19(2020~2021) 전후 변화 및 계절성 패턴을 확인합니다.")

    ts = dff[dff["year"].notna()].copy()
    ts["year_int"] = ts["year"].astype(float)
    yr_g = ts.groupby("year_int").agg(
        건수=("text","count"),
        긍정수=("sentiment", lambda x: (x=="긍정").sum()),
        부정수=("sentiment", lambda x: (x=="부정").sum()),
        평균평점=("rating","mean"),
    ).reset_index()
    yr_g["긍정률"] = (yr_g["긍정수"]/yr_g["건수"]*100).round(1)
    yr_g = yr_g[yr_g["year_int"]>=2016]

    ca, cb = st.columns(2)
    with ca:
        fig_yr = go.Figure()
        fig_yr.add_scatter(x=yr_g["year_int"], y=yr_g["긍정률"],
            mode="lines+markers", name="긍정률(%)",
            line=dict(color=C_POS, width=2.5), marker=dict(size=8))
        fig_yr.add_vrect(x0=2020, x1=2021.99, fillcolor="#FFF3CD", opacity=0.4,
            layer="below", line_width=0,
            annotation_text="코로나(2020~2021)", annotation_position="top left",
            annotation_font_size=11)
        fig_yr.update_layout(title="연도별 긍정률 추이", xaxis_title="연도",
            yaxis_title="%", height=360, plot_bgcolor="white",
            yaxis=dict(range=[60,95], gridcolor="#eeeeee"))
        st.plotly_chart(fig_yr, use_container_width=True)
    with cb:
        fig_vol = go.Figure()
        fig_vol.add_bar(x=yr_g["year_int"], y=yr_g["긍정수"], name="긍정", marker_color=C_POS)
        fig_vol.add_bar(x=yr_g["year_int"], y=yr_g["부정수"], name="부정", marker_color=C_NEG)
        fig_vol.add_vrect(x0=2020, x1=2021.99, fillcolor="#FFF3CD",
            opacity=0.4, layer="below", line_width=0)
        fig_vol.update_layout(barmode="stack", title="연도별 리뷰 건수",
            xaxis_title="연도", yaxis_title="건수", height=360, plot_bgcolor="white")
        st.plotly_chart(fig_vol, use_container_width=True)

    fig_rat_yr = px.line(yr_g, x="year_int", y="평균평점", markers=True,
        title="연도별 평균 평점 추이",
        color_discrete_sequence=[C_BLUE],
        labels={"year_int":"연도","평균평점":"평균 평점"})
    fig_rat_yr.add_vrect(x0=2020, x1=2021.99, fillcolor="#FFF3CD", opacity=0.4,
        layer="below", line_width=0,
        annotation_text="코로나(2020~2021)", annotation_position="top left",
        annotation_font_size=11)
    fig_rat_yr.update_layout(height=300, plot_bgcolor="white",
        yaxis=dict(range=[3.5, 5.0], gridcolor="#eeeeee"))
    st.plotly_chart(fig_rat_yr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 주요 인사이트")
    pre  = yr_g[yr_g["year_int"].between(2017,2019)]["긍정률"].mean()
    cov  = yr_g[yr_g["year_int"].between(2020,2021)]["긍정률"].mean()
    post = yr_g[yr_g["year_int"]>=2022]["긍정률"].mean()
    c1,c2,c3 = st.columns(3)
    with c1: html_metric("코로나 이전 평균 긍정률\n(2017~2019)", f"{pre:.1f}%")
    with c2: html_metric("코로나 기간 평균 긍정률\n(2020~2021)", f"{cov:.1f}%", f"{cov-pre:+.1f}%p")
    with c3: html_metric("코로나 이후 평균 긍정률\n(2022~2026)", f"{post:.1f}%", f"{post-pre:+.1f}%p", C_NEG)

# ═══════════════════════════════════════════════════
# TAB 5 · RQ4 보조 분석
# ═══════════════════════════════════════════════════
with tab_rq4:
    st.subheader("RQ4. 리뷰 길이·플랫폼은 감성 표현과 관련이 있는가?")

    ca, cb = st.columns(2)
    with ca:
        fig_box = px.box(dff, x="sentiment", y="text_len",
            color="sentiment", color_discrete_map=COLOR_MAP,
            title="감성별 텍스트 길이 분포",
            labels={"sentiment":"감성","text_len":"텍스트 길이(글자 수)"}, points="outliers")
        fig_box.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with cb:
        plat = dff.groupby(["platform","sentiment"]).size().reset_index(name="건수")
        fig_plat = px.bar(plat, x="platform", y="건수", color="sentiment",
            color_discrete_map=COLOR_MAP, barmode="group",
            title="플랫폼별 감성 분포",
            labels={"platform":"플랫폼","건수":"리뷰 수"})
        fig_plat.update_layout(height=360)
        st.plotly_chart(fig_plat, use_container_width=True)

    st.markdown("---")
    fig_hist = px.histogram(dff, x="sentiment_score", color="sentiment",
        color_discrete_map=COLOR_MAP, nbins=40, barmode="overlay", opacity=0.7,
        title="감성 신뢰도 점수 분포 (v3 파인튜닝 모델)",
        labels={"sentiment_score":"신뢰도 점수","sentiment":"감성"})
    fig_hist.update_layout(height=300, plot_bgcolor="white")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 텍스트 길이 구간별 부정률")
    dff2 = dff.copy()
    dff2["길이구간"] = pd.cut(dff2["text_len"],
        bins=[0,50,100,200,500,9999],
        labels=["~50자","51~100자","101~200자","201~500자","500자~"])
    len_g = dff2.groupby("길이구간", observed=True).agg(
        건수=("text","count"),
        부정률=("sentiment", lambda x: (x=="부정").mean()*100)
    ).reset_index()
    fig_len = px.bar(len_g, x="길이구간", y="부정률", text="부정률",
        color_discrete_sequence=[C_NEG],
        title="텍스트 길이 구간별 부정 비율(%)",
        labels={"길이구간":"텍스트 길이","부정률":"%"})
    fig_len.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_len.update_layout(height=320, showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_len, use_container_width=True)

# ═══════════════════════════════════════════════════
# TAB 6 · 모델 성능 비교
# ═══════════════════════════════════════════════════
with tab_model:
    st.subheader("🤖 모델 성능 개선 과정")
    st.caption("베이스라인 → 직접 레이블링 파인튜닝 → 의사 레이블링 파인튜닝 단계별 성능 변화")

    c1,c2,c3 = st.columns(3)
    with c1: html_metric("v1 베이스라인", "F1: 0.7323", "기준점")
    with c2: html_metric("v2 1차 파인튜닝", "F1: 0.8550", "+16.8% ↑", C_POS)
    with c3: html_metric("v3 의사레이블 파인튜닝", "F1: 0.8691", "+17.5% ↑", C_POS)

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        fig_f1 = px.bar(
            pd.DataFrame({
                "버전": ["v1\n베이스라인", "v2\n1차 파인튜닝", "v3\n의사레이블"],
                "Macro F1": [0.7323, 0.8550, 0.8691],
                "색상": ["gray", "orange", "teal"]
            }),
            x="버전", y="Macro F1", text="Macro F1",
            color="버전",
            color_discrete_map={"v1\n베이스라인": "#888888", "v2\n1차 파인튜닝": "#EDA100", "v3\n의사레이블": C_POS},
            title="버전별 Macro F1 비교",
        )
        fig_f1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_f1.update_layout(height=380, showlegend=False, yaxis=dict(range=[0.6, 0.95]))
        st.plotly_chart(fig_f1, use_container_width=True)
    with cb:
        fig_neg_v = px.bar(
            pd.DataFrame({
                "버전": ["v1 베이스라인", "v2 1차 파인튜닝", "v3 의사레이블"],
                "부정률(%)": [17.6, 22.1, 28.1],
                "5점인데 부정": [956, 2413, 3777],
            }),
            x="버전", y="부정률(%)", text="부정률(%)",
            color="버전",
            color_discrete_map={"v1 베이스라인": "#888888", "v2 1차 파인튜닝": "#EDA100", "v3 의사레이블": C_POS},
            title="버전별 부정률 비교 (파인튜닝 후 더 엄격해짐)",
        )
        fig_neg_v.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_neg_v.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_neg_v, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 파인튜닝 단계별 설명")
    model_info = [
        ("v1\n베이스라인", "alsgyu/sentiment-analysis-fine-tuned-model", "없음", "0.7323", "82.4%", "956건"),
        ("v2\n1차 파인튜닝", "alsgyu (도메인 특화)", "직접 레이블링 459건", "0.8550", "77.9%", "2,413건"),
        ("v3\n의사레이블", "alsgyu (의사레이블 확장)", "459 + 의사레이블 4,000건", "0.8691", "71.9%", "3,777건"),
    ]
    fig_mt = go.Figure(go.Table(
        header=dict(
            values=["버전","베이스 모델","학습 데이터","Macro F1","긍정률","5점인데 부정"],
            fill_color="#1F3864", font=dict(color="white", size=12), align="center"
        ),
        cells=dict(
            values=[[r[0] for r in model_info],[r[1] for r in model_info],
                    [r[2] for r in model_info],[r[3] for r in model_info],
                    [r[4] for r in model_info],[r[5] for r in model_info]],
            align="center", font=dict(size=12), height=30,
            fill_color=[["#f5f5f5","#e8f4fd","#e1f5ee"]]*6
        )
    ))
    fig_mt.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=160)
    st.plotly_chart(fig_mt, use_container_width=True)

    st.info("💡 파인튜닝 후 부정 판별이 엄격해져서 긍정률은 낮아졌지만, "
            "이는 '텍스트 내용에 더 충실한 분류'를 의미합니다. "
            "5점인데 부정 탐지가 956건 → 3,777건으로 늘어난 것이 이를 뒷받침합니다.")

# ═══════════════════════════════════════════════════
# TAB 7 · 데이터
# ═══════════════════════════════════════════════════
with tab_data:
    st.subheader("원본 데이터 조회 및 다운로드")

    c1, c2, c3 = st.columns(3)
    with c1:
        cat_opts2 = sorted(dff["category"].dropna().unique())
        f_cat_opt = st.selectbox("유형", ["전체"]+cat_opts2, key="d_cat")
        f_cat = cat_opts2 if f_cat_opt=="전체" else [f_cat_opt]
    with c2:
        f_sent_opt = st.selectbox("감성", ["전체","긍정","부정"], key="d_sent")
        f_sent = ["긍정","부정"] if f_sent_opt=="전체" else [f_sent_opt]
    with c3:
        f_score = st.selectbox("최소 신뢰도", [0.5,0.6,0.7,0.8,0.9], index=0, key="d_score")

    dfd = dff[dff["category"].isin(f_cat) & dff["sentiment"].isin(f_sent) &
              (dff["sentiment_score"]>=f_score)].head(500)
    total_match = len(dff[dff["category"].isin(f_cat) & dff["sentiment"].isin(f_sent) &
                          (dff["sentiment_score"]>=f_score)])
    st.caption(f"상위 500건 표시 (전체 조건 일치: {total_match:,}건)")

    sub = dfd[["category","place_name","rating","sentiment","sentiment_score","text"]].reset_index(drop=True)
    sub["text"] = sub["text"].str[:80] + "…"
    sub["sentiment_score"] = sub["sentiment_score"].round(3)
    fig_dt = go.Figure(go.Table(
        header=dict(values=["유형","업소명","평점","감성","신뢰도","리뷰(앞 80자)"],
                    fill_color="#e8eaf6", align="left", font=dict(size=12)),
        cells=dict(values=[sub["category"],sub["place_name"],sub["rating"],
                           sub["sentiment"],sub["sentiment_score"],sub["text"]],
                   align="left", font=dict(size=11), height=24)
    ))
    fig_dt.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=500)
    st.plotly_chart(fig_dt, use_container_width=True)

    st.download_button(
        "⬇️ 필터 결과 CSV 다운로드 (전체)",
        data=csv_bytes(dff[dff["category"].isin(f_cat) & dff["sentiment"].isin(f_sent) &
                           (dff["sentiment_score"]>=f_score)]),
        file_name="filtered_result_v3.csv", mime="text/csv"
    )
