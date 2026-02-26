import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
from datetime import datetime
import math
import ast
import re

# ---------------------------------------------------------
# 0. 페이지 설정 & 다크모드 대응 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="PRISM Dashboard", layout="wide", initial_sidebar_state="collapsed")

if 'filtered_cache' not in st.session_state:
    st.session_state['filtered_cache'] = {}

st.markdown("""
<style>
.case-card {
    border: 1px solid rgba(128,128,128,0.3);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    background: rgba(128,128,128,0.06);
}
.case-card .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.case-card .card-title { font-weight: 700; font-size: 15px; }
.case-card .card-date { opacity: 0.6; font-size: 13px; }
.case-card .card-body { font-size: 14px; font-weight: 600; margin: 4px 0; }
.case-card .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
    margin-right: 6px;
    margin-top: 8px;
}
.tag-red { background: rgba(255,107,107,0.2); }
.tag-blue { background: rgba(30,144,255,0.2); }
.tag-green { background: rgba(78,205,196,0.2); }
.tag-orange { background: rgba(255,165,0,0.2); }
.tag-purple { background: rgba(162,155,254,0.2); }

.kpi-box {
    text-align: center;
    padding: 20px;
    background: rgba(128,128,128,0.08);
    border-radius: 12px;
    margin-top: 20px;
}
.kpi-box .kpi-label { opacity: 0.6; font-size: 14px; margin-bottom: 4px; }
.kpi-box .kpi-value { font-size: 42px; font-weight: 700; margin: 0; }
.kpi-box .kpi-sub { opacity: 0.6; font-size: 13px; margin-top: 4px; }

.info-box {
    padding: 20px;
    background: rgba(128,128,128,0.08);
    border-radius: 12px;
    margin-top: 20px;
}
.info-box p { margin: 4px 0; font-size: 15px; }

.legend-row {
    display: flex; gap: 24px; justify-content: center; margin-top: -10px;
}
.legend-row .leg-red { color: #FF6B6B; font-weight: 600; }
.legend-row .leg-blue { color: #4DABF7; font-weight: 600; }

.big-metric { margin-top: 10px; }
.big-metric .metric-label { font-size: 14px; opacity: 0.6; margin-bottom: 0; }
.big-metric .metric-value { font-size: 26px; font-weight: 600; margin-top: 0; line-height: 1.2; }
.big-metric .metric-sub { font-size: 20px; opacity: 0.6; font-weight: 400; }

button[data-baseweb="tab"] {
    font-size: 1.2rem !important;
    padding: 12px 24px !important;
}
button[data-baseweb="tab"] > div {
    font-size: 1.2rem !important;
}

.news-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #2E86AB;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.news-card:hover { border-left-color: #FF6B6B; }
.news-card .news-title { font-weight: 600; font-size: 14px; margin: 0 0 4px 0; line-height: 1.4; }
.news-card .news-title a { text-decoration: none; color: inherit; }
.news-card .news-title a:hover { color: #2E86AB; }
.news-card .news-meta { font-size: 12px; opacity: 0.55; margin: 0; }
.news-card .news-summary { font-size: 13px; opacity: 0.7; margin: 4px 0 0 0; line-height: 1.4; }
.news-section-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.news-badge { background: linear-gradient(135deg, #2E86AB, #1a5276); color: #fff; font-size: 11px;
    padding: 2px 8px; border-radius: 10px; font-weight: 600; }

/* ── Top Navigation Bar (Sticky) ── */
.nav-bar { display:flex; gap:6px; padding:8px 0 12px 0; border-bottom:2px solid rgba(128,128,128,0.15); margin-bottom:16px; flex-wrap:wrap; }
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: linear-gradient(135deg, #0d1b2a, #1b2a4a) !important;
    border-color: #2a3f6a !important;
    color: #e0e8ff !important;
    box-shadow: 0 2px 8px rgba(13,27,42,0.5);
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    border: 1.5px solid rgba(100,100,100,0.25);
    background: rgba(128,128,128,0.06);
}

/* ── KPI Stat Cards (image reference style) ── */
.stat-card { border-radius:10px; overflow:hidden; border:1px solid rgba(128,128,128,0.15); height:100%; }
.stat-card-header { background:#1a1a2e; color:#fff; padding:10px 16px; font-size:12px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center; }
.stat-card-header .stat-icon { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:16px; }
.stat-card-body { padding:16px; background:rgba(128,128,128,0.04); }
.stat-card-body .stat-value { font-size:32px; font-weight:800; margin:4px 0 8px 0; line-height:1; }
.stat-card-body .stat-delta { font-size:12px; font-weight:600; }
.stat-delta.positive { color:#2ecc71; }
.stat-delta.negative { color:#FF6B6B; }

/* ── News Grid ── */
.news-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.news-grid-card { border:1px solid rgba(128,128,128,0.2); border-left:3px solid #2E86AB; border-radius:8px; padding:14px 16px; transition:all 0.2s; }
.news-grid-card:hover { border-left-color:#FF6B6B; background:rgba(128,128,128,0.04); }
.news-grid-card .ng-title { font-weight:600; font-size:15px; margin:0 0 6px 0; line-height:1.4; }
.news-grid-card .ng-title a { text-decoration:none; color:inherit; }
.news-grid-card .ng-title a:hover { color:#2E86AB; }
.news-grid-card .ng-meta { font-size:13px; opacity:0.55; margin:0; }
.news-grid-card .ng-summary { font-size:14px; opacity:0.7; margin:4px 0 0 0; line-height:1.3; }
</style>
""", unsafe_allow_html=True)

pio.templates["darkfix"] = go.layout.Template(
    layout=dict(
        font=dict(color="rgba(255,255,255,0.85)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)", automargin=True),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", automargin=True),
    )
)
pio.templates.default = "plotly_dark+darkfix"

# ---------------------------------------------------------
# 1. 상수 정의
# ---------------------------------------------------------
METADATA_MAP = {
    "재무/회계/계약": ["계약","회계","예산","지출","구매","입찰","정산","금전","수입"],
    "인사/채용/복무": ["인사","채용","복무","근태","휴직","급여","임용","퇴직","징계"],
    "시설/안전/환경": ["시설","안전","건설","공사","환경","재난","건축","하자","소방"],
    "정보보안/IT": ["보안","정보","시스템","전산","데이터","네트워크","개인정보"],
    "윤리/부패/비위": ["청렴","부패","비위","행동강령","갑질","향응","금품"],
    "사업/운영/성과": ["사업","운영","성과","관리","제도","평가","기획","경영"]
}

DISPOSITION_SEVERITY = {"중징계":10, "경징계":9, "시정":8, "경고/주의":7, "통보":6, "현지조치":5}

DISP_ORDER = ["중징계", "경징계", "시정", "경고/주의", "통보", "현지조치"]

DISP_COLOR_MAP = {
    '중징계':'#FF6B6B','경징계':'#FFD93D','시정':'#4ECDC4',
    '경고/주의':'#74b9ff','통보':'#a29bfe','현지조치':'#dfe6e9','기타':'#D3D3D3'
}

GROUP_MEMBERS = {
    "중징계":["파면","정직","중징계","고발","문책"],
    "경징계":["감봉","견책","경징계","징계"],
    "시정":["시정","감액","회수","환수"],
    "경고/주의":["경고","주의","면책"],
    "통보":["통보","개선","권고","마련"],
    "현지조치":["현지조치","현지시정"],
}

# 처분 분류 기준 통일 disclaimer (Macro/Micro 공용)
DISPOSITION_DISCLAIMER = """
| 처분 그룹 | 강도점수 | 포함 키워드 | 행정적 성격 |
|---|:---:|---|---|
| **중징계** | 10점 | 파면, 정직, 중징계, 고발, 문책 | 신분 박탈 및 사법적 조치 (최고 수위) |
| **경징계** | 9점 | 감봉, 견책, 경징계, 징계 | 경제적 징벌 및 인사 기록 반영 |
| **시정** | 8점 | 시정, 감액, 회수, 환수 | 행정상 원상복구 및 금전적 회수 조치 |
| **경고/주의** | 7점 | 경고, 주의, 면책 | 과실 환기 및 주의 촉구 |
| **통보** | 6점 | 통보, 개선, 권고, 마련 | 제도 개선 및 자율적 시정 유도 |
| **현지조치** | 5점 | 현지조치, 현지시정 | 현장 즉시 시정 (경미 사항) |
"""

# ---------------------------------------------------------
# 2. 데이터 로드 — v4.2 (audit_v10.json 스키마)
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data(version="v4.2"):
    for fname in ['audit_v10.json', 'audit_v9.json', 'audit_v4_clean(no_sub_category_added).json']:
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
            break
        except FileNotFoundError:
            continue
    else:
        st.error("⚠️ JSON 파일이 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # ── 날짜 파싱 (epoch ms 또는 문자열) ──
    if 'date' in df.columns:
        sample = df['date'].dropna().iloc[0] if not df['date'].dropna().empty else None
        if isinstance(sample, (int, float)) and sample > 1e10:
            df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year.astype('Int64')
        df['month'] = df['date'].dt.month.astype('Int64')
        df['quarter'] = df['date'].dt.quarter.astype('Int64')
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        df['year_month'] = df['date'].dt.to_period('M').astype(str)

    # ── 금액 ──
    if 'penalty_amount' in df.columns:
        df['penalty_amount'] = pd.to_numeric(df['penalty_amount'], errors='coerce').fillna(0)
        df['penalty_amount_mill'] = df['penalty_amount'] / 1_000_000
    if 'penalty_type' not in df.columns: df['penalty_type'] = 'N/A'
    if 'doc_code' not in df.columns: df['doc_code'] = ''

    # ── audit_report_type (v9) / audit_type (legacy) ──
    if 'audit_report_type' in df.columns:
        df['audit_report_type'] = df['audit_report_type'].fillna('미분류').replace('', '미분류')
    elif 'audit_type' in df.columns:
        df.rename(columns={'audit_type': 'audit_report_type'}, inplace=True)
        df['audit_report_type'] = df['audit_report_type'].fillna('미분류').replace('', '미분류')
    else:
        df['audit_report_type'] = '미분류'

    if 'site' not in df.columns: df['site'] = '미분류'

    # ── agency_category 직접 사용 (v10에서 sub_category→agency_category 변경완료) ──
    if 'agency_category' in df.columns:
        df['agency_category'] = df['agency_category'].fillna('미분류').replace('', '미분류')
        # org_name alias removed — using agency_category directly
    elif 'sub_category' in df.columns:
        df['agency_category'] = df['sub_category'].fillna('미분류')
        df['agency_category'] = df['agency_category']
    elif 'category' in df.columns:
        df.rename(columns={'category': 'agency_category'}, inplace=True)
        df['agency_category'] = df['agency_category'].fillna('미분류').replace('', '미분류')
        # legacy parsing removed
    else:
        df['agency_category'] = '미분류'
        df['agency_category'] = '미분류'

    # ── org_category (v9 신규) ──
    if 'org_category' not in df.columns:
        df['org_category'] = '미분류'
    else:
        df['org_category'] = df['org_category'].fillna('미분류').replace('', '미분류')

    # ── contents_summary.title → summary_title ──
    def _extract_summary_title(val):
        # Case 1: already a dict (from json.load)
        if isinstance(val, dict):
            return val.get('title', '')
        # Case 2: NaN / None / non-string
        if not isinstance(val, str) or not val.strip():
            return ''
        val = val.strip()
        # Case 3: JSON string  '{"title": "..."}'
        if val.startswith('{'):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed.get('title', '')
            except (json.JSONDecodeError, ValueError):
                pass
            # Case 4: Python repr string  "{'title': '...'}"
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, dict):
                    return parsed.get('title', '')
            except (ValueError, SyntaxError):
                pass
        # Case 5: plain string (legacy contents_summary)
        return val
    if 'contents_summary' in df.columns:
        df['summary_title'] = df['contents_summary'].apply(_extract_summary_title)
    else:
        df['summary_title'] = ''
    # 비어있으면 원본 title fallback
    mask_empty = df['summary_title'].fillna('').str.strip() == ''
    df.loc[mask_empty, 'summary_title'] = df.loc[mask_empty, 'title'].fillna('')

    # ── risk_category (v9: 사전 분류 완료) ──
    VALID_RISK_CATS = set(METADATA_MAP.keys())
    if 'risk_category' not in df.columns:
        df['risk_category'] = '사업/운영/성과'
    else:
        df['risk_category'] = df['risk_category'].fillna('사업/운영/성과')
        df['risk_category'] = df['risk_category'].str.replace('\\/', '/', regex=False)
        invalid = ~df['risk_category'].isin(VALID_RISK_CATS)
        if invalid.any():
            df.loc[invalid, 'risk_category'] = '사업/운영/성과'

    # ── disposition_level (v9: 그룹 수준으로 사전 분류 완료) ──
    # v9의 disposition_level = 중징계/경징계/시정/경고·주의/통보/현지조치
    VALID_DISP = set(DISPOSITION_SEVERITY.keys())
    if 'disposition_level' not in df.columns:
        df['disposition_level'] = '기타'
    else:
        df['disposition_level'] = df['disposition_level'].fillna('기타')
        # "경고/주의" 변환 처리 (JSON 이스케이프)
        df['disposition_level'] = df['disposition_level'].str.replace('경고\\/주의', '경고/주의', regex=False)
        invalid_d = ~df['disposition_level'].isin(VALID_DISP) & (df['disposition_level'] != '기타')
        if invalid_d.any():
            df.loc[invalid_d, 'disposition_level'] = '기타'
    df['disposition_severity'] = df['disposition_level'].map(DISPOSITION_SEVERITY).fillna(0)

    return df

@st.cache_data
def load_risk_keywords():
    try:
        with open('matched_keywords_only.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return METADATA_MAP.copy()

if 'df' not in st.session_state or 'risk_category' not in st.session_state['df'].columns:
    with st.spinner('🚀 데이터 로딩 중...'):
        st.cache_data.clear()
        st.session_state['df'] = load_and_process_data()
df = st.session_state['df']

# 방어 검증
for _col, _def in [('risk_category','사업/운영/성과'),('disposition_level','기타'),
                    ('summary_title',''),('org_category','미분류'),('audit_report_type','미분류')]:
    if _col not in df.columns:
        df[_col] = _def

# Helper: 카드 2열 렌더링
# ---------------------------------------------------------
def render_cards_2col(cases_df, max_rows=5, show_doc_code=False):
    """카드를 1행 2열, 최대 max_rows행(=max_rows*2건)으로 표시"""
    display_df = cases_df.head(max_rows * 2)
    rows_list = list(display_df.iterrows())
    for i in range(0, len(rows_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(rows_list):
                _, row = rows_list[i + j]
                with col:
                    date_str = str(row.get('date',''))[:10]
                    org = row.get('agency_category','N/A')
                    title = str(row.get('summary_title','') or row.get('title',''))[:80]
                    atype = row.get('audit_report_type','')
                    disp = row.get('disposition_level','')
                    risk = row.get('risk_category','')
                    org_cat = row.get('org_category','')
                    doc_code = str(row.get('doc_code','')) if show_doc_code else ''
                    doc_tag = f'<span class="tag tag-purple">📄 {doc_code}</span>' if doc_code and doc_code != 'nan' and doc_code.strip() else ''
                    org_cat_tag = f'<span class="tag tag-orange">🏛️ {org_cat}</span>' if org_cat and org_cat != '미분류' else ''
                    st.markdown(
                        f'<div class="case-card">'
                        f'<div class="card-header"><span class="card-title">🏢 {org}</span><span class="card-date">📅 {date_str}</span></div>'
                        f'<p class="card-body">{title}</p>'
                        f'<span class="tag tag-red">📌 {disp}</span>'
                        f'<span class="tag tag-blue">🔍 {atype}</span>'
                        f'<span class="tag tag-green">📂 {risk}</span>'
                        f'{org_cat_tag}'
                        f'{doc_tag}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    total = len(cases_df)
    shown = min(total, max_rows * 2)
    if total > shown:
        st.info(f"상위 {shown}건 표시 (전체 {total}건)")

# Helper: 형평성 통계 블록 (Micro drilldown + 사례검색 공용)
def render_equity_stats(sdf, section_key="eq"):
    """필터링된 결과 집단의 처분 분포 통계 시각화"""
    if sdf.empty:
        return

    disp_counts = sdf['disposition_level'].value_counts()
    total_srch = len(sdf)
    top_disp = disp_counts.index[0] if not disp_counts.empty else "N/A"
    top_disp_pct = (disp_counts.iloc[0] / total_srch * 100) if not disp_counts.empty else 0

    eq_k1, eq_k2, eq_k3, eq_k4 = st.columns(4)
    eq_k1.metric("📋 검색 건수", f"{total_srch:,}건")
    eq_k2.metric("🏆 최다 처분", f"{top_disp}")
    eq_k3.metric("📐 최다 비율", f"{top_disp_pct:.1f}%")
    heavy = disp_counts.get('중징계', 0)
    eq_k4.metric("🔴 중징계", f"{heavy}건 ({heavy/total_srch*100:.1f}%)" if total_srch > 0 else "0건")

    # 형평성 근거 메시지
    if not disp_counts.empty:
        msg_parts = []
        for dg in DISP_ORDER:
            cnt = disp_counts.get(dg, 0)
            if cnt > 0:
                msg_parts.append(f"{dg} {cnt}건({cnt/total_srch*100:.0f}%)")
        st.success(f"💡 **형평성 근거:** 검색된 {total_srch}건 중 — {' · '.join(msg_parts)}")

    # Pie + Bar
    ch1, ch2 = st.columns(2)
    with ch1:
        disp_df = disp_counts.reset_index()
        disp_df.columns = ['처분그룹', '건수']
        order_map = {d: i for i, d in enumerate(DISP_ORDER + ['기타'])}
        disp_df['order'] = disp_df['처분그룹'].map(order_map).fillna(99)
        disp_df = disp_df.sort_values('order')
        fig_pie = go.Figure(go.Pie(
            labels=disp_df['처분그룹'], values=disp_df['건수'],
            hole=0.45, textinfo='label+percent+value',
            textposition='auto',
            marker=dict(colors=[DISP_COLOR_MAP.get(d, '#D3D3D3') for d in disp_df['처분그룹']])
        ))
        fig_pie.update_layout(
            title=dict(text="처분 그룹별 분포", font=dict(size=14)),
            height=350, margin=dict(l=10,r=10,t=40,b=10), showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        if 'risk_category' in sdf.columns:
            risk_eq = sdf.groupby('risk_category').agg(
                건수=('disposition_severity', 'count'),
                평균강도=('disposition_severity', 'mean')
            ).reset_index().sort_values('평균강도', ascending=True)
            fig_bar = go.Figure(go.Bar(
                x=risk_eq['평균강도'], y=risk_eq['risk_category'],
                orientation='h', text=risk_eq['평균강도'].apply(lambda x: f"{x:.1f}"),
                textposition='auto', textfont=dict(size=10),
                marker=dict(color=risk_eq['평균강도'], colorscale='RdYlGn_r', showscale=False)
            ))
            fig_bar.update_layout(
                title=dict(text="위반 유형별 평균 처분 강도", font=dict(size=14)),
                xaxis_title="평균 강도", height=350, margin=dict(l=10,r=10,t=40,b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# Helper: AI 검색 추천 키워드 카드
# ---------------------------------------------------------
def render_ai_search_cards(context_df, full_df, context_label="", section_key="ai"):
    """현재 필터 맥락에서 유사 사례 6건 추출 + AI 검색 추천 키워드 제공"""
    if context_df.empty or full_df.empty:
        return

    # 현재 맥락의 주요 프로필 추출
    top_risk = context_df['risk_category'].mode().iloc[0] if not context_df['risk_category'].mode().empty else None
    top_disp = context_df['disposition_level'].mode().iloc[0] if not context_df['disposition_level'].mode().empty else None
    top_atype = context_df['audit_report_type'].mode().iloc[0] if not context_df['audit_report_type'].mode().empty else None
    context_orgs = set(context_df['agency_category'].unique())

    # 유사 사례 추출: 동일 맥락 속성 2개 이상 매치, 다른 기관 우선
    cond = pd.Series([False]*len(full_df), index=full_df.index)
    match_score = pd.Series([0]*len(full_df), index=full_df.index)
    if top_risk: match_score += (full_df['risk_category'] == top_risk).astype(int)
    if top_disp: match_score += (full_df['disposition_level'] == top_disp).astype(int)
    if top_atype: match_score += (full_df['audit_report_type'] == top_atype).astype(int)

    candidates = full_df[match_score >= 2].copy()
    # 다른 기관 우선, 기관별 최대 2건
    other_orgs = candidates[~candidates['agency_category'].isin(context_orgs)]
    if len(other_orgs) >= 6:
        similar = other_orgs.groupby('agency_category').head(2).sort_values('date', ascending=False).head(6)
    else:
        similar = candidates.groupby('agency_category').head(2).sort_values('date', ascending=False).head(6)

    if similar.empty:
        return

    st.divider()
    st.subheader("🤖 AI 검색 추천 키워드")
    st.caption("현재 조회 맥락 기반 유사 사례 — 키워드를 복사하여 AI 검색에서 상세 조회하세요")

    # 현재 맥락 요약 키워드
    context_parts = []
    if top_risk: context_parts.append(top_risk)
    if top_disp: context_parts.append(top_disp)
    if top_atype: context_parts.append(top_atype)
    if context_label: context_parts.insert(0, context_label)
    context_kw = " ".join(context_parts)
    st.markdown("**📋 현재 맥락 추천 키워드:**")
    st.code(context_kw, language=None)

    st.markdown(f"**📌 유사 사례 {len(similar)}건** (키워드 복사 후 AI 검색 활용)")
    rows_list = list(similar.iterrows())
    for i in range(0, len(rows_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(rows_list):
                _, row = rows_list[i + j]
                with col:
                    org = row.get('agency_category','N/A')
                    org_cat = row.get('org_category','')
                    risk = row.get('risk_category','')
                    disp = row.get('disposition_level','')
                    atype = row.get('audit_report_type','')
                    title = str(row.get('summary_title','') or row.get('title',''))[:60]
                    date_str = str(row.get('date',''))[:10]
                    pen_amt = row.get('penalty_amount', 0)
                    pen_label = f"{pen_amt/1e6:,.0f}백만원" if pen_amt > 0 else "해당없음"
                    pen_type_raw = str(row.get('penalty_type','')) if pd.notna(row.get('penalty_type','')) else ''
                    pen_type = re.sub(r'^\d+\.', '', pen_type_raw).strip() if pen_type_raw else ''
                    doc_code = str(row.get('doc_code',''))
                    dl_url = str(row.get('download_url','')) if pd.notna(row.get('download_url','')) else ''
                    title_html = f'<a href="{dl_url}" target="_blank" style="text-decoration:none;color:inherit;">{title} </a>' if dl_url and dl_url.startswith('http') else title
                    oc_tag = f'<span class="tag tag-orange" title="기관 유형">🏛️ {org_cat}</span>' if org_cat and org_cat != '미분류' else ''
                    at_tag = f'<span class="tag tag-blue" title="감사 종류">🔍 {atype}</span>'
                    rk_tag = f'<span class="tag tag-green" title="위반 유형">📂 {risk}</span>'
                    dl_tag = f'<span class="tag tag-red" title="처분 유형">📌 {disp}</span>'
                    pt_tag = f'<span class="tag tag-blue" title="재정 처분 유형">💼 {pen_type}</span>' if pen_type and pen_type not in ('','N/A','nan') else ''
                    pa_tag = f'<span class="tag tag-orange" title="처분 금액">💰 {pen_label}</span>'
                    dc_tag = f'<span class="tag tag-purple" title="문서 번호">📄 {doc_code}</span>' if doc_code and doc_code not in ('','nan') and doc_code.strip() else ''
                    st.markdown(
                        f'<div class="case-card">'
                        f'<div class="card-header"><span class="card-title">🏢 {org}</span><span class="card-date">📅 {date_str}</span></div>'
                        f'<p class="card-body">{title_html}</p>'
                        f'{oc_tag}{at_tag}{rk_tag}{dl_tag}{pt_tag}{pa_tag}{dc_tag}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    kw = f"{org} {disp} {risk} {atype} {pen_type} 패널티({pen_label})"
                    st.code(kw, language=None)

# ---------------------------------------------------------
# Helper: 키워드 클라우드 (Layer 1 — 분류 키워드 빈도 기반)
# ---------------------------------------------------------
def render_keyword_cloud(target_df, section_key="kwc"):
    """
    A+B 하이브리드 키워드 클라우드
    A: 범용어 하드코딩 제외 — 리스크 식별에 기여하지 않는 일반 행정용어(성과,수익 등 8개)만 제외
    B: TF-IDF 가중치 — 특정 카테고리에 집중된 키워드일수록 크게 표시
    """
    if target_df.empty:
        return

    RISK_KW = load_risk_keywords()
    RISK_COLORS = {
        "재무/회계/계약": "#FF6B6B", "인사/채용/복무": "#FFD93D",
        "시설/안전/환경": "#4ECDC4", "정보보안/IT": "#74b9ff",
        "윤리/부패/비위": "#a29bfe", "사업/운영/성과": "#fd79a8"
    }
    num_cats = len(RISK_KW)  # 전체 카테고리 수 (6)
    # 범용어 제외 목록: 5개+ 분야에 공통 출현하나 리스크 식별에 기여하지 않는 일반 행정용어만 제외
    EXCLUDE_GENERIC = {'성과', '수익', '실태', '연구', '정산', '지출', '평가', '품질'}

    # --- Step 1: 카테고리별 텍스트 구축 ---
    cat_texts = {}
    for cat in RISK_KW.keys():
        sub = target_df[target_df['risk_category'] == cat]
        if sub.empty:
            cat_texts[cat] = ""
            continue
        cat_texts[cat] = (
            sub['summary_title'].astype(str) + ' ' +
            sub.get('action', pd.Series(['']*len(sub), index=sub.index)).astype(str)
        ).str.cat(sep=' ')

    # --- Step 2: 키워드별 카테고리 출현 빈도 매트릭스 ---
    kw_cat_freq = {}   # {keyword: {cat: count}}
    kw_home_cat = {}   # {keyword: 원래 소속 카테고리}
    for cat, keywords in RISK_KW.items():
        for kw in keywords:
            if kw not in kw_home_cat:
                kw_home_cat[kw] = cat
            if kw not in kw_cat_freq:
                kw_cat_freq[kw] = {}
            for c, txt in cat_texts.items():
                cnt = txt.count(kw)
                if cnt > 0:
                    kw_cat_freq[kw][c] = kw_cat_freq[kw].get(c, 0) + cnt

    # --- Step 3: 범용어 제외 (하드코딩 목록 기반) ---
    filtered_kws = {}
    excluded_kws = []
    for kw, cat_counts in kw_cat_freq.items():
        total_freq = sum(cat_counts.values())
        if total_freq == 0:
            continue
        if kw in EXCLUDE_GENERIC:
            excluded_kws.append(kw)
            continue
        filtered_kws[kw] = cat_counts

    # --- Step 4 (방법 B): TF-IDF 계산 ---
    # 카테고리별 전체 키워드 빈도 합 (TF 분모)
    cat_total_freq = {}
    for kw, cat_counts in filtered_kws.items():
        for c, cnt in cat_counts.items():
            cat_total_freq[c] = cat_total_freq.get(c, 0) + cnt

    kw_records = []
    for kw, cat_counts in filtered_kws.items():
        spread = len(cat_counts)
        idf = math.log(num_cats / spread) if spread > 0 else 0

        # 가장 많이 등장한 카테고리를 대표 카테고리로 선정
        best_cat = max(cat_counts, key=cat_counts.get)
        best_freq = cat_counts[best_cat]
        total_freq = sum(cat_counts.values())

        # TF = 대표 카테고리 내 빈도 / 해당 카테고리 전체 키워드 빈도합
        tf = best_freq / cat_total_freq[best_cat] if cat_total_freq.get(best_cat, 0) > 0 else 0
        tfidf = tf * idf

        # 소속 카테고리 결정: 원래 정의된 카테고리 우선, 아니면 최다 출현 카테고리
        home = kw_home_cat.get(kw, best_cat)

        kw_records.append({
            'keyword': kw,
            'category': home,
            'count': total_freq,
            'tfidf': tfidf,
            'spread': spread,
            'color': RISK_COLORS.get(home, '#ccc')
        })

    if not kw_records:
        return

    kw_df = pd.DataFrame(kw_records)
    # TF-IDF 점수 기준 정렬 → 상위 60개
    kw_df = kw_df.sort_values('tfidf', ascending=False).head(60)

    # --- Step 5: Plotly scatter 워드클라우드 ---
    np.random.seed(42)
    n = len(kw_df)
    kw_df['x'] = np.random.uniform(0, 100, n)
    kw_df['y'] = np.random.uniform(0, 100, n)
    max_score = kw_df['tfidf'].max()
    min_score = kw_df['tfidf'].min()
    kw_df['size'] = 10 + (kw_df['tfidf'] - min_score) / max(max_score - min_score, 1e-9) * 40

    fig = go.Figure()
    for cat in kw_df['category'].unique():
        sub = kw_df[kw_df['category'] == cat]
        fig.add_trace(go.Scatter(
            x=sub['x'], y=sub['y'], mode='text',
            text=sub['keyword'], name=cat,
            textfont=dict(size=sub['size'].tolist(), color=sub['color'].tolist()),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'TF-IDF: %{customdata[0]:.4f}<br>'
                '빈도: %{customdata[1]}회<br>'
                '집중도: %{customdata[2]}/6 카테고리<br>'
                '분야: %{customdata[3]}<extra></extra>'
            ),
            customdata=list(zip(sub['tfidf'], sub['count'], sub['spread'], sub['category']))
        ))
    fig.update_layout(
        height=420, margin=dict(l=10,r=10,t=30,b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=True, legend=dict(orientation="h",y=-0.05,x=0.5,xanchor="center",font=dict(size=11)),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 제외된 범용 키워드 표시
    if excluded_kws:
        with st.expander(f"ℹ️ 범용 키워드 {len(excluded_kws)}개 제외됨"):
            st.caption("5개 이상 분야에 공통 출현하되, 리스크 식별보다는 일반 행정 맥락에 해당하는 단어만 제외했습니다. "
                       "감사 문맥에서 리스크 파악에 필요한 핵심 단어(예: 관리, 운영, 계약 등)는 다수 분야에 걸쳐도 포함합니다.")
            st.caption(" · ".join(sorted(excluded_kws)))

# ---------------------------------------------------------
# Helper: 블랙엣지뉴스 피드 (기획 가안 — 샘플 데이터)
# TODO: 추후 RSS 또는 크롤링 라이브 연동 시 이 리스트를 동적 fetch로 교체
# ---------------------------------------------------------
BLACKEDGE_NEWS = [
    {"date": "2026-02-12", "title": "공공기관의 안전한 데이터 활용, 전문기관이 '원스톱'으로 해결한다",
     "summary": "개인정보보호위원회, 가명정보 원스톱 지원센터 시범운영기관 지정 공모 진행",
     "url": "https://blackedge.news/news/6435", "tag": "정보보안/IT"},
    {"date": "2026-02-05", "title": "무역보험공사 '장기 연대보증채권' 방치…종결기준·재산조사 개선 요구",
     "summary": "감사원, 국내채권 관리체계 전반 개선 필요 지적",
     "url": "https://blackedge.news/news/6325", "tag": "재무/회계/계약"},
    {"date": "2026-02-05", "title": "사실상 폐업기업에도 '무감액 연장'…무역보험공사 수출신용보증 사후관리 허점 드러나",
     "summary": "수출신용보증 무감액 만기연장 제도의 사후관리 부실",
     "url": "https://blackedge.news/news/6321", "tag": "재무/회계/계약"},
    {"date": "2026-02-05", "title": "군 복무 경력 전면 가산에 승진 격차…무역보험공사 인사제도 개선 요구",
     "summary": "병역의무 이행기간 전면 가산이 승진 형평성 훼손",
     "url": "https://blackedge.news/news/6324", "tag": "인사/채용/복무"},
    {"date": "2026-02-04", "title": "정부, 금융감독원 공공기관 지정 유보",
     "summary": "기획재정부, 금감원 공공기관 지정 여부 재검토",
     "url": "https://blackedge.news/news/6304", "tag": "사업/운영/성과"},
    {"date": "2026-02-04", "title": "외부 해킹이 주범인데 내부 통제만 강화…공공 개인정보 관리의 허점 드러나",
     "summary": "외부 해킹 대응 vs 내부 통제 강화 간 불균형 지적",
     "url": "https://blackedge.news/news/6263", "tag": "정보보안/IT"},
]

RISK_TAG_COLORS = {
    "재무/회계/계약": "#FF6B6B", "인사/채용/복무": "#FFD93D",
    "시설/안전/환경": "#4ECDC4", "정보보안/IT": "#74b9ff",
    "윤리/부패/비위": "#a29bfe", "사업/운영/성과": "#fd79a8"
}

def render_news_feed(max_items=5, compact=False):
    """블랙엣지뉴스 Audit News 피드 렌더링
    compact=True: Home용 (제목+날짜만), compact=False: 전체 피드 (요약 포함)
    """
    items = BLACKEDGE_NEWS[:max_items]
    for news in items:
        tag_color = RISK_TAG_COLORS.get(news["tag"], "#888")
        tag_html = f'<span style="background:rgba({int(tag_color[1:3],16)},{int(tag_color[3:5],16)},{int(tag_color[5:7],16)},0.2);color:{tag_color};font-size:11px;padding:1px 6px;border-radius:8px;font-weight:600;">{news["tag"]}</span>'
        summary_html = f'<p class="news-summary">{news["summary"]}</p>' if not compact and news.get("summary") else ''
        st.markdown(
            f'<div class="news-card">'
            f'<p class="news-title"><a href="{news["url"]}" target="_blank">{news["title"]}</a></p>'
            f'<p class="news-meta">📅 {news["date"]}  {tag_html}</p>'
            f'{summary_html}'
            f'</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# 3. 네비게이션 시스템
# ---------------------------------------------------------
MENU_OPTIONS = [
    ("Home", "Home"),
    ("감사 트렌드", "감사 트렌드"),
    ("기관 벤치마크 (Macro)", "리스크 - 기관 벤치마크 (Macro view)"),
    ("처분 분석 (Micro)", "리스크 - 처분 분석 (Micro view)"),
    ("AI 검색 챗봇 및 최신 뉴스", "감사 정보 AI 검색 및 최신 뉴스"),
]

if "menu_select" not in st.session_state:
    st.session_state["menu_select"] = "Home"

def render_nav_bar():
    """각 섹션 최상단에 메뉴 내비게이션 바 렌더링"""
    current = st.session_state.get("menu_select", "Home")
    cols = st.columns(len(MENU_OPTIONS))
    for i, (label, key) in enumerate(MENU_OPTIONS):
        with cols[i]:
            is_active = (key == current)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{i}", use_container_width=True, type=btn_type):
                if key != current:
                    st.session_state["menu_select"] = key
                    st.rerun()

menu = st.session_state["menu_select"]

# ★ 사이드바: 로고 + 외부 링크
st.sidebar.image("PRISM_logo.png", use_container_width=True)
st.sidebar.markdown("**🔗 주요 링크 바로가기**")
link_data = [
    ("감사원", "https://www.bai.go.kr/bai/"),
    ("공공감사", "https://www.pap.go.kr/"),
    ("알리오", "https://www.alio.go.kr/main.do"),
    ("기획재정부", "https://www.moef.go.kr/"),
    ("열린재정", "https://www.openfiscaldata.go.kr/op/ko/index"),
    ("블랙엣지뉴스", "https://blackedge.news/"),
]
for label, url in link_data:
    st.sidebar.link_button(label, url, use_container_width=True)

# =============================================================
# HOME
# =============================================================
if menu == "Home":
    render_nav_bar()

    # 타이틀 + 로그인 버튼 (우측 상단)
    h_left, h_right = st.columns([5, 1])
    with h_left:
        st.title("PRISM 감사 리스크 인텔리전스 대시보드")
        st.caption("핵심 KPI와 알림을 한눈에 파악하는 진입 화면")
    with h_right:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐 Login", use_container_width=True):
            st.session_state["show_login"] = True

    # 로그인 팝업 (st.dialog)
    @st.dialog("🔐 로그인 (후속 개발 아이디어)")
    def login_dialog():
        st.info("추후 관련 정부처와의 협업을 통해 로그인 시 소속 기관 자동 인식되어 개인화 대시보드를 제공하기 위한 아이디어입니다.")
        st.text_input("🏢 소속기관", placeholder="예: 한국토지주택공사")
        st.text_input("🪪 사원번호", placeholder="예: EMP-20210001")
        st.text_input("👤 이름", placeholder="이정훈")
        st.text_input("🔑 비밀번호", type="password")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("로그인 (미구현)", use_container_width=True, disabled=True):
                pass
        with c2:
            if st.button("닫기", use_container_width=True):
                st.rerun()

    if st.session_state.get("show_login", False):
        st.session_state["show_login"] = False
        login_dialog()

    st.divider()

    if not df.empty:
        # ── HOME 기관 필터 ──
        hf1, hf2 = st.columns(2)
        with hf1:
            all_oc_home = sorted(df['org_category'].dropna().unique().tolist())
            home_oc = st.multiselect("🏛️ 기관 유형", ["전체"]+all_oc_home, default=["전체"], key="home_oc")
            home_oc_sel = all_oc_home if "전체" in home_oc else home_oc
        with hf2:
            avail_orgs_home = sorted(df[df['org_category'].isin(home_oc_sel)]['agency_category'].dropna().unique().tolist()) if home_oc_sel else []
            home_org = st.multiselect("🏢 기관 (상세)", ["전체"]+avail_orgs_home, default=["전체"], key="home_org")
            home_org_sel = [] if "전체" in home_org else home_org

        home_df = df.copy()
        if home_oc_sel and len(home_oc_sel) < len(all_oc_home):
            home_df = home_df[home_df['org_category'].isin(home_oc_sel)]
        if home_org_sel:
            home_df = home_df[home_df['agency_category'].isin(home_org_sel)]

        latest_year = home_df['year'].dropna().max()
        prev_year = latest_year - 1 if pd.notna(latest_year) else None
        cur_df = home_df[home_df['year'] == latest_year] if pd.notna(latest_year) else home_df
        prev_df = home_df[home_df['year'] == prev_year] if prev_year else pd.DataFrame()

        cur_cnt = len(cur_df)
        prev_cnt = len(prev_df) if not prev_df.empty else 0
        yoy_pct = ((cur_cnt - prev_cnt) / prev_cnt * 100) if prev_cnt > 0 else 0
        avg_sev = cur_df['disposition_severity'].mean() if not cur_df.empty else 0

        org_scores = []
        for o in cur_df['agency_category'].unique():
            od = cur_df[cur_df['agency_category'] == o]
            c = len(od); s = od['disposition_severity'].mean()
            cc = od['risk_category'].value_counts(); r = (cc >= 3).sum() / max(len(cc), 1) * 100
            org_scores.append({'cnt': c, 'sev': s, 'rep': r})
        if org_scores:
            osd = pd.DataFrame(org_scores)
            for col in ['cnt', 'sev', 'rep']:
                mx = osd[col].max(); osd[f'{col}_n'] = (osd[col] / mx * 100) if mx > 0 else 0
            osd['score'] = (osd['cnt_n'] * 0.4 + osd['sev_n'] * 0.4 + osd['rep_n'] * 0.2)
            avg_risk = osd['score'].mean()
        else:
            avg_risk = 0

        # ── KPI 카드 4종 (이미지 참고 디자인) ──
        yoy_delta = cur_cnt - prev_cnt
        kpi_data = [
            {"label": f"지적 건수 ({int(latest_year) if pd.notna(latest_year) else '?'}년)",
             "value": f"{cur_cnt:,}건", "icon": "📋",
             "delta": f"{yoy_delta:+,}건 vs 전년" if prev_cnt > 0 else "기준년 없음",
             "positive": yoy_delta <= 0, "color": "#FF6B6B"},
            {"label": "평균 리스크 점수",
             "value": f"{avg_risk:.1f}점", "icon": "⚡",
             "delta": "40% 건수 + 40% 강도 + 20% 반복",
             "positive": avg_risk < 50, "color": "#FFD93D"},
            {"label": "전년 대비 증감율",
             "value": f"{yoy_pct:+.1f}%", "icon": "📈",
             "delta": f"{'감소 추세' if yoy_pct < 0 else '증가 추세'}",
             "positive": yoy_pct <= 0, "color": "#4ECDC4"},
            {"label": "평균 처분 강도",
             "value": f"{avg_sev:.1f}점", "icon": "⚖️",
             "delta": "파면(10)~현지조치(5) 기준",
             "positive": avg_sev < 7, "color": "#a29bfe"},
        ]

        kc = st.columns(4)
        for i, kpi in enumerate(kpi_data):
            delta_cls = "positive" if kpi["positive"] else "negative"
            delta_arrow = "▼ " if kpi["positive"] else "▲ "
            kc[i].markdown(
                f'<div class="stat-card">'
                f'<div class="stat-card-header">{kpi["label"]}'
                f'<span class="stat-icon" style="background:{kpi["color"]};">{kpi["icon"]}</span></div>'
                f'<div class="stat-card-body">'
                f'<div class="stat-value">{kpi["value"]}</div>'
                f'<span class="stat-delta {delta_cls}">{delta_arrow}{kpi["delta"]}</span>'
                f'</div></div>', unsafe_allow_html=True)

        # 데이터 소스 요약 + 처분강도 기준 안내
        st.caption("📌 데이터 소스: 감사원/ALIO, 분석 기간: 2021년 1월~2024년 6월")
        st.caption("ℹ️ 평균 처분 강도: 중징계(10점)~현지조치(5점) 강도점수 기준. "
                   "상세 기준은 [처분 분석 Micro] > 위반×처분 히트맵 탭 > 💡 처분 수위 분류 기준 상세 보기 참조")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 뉴스 피드: 3행 2열 카드 그리드 (6건) ──
        st.subheader("📰 감사·내부통제 최신 뉴스")
        st.caption("출처: 블랙엣지뉴스 Audit News · 기획 가안 (추후 라이브 연동 예정)")
        news_items = BLACKEDGE_NEWS[:6]
        news_html = '<div class="news-grid">'
        for news in news_items:
            tag_color = RISK_TAG_COLORS.get(news["tag"], "#888")
            r, g, b = int(tag_color[1:3],16), int(tag_color[3:5],16), int(tag_color[5:7],16)
            tag_html = f'<span style="background:rgba({r},{g},{b},0.2);color:{tag_color};font-size:10px;padding:1px 6px;border-radius:8px;font-weight:600;">{news["tag"]}</span>'
            news_html += (
                f'<div class="news-grid-card">'
                f'<p class="ng-title"><a href="{news["url"]}" target="_blank">{news["title"]}</a></p>'
                f'<p class="ng-summary">{news.get("summary","")}</p>'
                f'<p class="ng-meta">📅 {news["date"]}  {tag_html}</p>'
                f'</div>'
            )
        news_html += '</div>'
        st.markdown(news_html, unsafe_allow_html=True)
        st.link_button("🔗 블랙엣지뉴스 전체 보기", "https://blackedge.news/news/list.php?mcode=m765266")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 타임라인 — download_url 하이퍼링크 지원 ──
        st.subheader("🔔 최근 감사 이벤트 타임라인")
        st.caption("최신 감사 결과 및 주요 이벤트 (기관별 최대 1건) · 클릭 시 원문 이동")

        recent_base = home_df.dropna(subset=['date']).sort_values('date', ascending=False)
        if not recent_base.empty:
            diversified = recent_base.groupby('agency_category').head(1).sort_values('date', ascending=False).head(10)
            display_rows = list(diversified.iterrows())
            for i in range(0, len(display_rows), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(display_rows):
                        _, row = display_rows[i + j]
                        with col:
                            date_str = str(row.get('date',''))[:10]
                            org = row.get('agency_category','N/A')
                            title = str(row.get('summary_title','') or row.get('title',''))[:80]
                            atype = row.get('audit_report_type','')
                            disp = row.get('disposition_level','')
                            risk = row.get('risk_category','')
                            org_cat = row.get('org_category','')
                            doc_code = str(row.get('doc_code',''))
                            dl_url = str(row.get('download_url','')) if pd.notna(row.get('download_url','')) else ''
                            title_html = f'<a href="{dl_url}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a>' if dl_url and dl_url.startswith('http') else title
                            link_icon = ' ' if dl_url and dl_url.startswith('http') else ''
                            oc_tag = f'<span class="tag tag-orange" title="기관 유형">🏛️ {org_cat}</span>' if org_cat and org_cat != '미분류' else ''
                            at_tag = f'<span class="tag tag-blue" title="감사 종류">🔍 {atype}</span>' if atype else ''
                            rk_tag = f'<span class="tag tag-green" title="위반 유형">📂 {risk}</span>' if risk else ''
                            dl_tag = f'<span class="tag tag-red" title="처분 유형">📌 {disp}</span>' if disp else ''
                            dc_tag = f'<span class="tag tag-purple" title="문서 번호">📄 {doc_code}</span>' if doc_code and doc_code != 'nan' and doc_code.strip() else ''
                            st.markdown(
                                f'<div class="case-card">'
                                f'<div class="card-header"><span class="card-title">🏢 {org}{link_icon}</span><span class="card-date">📅 {date_str}</span></div>'
                                f'<p class="card-body">{title_html}</p>'
                                f'{oc_tag}{at_tag}{rk_tag}{dl_tag}{dc_tag}'
                                f'</div>', unsafe_allow_html=True)
        else:
            st.info("표시할 이벤트가 없습니다.")
    else:
        st.error("데이터가 없습니다.")


# =============================================================
# ★ 변경5: AI 검색 메뉴 — 준비 중
# =============================================================
elif menu == "감사 정보 AI 검색 및 최신 뉴스":
    render_nav_bar()
    st.title("🤖 감사 정보 AI 검색 및 최신 뉴스")
    st.divider()

    ai_tab, news_tab = st.tabs(["🔍 AI 검색 (준비 중)", "📰 최신 뉴스"])

    with ai_tab:
        st.info("🚧 **RAG 기반 감사 자료 AI 검색 챗봇**이 준비 중입니다.")
        st.markdown("대시보드의 **AI 추천 키워드**를 아래 검색창에 붙여넣어 유사 사례를 심층 탐색할 수 있습니다.")

        # 가안: 검색 UI 프로토타입
        search_query = st.text_input("🔎 감사 사례 검색", placeholder="예: 경징계 재무/회계/계약 종합감사")
        if search_query:
            st.warning(f"'{search_query}' 검색 기능은 RAG 엔진 연동 후 활성화됩니다.")
            # 키워드 매칭 미리보기 (기존 데이터 내 간이 검색)
            mask = df.apply(lambda r: search_query.replace(" ", "") in
                (str(r.get('summary_title',''))+str(r.get('action',''))+str(r.get('risk_category',''))+str(r.get('disposition_level',''))).replace(" ",""),
                axis=1)
            preview = df[mask].head(6)
            if not preview.empty:
                st.markdown(f"**📋 키워드 매칭 미리보기** ({len(preview)}건)")
                for _, row in preview.iterrows():
                    org = row.get('agency_category', 'N/A')
                    title = str(row.get('summary_title','') or row.get('title',''))[:70]
                    risk = row.get('risk_category', '')
                    disp = row.get('disposition_level', '')
                    date_str = str(row.get('date', ''))[:10]
                    st.markdown(
                        f'<div class="news-card">'
                        f'<p class="news-title">🏢 {org} — {title}</p>'
                        f'<p class="news-meta">📅 {date_str}  📂 {risk}  📌 {disp}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("키워드 매칭 결과 없음 — RAG 검색 연동 시 자연어 검색이 가능합니다.")

    with news_tab:
        st.markdown(
            '<div class="news-section-header">'
            '<span style="font-size:18px;font-weight:700;">감사·내부통제 최신 뉴스</span>'
            '<span class="news-badge">BlackEdge News</span>'
            '<span class="news-badge" style="background:linear-gradient(135deg,#e74c3c,#c0392b);">기획 가안</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.caption("출처: [블랙엣지뉴스](https://blackedge.news/) Audit News 섹션 · 추후 RSS/크롤링 라이브 연동 예정")
        st.markdown("")

        # 카테고리 필터
        all_tags = sorted(set(n["tag"] for n in BLACKEDGE_NEWS))
        tag_filter = st.multiselect("📂 리스크 카테고리 필터", all_tags, default=all_tags, key="news_tag_filter")
        filtered_news = [n for n in BLACKEDGE_NEWS if n["tag"] in tag_filter]

        # 뉴스 카드 렌더링 (전체)
        for news in filtered_news:
            tag_color = RISK_TAG_COLORS.get(news["tag"], "#888")
            r, g, b = int(tag_color[1:3],16), int(tag_color[3:5],16), int(tag_color[5:7],16)
            tag_html = f'<span style="background:rgba({r},{g},{b},0.2);color:{tag_color};font-size:11px;padding:2px 8px;border-radius:8px;font-weight:600;">{news["tag"]}</span>'
            st.markdown(
                f'<div class="news-card">'
                f'<p class="news-title"><a href="{news["url"]}" target="_blank">{news["title"]}</a></p>'
                f'<p class="news-meta">📅 {news["date"]}  {tag_html}</p>'
                f'<p class="news-summary">{news["summary"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.link_button("🔗 블랙엣지뉴스 Audit News", "https://blackedge.news/news/list.php?mcode=m765266", use_container_width=True)
        with col_link2:
            st.link_button("📋 블랙엣지뉴스 감사·지적 사례", "https://blackedge.news/news/list.php?mcode=m79nfq1", use_container_width=True)

        # 향후 연동 가이드
        with st.expander("ℹ️ 라이브 연동 계획"):
            st.markdown("""
**현재 상태:** 블랙엣지뉴스 Audit News 헤드라인 12건을 샘플 데이터로 하드코딩한 기획 가안입니다.

**라이브 연동 방안 (우선순위순):**
1. **RSS 피드 요청** — 블랙엣지 측에 RSS 활성화 요청 (그누보드 관리자 설정에서 가능)
2. **헤드라인 크롤링** — `requests + BeautifulSoup`으로 제목/날짜/링크 파싱 (저작권 허락 필요)
3. **Google News RSS 보완** — `news.google.com/rss/search?q=감사원+내부통제` 키워드 뉴스 병행

**교체 시 수정 포인트:** `BLACKEDGE_NEWS` 리스트를 `fetch_blackedge_news()` 함수 호출로 교체
""")


# =============================================================
# EPIC-02: 감사 트렌드
# =============================================================
elif menu == "감사 트렌드":
    render_nav_bar()
    st.title("감사 트렌드 분석")
    st.caption('페르소나 시나리오: "요즘 감사 트렌드가 뭐지?" - 시계열 차트 + Top-N 랭킹 차트')
    st.divider()

    if df.empty:
        st.error("데이터가 없습니다.")
    else:
        st.subheader("데이터 컨트롤 패널(트렌드 섹션 전체 적용)")
        vd = df['date'].dropna()
        mn_d = vd.min().date() if not vd.empty else datetime(2020,1,1).date()
        mx_d = vd.max().date() if not vd.empty else datetime(2024,12,31).date()

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        with fc1:
            st.markdown("**📂 데이터 소스**")
            sel_sites_raw = st.multiselect("src", ["전체"]+sorted(df['site'].dropna().unique().tolist()), default=["전체"], label_visibility="collapsed", key="t_s")
            sel_sites = sorted(df['site'].dropna().unique().tolist()) if "전체" in sel_sites_raw else sel_sites_raw
        with fc2:
            st.markdown("**📅 기간**")
            pp = st.selectbox("p",["전체","최근 1년","최근 2년","직접 설정"], label_visibility="collapsed", key="t_p")
            if pp=="최근 1년": ds,de = pd.Timestamp(mx_d)-pd.DateOffset(years=1), pd.Timestamp(mx_d)
            elif pp=="최근 2년": ds,de = pd.Timestamp(mx_d)-pd.DateOffset(years=2), pd.Timestamp(mx_d)
            elif pp=="직접 설정":
                dr = st.date_input("r",[mn_d,mx_d],min_value=mn_d,max_value=mx_d,key="t_dr")
                ds,de = (pd.Timestamp(dr[0]),pd.Timestamp(dr[1])) if len(dr)==2 else (pd.Timestamp(mn_d),pd.Timestamp(mx_d))
            else: ds,de = pd.Timestamp(mn_d), pd.Timestamp(mx_d)
        with fc3:
            st.markdown("**🔍 감사 유형**")
            all_at_t = sorted(df['audit_report_type'].unique().tolist())
            sel_at_raw = st.multiselect("at", ["전체"]+all_at_t, default=["전체"], label_visibility="collapsed", key="t_at")
            sel_at = all_at_t if "전체" in sel_at_raw else sel_at_raw
        with fc4:
            st.markdown("**🏛️ 기관 유형**")
            all_org_cats_t = sorted(df['org_category'].dropna().unique().tolist())
            sel_oc_raw = st.multiselect("oc", ["전체"]+all_org_cats_t, default=["전체"], label_visibility="collapsed", key="t_oc")
            sel_org_cat = all_org_cats_t if "전체" in sel_oc_raw else sel_oc_raw
        with fc5:
            st.markdown("**🏢 기관 (상세)**")
            # 연쇄 필터: 선택된 기관유형에 해당하는 기관만 표시
            available_orgs_t = sorted(df[df['org_category'].isin(sel_org_cat)]['agency_category'].dropna().unique().tolist()) if sel_org_cat else []
            sel_org_raw = st.multiselect("org", ["전체"]+available_orgs_t, default=["전체"], label_visibility="collapsed", key="t_org")
            sel_org = [] if "전체" in sel_org_raw else sel_org_raw

        fdf = df.copy()
        if sel_sites and "전체" not in sel_sites_raw: fdf = fdf[fdf['site'].isin(sel_sites)]
        fdf = fdf[(fdf['date']>=ds)&(fdf['date']<=de)]
        if sel_at and "전체" not in sel_at_raw: fdf = fdf[fdf['audit_report_type'].isin(sel_at)]
        if sel_org_cat and "전체" not in sel_oc_raw: fdf = fdf[fdf['org_category'].isin(sel_org_cat)]
        if sel_org: fdf = fdf[fdf['agency_category'].isin(sel_org)]

        kc1,kc2,kc3,kc4,kc5 = st.columns(5)
        kc1.metric("📋 총 지적 건수",f"{len(fdf):,}건"); kc2.metric("🏢 기관 수",f"{fdf['agency_category'].nunique():,}개")
        kc3.metric("📁 감사 유형 수",f"{fdf['audit_report_type'].nunique()}개"); kc4.metric("🏛️ 기관 유형 수",f"{fdf['org_category'].nunique()}개"); kc5.metric("📅 기간",f"{ds.strftime('%Y.%m')}~{de.strftime('%Y.%m')}")
        st.caption("📌 감사 유형, 기관 유형은 감사원/ALIO의 보고서명 및 공공기관 지정 현황을 준수하여 분류하였습니다. * ALIO 공공기관 현황: https://alio.go.kr/guide/publicAgencyStatus.do")
        st.divider()

        if fdf.empty:
            st.warning("조건에 해당하는 데이터가 없습니다.")
        else:
            c1, c2 = st.columns(2)

            # 시계열 — 체크박스 1행 + audit_report_type 드롭다운
            with c1:
                st.subheader("1. 지적 건수 시계열")
                ts_opt1, ts_opt2 = st.columns([2, 1])
                with ts_opt1:
                    ts_period = st.radio("기간 단위", ["월별", "분기별", "연도별"], horizontal=True, key="ts_period")
                with ts_opt2:
                    ts_bytype = st.checkbox("유형별 분리", True, key="ts_bytype")

                at_list = sorted(fdf['audit_report_type'].unique().tolist())
                ts_at_filter = st.selectbox("감사 유형 필터", ["전체"] + at_list, key="ts_at_filter")

                ts_data = fdf.copy()
                if ts_at_filter != "전체":
                    ts_data = ts_data[ts_data['audit_report_type'] == ts_at_filter]

                if ts_period == "분기별": tc = 'year_quarter'
                elif ts_period == "연도별": tc = 'year'
                else: tc = 'year_month'

                if not ts_data.empty:
                    if ts_bytype and ts_at_filter == "전체":
                        td = ts_data.groupby([tc,'audit_report_type']).size().reset_index(name='건수').sort_values(tc)
                        td[tc] = td[tc].astype(str)
                        fig = px.line(td,x=tc,y='건수',color='audit_report_type',markers=True,
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                    else:
                        td = ts_data.groupby(tc).size().reset_index(name='건수').sort_values(tc)
                        td[tc] = td[tc].astype(str)
                        fig = go.Figure(go.Scatter(x=td[tc],y=td['건수'],mode='lines+markers',
                                                   line=dict(color='#4ECDC4',width=3)))
                    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10),
                        legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center"), hovermode='x unified')
                    fig.update_xaxes(tickangle=-45, automargin=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("선택 조건에 해당하는 데이터가 없습니다.")

            # 시즈널리티 히트맵 — 연도 드롭다운
            with c2:
                st.subheader("2. 시즈널리티 히트맵")
                hm_years = sorted(fdf['year'].dropna().astype(int).unique().tolist(), reverse=True)
                hm_year_sel = st.selectbox("연도 필터", ["전체"] + hm_years, key="hm_year_filter")

                hdf = fdf.dropna(subset=['month']).copy()
                hdf['month'] = hdf['month'].astype(int)
                if hm_year_sel != "전체":
                    hdf = hdf[hdf['year'] == hm_year_sel]

                if not hdf.empty:
                    hp = hdf.groupby(['audit_report_type','month']).size().reset_index(name='건수')
                    hp = hp.pivot_table(index='audit_report_type',columns='month',values='건수',fill_value=0)
                    for m in range(1,13):
                        if m not in hp.columns: hp[m]=0
                    hp = hp[sorted(hp.columns)]

                    fig2 = go.Figure(go.Heatmap(
                        z=hp.values, x=[f"{m}월" for m in hp.columns], y=hp.index.tolist(),
                        colorscale='YlOrRd', text=hp.values, texttemplate='%{text}',
                        textfont=dict(size=11),
                        hovertemplate='<b>%{y}</b> %{x}: %{z}건<extra></extra>',
                        colorbar=dict(title="건수")
                    ))
                    fig2.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                    fig2.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig2, use_container_width=True)
                    with st.expander("📋 상세 테이블"):
                        dp = hp.copy()
                        dp.columns = [f"{m}월" for m in dp.columns]
                        dp['합계'] = dp.sum(axis=1)
                        st.dataframe(dp, use_container_width=True)
                else:
                    st.warning("선택 조건에 해당하는 데이터가 없습니다.")

            st.markdown("<br>",unsafe_allow_html=True); st.divider()

            c3,c4 = st.columns(2)
            with c3:
                st.subheader("3. Top-N 랭킹")
                tn = st.slider("상위 N",5,30,10,5,key="tn")
                ork = fdf.groupby('agency_category').size().reset_index(name='건수').sort_values('건수',ascending=True).tail(tn)
                fig3 = go.Figure(go.Bar(x=ork['건수'],y=ork['agency_category'],orientation='h',
                    text=ork['건수'],texttemplate='%{text:,}건',textposition='auto',
                    textfont=dict(size=10),
                    marker=dict(color=ork['건수'],colorscale='Tealgrn',showscale=False)))
                fig3.update_layout(title=dict(text=f"기관별 Top {tn}",font=dict(size=14)),
                    height=max(350,tn*30),margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig3,use_container_width=True)
                st.markdown("---"); st.markdown("**감사 유형별 구성**")
                trk = fdf.groupby('audit_report_type').size().reset_index(name='건수').sort_values('건수',ascending=False)
                figd = go.Figure(go.Pie(labels=trk['audit_report_type'],values=trk['건수'],hole=0.45,
                    textinfo='label+percent',textposition='auto',
                    marker=dict(colors=px.colors.qualitative.Pastel)))
                figd.update_layout(height=350,margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
                st.plotly_chart(figd,use_container_width=True)

            with c4:
                st.subheader("4. 전년 대비(YoY) 증감")
                ydf = fdf.dropna(subset=['year']).copy(); ydf['year']=ydf['year'].astype(int)
                if not ydf.empty:
                    yc = ydf.groupby('year').size().reset_index(name='건수').sort_values('year')
                    yc['prev']=yc['건수'].shift(1); yc['yoy']=((yc['건수']-yc['prev'])/yc['prev']*100).round(1)
                    ych = yc.dropna(subset=['yoy'])
                    if not ych.empty:
                        ych = ych.copy()
                        ych['clr']=ych['yoy'].apply(lambda x:'#FF6B6B' if x>0 else '#4DABF7')
                        ych['lbl']=ych['yoy'].apply(lambda x:f"+{x:.1f}%" if x>0 else f"{x:.1f}%")
                        figy = go.Figure(go.Bar(x=ych['year'].astype(str),y=ych['yoy'],
                            text=ych['lbl'],textposition='auto',textfont=dict(size=10),
                            marker=dict(color=ych['clr'])))
                        figy.add_hline(y=0,line_dash="dash",line_color="gray")
                        figy.update_layout(height=400,margin=dict(l=10,r=10,t=30,b=10),xaxis=dict(type='category'),showlegend=False)
                        st.plotly_chart(figy,use_container_width=True)
                        st.markdown('<div class="legend-row"><span class="leg-red">● 증가</span><span class="leg-blue">● 감소</span></div>',unsafe_allow_html=True)
                    st.markdown("---"); st.markdown("**연도별 현황**")
                    yd=yc.copy(); yd['year']=yd['year'].astype(str)+'년'
                    st.dataframe(yd.rename(columns={'year':'연도','건수':'지적건수','prev':'전년','yoy':'YoY(%)'})[['연도','지적건수','전년','YoY(%)']],use_container_width=True,hide_index=True)

            # ★ 키워드 클라우드 (감사 트렌드 하단)
            st.markdown("<br>",unsafe_allow_html=True); st.divider()
            st.subheader("5. 감사 핵심 키워드 클라우드")
            st.caption("TF-IDF 가중치 기반 — 특정 분야에 집중된 차별 키워드일수록 크게 표시 (4개+ 분야 공통 범용어 자동 제외)")
            render_keyword_cloud(fdf, section_key="trend_kwc")


# =============================================================
# EPIC-03 Macro
# =============================================================
elif menu == "리스크 - 기관 벤치마크 (Macro view)":
    render_nav_bar()
    st.title("리스크 관리 - 기관 벤치마크 (Macro view)")
    st.caption('페르소나 시나리오: "유사 기관 대비 우리 감사 리스크는?" - 피어그룹 비교 + 재정처분 프로파일링')
    st.divider()

    if df.empty:
        st.error("데이터가 없습니다.")
    else:
        st.subheader("🏢 피어그룹 설정")
        all_org_cats = sorted(df['org_category'].unique().tolist())
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("**🏛️ 기관 유형**")
            org_cat_filter_raw = st.multiselect("oc_bm", ["전체"]+all_org_cats, default=["전체"], label_visibility="collapsed", key="bm_org_cat")
            org_cat_filter = all_org_cats if "전체" in org_cat_filter_raw else org_cat_filter_raw
        with cs2:
            st.markdown("**🏢 기관 (상세)**")
            # 연쇄 필터: 선택된 기관유형에 해당하는 기관만 표시
            available_orgs_bm = sorted(df[df['org_category'].isin(org_cat_filter)]['agency_category'].dropna().unique().tolist()) if org_cat_filter else []
            my_org = st.selectbox("🚩 우리 기관",["선택하세요"]+available_orgs_bm,key="bm_my")

        if my_org == "선택하세요":
            st.info("👆 기관 유형을 선택한 후, 기관(상세)에서 우리 기관을 선택하세요.")
        else:
            # org_category 필터 적용된 기관 목록
            filtered_orgs = available_orgs_bm
            org_stats = df.groupby('agency_category').agg(cnt=('idx','count'),avg_sev=('disposition_severity','mean')).reset_index()
            my_cnt = org_stats.loc[org_stats['agency_category']==my_org,'cnt'].iloc[0] if not org_stats[org_stats['agency_category']==my_org].empty else 0
            auto_peers = org_stats[
                (org_stats['cnt']>=my_cnt*0.5)&(org_stats['cnt']<=my_cnt*1.5)
                &(org_stats['agency_category']!=my_org)
                &(org_stats['agency_category'].isin(filtered_orgs))
            ]['agency_category'].tolist()
            st.markdown(f"자동 추천: 유사 **{len(auto_peers)}**개 기관")
            st.caption(f"📐 추천 기준: {my_org}의 지적건수({my_cnt}건) ±50% 범위 내 기관 (선택된 기관유형 내)")
            peer_candidates = [o for o in available_orgs_bm if o != my_org]
            peer_orgs = st.multiselect("피어그룹 (수정 가능)", peer_candidates, default=auto_peers[:10], key="bm_peers")

            if not peer_orgs:
                st.warning("피어그룹을 1개 이상 선택하세요.")
            else:
                st.divider()
                def calc_risk(sub):
                    c=len(sub); s=sub['disposition_severity'].mean() if not sub.empty else 0
                    cc=sub['risk_category'].value_counts(); r=(cc>=3).sum()/max(len(cc),1)*100
                    return c,s,r

                all_orgs = [my_org]+peer_orgs
                rows=[]
                for o in all_orgs:
                    od=df[df['agency_category']==o]; c,s,r=calc_risk(od)
                    rows.append({'agency_category':o,'지적건수':c,'처분강도':s,'반복비율':r})
                sdf = pd.DataFrame(rows)
                for col in ['지적건수','처분강도','반복비율']:
                    mx=sdf[col].max(); sdf[f'{col}_n']=(sdf[col]/mx*100) if mx>0 else 0
                sdf['리스크점수']=(sdf['지적건수_n']*0.4+sdf['처분강도_n']*0.4+sdf['반복비율_n']*0.2).round(1)
                my_score=sdf[sdf['agency_category']==my_org]['리스크점수'].iloc[0]
                pctl=(sdf['리스크점수']<my_score).sum()/len(sdf)*100
                pavg=sdf[sdf['agency_category']!=my_org]['리스크점수'].mean()
                pmn=sdf[sdf['agency_category']!=my_org]['리스크점수'].min()
                pmx=sdf[sdf['agency_category']!=my_org]['리스크점수'].max()

                st.subheader("1.리스크 점수 벤치마크")
                gc1,gc2,gc3 = st.columns([2,1,1])
                with gc1:
                    fg = go.Figure(go.Indicator(mode="gauge+number+delta",value=my_score,
                        delta={'reference':pavg,'valueformat':'.1f','increasing':{'color':'#FF6B6B'},'decreasing':{'color':'#4DABF7'}},
                        title={'text':f"<b>{my_org}</b><br><span style='font-size:12px;color:gray;'>피어 평균 대비</span>"},
                        number={'font':{'size':48}},
                        gauge={'axis':{'range':[0,100]},'bar':{'color':'#FF4757','thickness':0.3},
                               'steps':[{'range':[0,33],'color':'#E8F8F5'},{'range':[33,66],'color':'#FFF9E6'},{'range':[66,100],'color':'#FDEDEC'}],
                               'threshold':{'line':{'color':'#1E90FF','width':4},'thickness':0.8,'value':pavg}}))
                    fg.update_layout(height=300,margin=dict(l=30,r=30,t=80,b=30))
                    st.plotly_chart(fg,use_container_width=True)
                with gc2:
                    st.markdown(f'<div class="kpi-box"><p class="kpi-label">피어그룹 내 위치</p><p class="kpi-value" style="color:#FF4757;">상위 {100-pctl:.0f}%</p><p class="kpi-sub">{len(peer_orgs)+1}개 기관 중</p></div>',unsafe_allow_html=True)
                with gc3:
                    st.markdown(f'<div class="info-box"><p style="opacity:0.6;font-size:13px;">📌 점수 비교</p><p><b style="color:#FF4757;">우리:</b> {my_score:.1f}점</p><p><b style="color:#1E90FF;">피어 평균:</b> {pavg:.1f}점</p><p>범위: {pmn:.1f}~{pmx:.1f}</p></div>',unsafe_allow_html=True)

                # ★ 변경4: Macro disclaimer를 Micro 6그룹 기준과 통일
                with st.expander("💡 리스크 점수 산정 기준 상세 보기"):
                    st.markdown(f"""
#### 1. 리스크 점수 산출 공식
**리스크 점수 = (지적건수 × 40%) + (처분강도 × 40%) + (반복비율 × 20%)**
*(※ 각 지표는 분석 대상 전체 기관 중 최대값을 100점으로 환산한 상대평가 점수입니다.)*

#### 2. 처분 강도 점수 기준표
{DISPOSITION_DISCLAIMER}

#### 3. 분류 규칙
1. **현지조치 우선**: '현지조치' 또는 '현지시정' 키워드가 있으면 즉시 현지조치로 분류
2. **최고 수위 채택**: 여러 처분 키워드가 동시 언급된 경우, 가장 높은 수위의 그룹을 채택
3. **미분류(기타)**: 어떤 키워드에도 매칭되지 않으면 '기타'로 분류

*(※ 이 분류 기준은 '리스크 - 처분 분석 (Micro view)'의 히트맵 분류 기준과 동일합니다.)*

**리스크 카테고리 분류 방식:**
리스크 카테고리와 처분 수준은 audit_v10 데이터셋에서 사전 분류된 값을 사용합니다.
키워드 클라우드는 TF-IDF 기반 키워드 빈도 분석으로 별도 산출됩니다.
                    """)

                st.divider()

                st.subheader("2. 기관 리스크 분포도")
                ax_opts=['지적건수','처분강도','반복비율','리스크점수']
                sx1,sx2 = st.columns(2)
                xa=sx1.selectbox("X축",ax_opts,0,key="sc_x"); ya=sx2.selectbox("Y축",ax_opts,1,key="sc_y")
                sdf['구분']=sdf['agency_category'].apply(lambda x:'🚩 우리 기관' if x==my_org else '피어그룹')
                sdf['sz']=sdf['구분'].apply(lambda x:20 if x=='🚩 우리 기관' else 10)
                fsc=px.scatter(sdf,x=xa,y=ya,color='구분',size='sz',hover_name='agency_category',
                    color_discrete_map={'🚩 우리 기관':'#FF4757','피어그룹':'#4DABF7'},
                    hover_data={'지적건수':True,'처분강도':':.1f','리스크점수':':.1f','sz':False,'구분':False})
                fsc.update_layout(height=450,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center"))
                st.plotly_chart(fsc,use_container_width=True)

                st.divider()

                st.subheader("3. 연도별 리스크 추이")
                ys_rows=[]
                for o in all_orgs:
                    od=df[df['agency_category']==o]
                    for yr in sorted(od['year'].dropna().unique()):
                        yd=od[od['year']==yr]; c,s,r=calc_risk(yd)
                        ys_rows.append({'agency_category':o,'year':int(yr),'지적건수':c,'처분강도':s,'반복비율':r})
                if ys_rows:
                    ysd=pd.DataFrame(ys_rows)
                    for yr in ysd['year'].unique():
                        m=ysd['year']==yr
                        for col in ['지적건수','처분강도','반복비율']:
                            mx=ysd.loc[m,col].max(); ysd.loc[m,f'{col}_n']=(ysd.loc[m,col]/mx*100) if mx>0 else 0
                    ysd['리스크점수']=(ysd['지적건수_n']*0.4+ysd['처분강도_n']*0.4+ysd['반복비율_n']*0.2).round(1)
                    py=ysd[ysd['agency_category']!=my_org].groupby('year').agg(avg=('리스크점수','mean'),mn=('리스크점수','min'),mx=('리스크점수','max')).reset_index()
                    myy=ysd[ysd['agency_category']==my_org][['year','리스크점수']].sort_values('year')
                    if len(myy)>=2:
                        fb=go.Figure()
                        fb.add_trace(go.Scatter(x=py['year'],y=py['mx'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
                        fb.add_trace(go.Scatter(x=py['year'],y=py['mn'],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(30,144,255,0.15)',name='피어 범위',hoverinfo='skip'))
                        fb.add_trace(go.Scatter(x=py['year'],y=py['avg'],mode='lines+markers',name='피어 평균',line=dict(color='#1E90FF',width=2,dash='dash'),marker=dict(size=6)))
                        fb.add_trace(go.Scatter(x=myy['year'],y=myy['리스크점수'],mode='lines+markers',name=my_org,line=dict(color='#FF4757',width=3),marker=dict(size=10,symbol='diamond')))
                        fb.update_layout(xaxis=dict(title='연도',tickmode='linear',dtick=1,tickformat='d'),yaxis=dict(title='리스크 점수',range=[0,105]),height=400,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center"))
                        st.plotly_chart(fb,use_container_width=True)
                    else:
                        st.info("밴드 차트에 2개년 이상 데이터가 필요합니다.")

                st.divider()

                # 레이더
                st.subheader("4. 기관별 재정처분 프로파일링")
                st.caption("피어그룹과 재정처분 특성 비교")
                penalty_df = df[(df['penalty_type'].notnull())&(df['penalty_type']!='N/A')&(df['penalty_type']!='')].copy()
                if penalty_df.empty:
                    st.warning("유효한 penalty 데이터가 없습니다.")
                else:
                    vd_r = penalty_df['date'].dropna()
                    yr_r = ["전체"] + list(range(vd_r.max().year, vd_r.min().year-1, -1)) if not vd_r.empty else ["전체"]
                    site_r = ["전체"] + sorted(penalty_df['site'].unique().tolist())
                    pen_r = ["전체"] + sorted(penalty_df['penalty_type'].unique().tolist())

                    rc1, rc2, rc3 = st.columns(3)
                    yr5 = rc1.selectbox("기간", yr_r, key="rd_y")
                    sr5 = rc2.selectbox("데이터 소스", site_r, key="rd_s")
                    pr5 = rc3.selectbox("처분 종류", pen_r, key="rd_p")

                    df_profile = penalty_df.copy()
                    if yr5 != "전체": df_profile = df_profile[df_profile['year'] == yr5]
                    if sr5 != "전체": df_profile = df_profile[df_profile['site'] == sr5]
                    if pr5 != "전체": df_profile = df_profile[df_profile['penalty_type'] == pr5]

                    if df_profile.empty:
                        st.warning("선택 조건에 해당하는 데이터가 없습니다.")
                    else:
                        import re
                        def calc_profile(sub):
                            ta=sub['penalty_amount'].sum(); tc=len(sub); aa=ta/tc if tc>0 else 0
                            ic=sub[sub['penalty_target'].str.contains("대내",na=False)].shape[0]
                            pt_str = sub['penalty_type'].astype(str)
                            hc=sub[pt_str.str.contains(r'징벌|과징금|과태료',na=False,flags=re.IGNORECASE)].shape[0]
                            rc=sub[pt_str.str.contains(r'환수|감액|공제',na=False,flags=re.IGNORECASE)].shape[0]
                            high_threshold = sub['penalty_amount'].quantile(0.75) if tc >= 4 else sub['penalty_amount'].median()
                            high_c = (sub['penalty_amount'] >= high_threshold).sum() if high_threshold > 0 else 0
                            return pd.Series({
                                '총금액':ta,'총건수':tc,'건당단가':aa,
                                '대내비중':(ic/tc*100) if tc>0 else 0,
                                '징벌비중':(hc/tc*100) if tc>0 else 0,
                                '고액비중':(high_c/tc*100) if tc>0 else 0,
                                '감액비중':(rc/tc*100) if tc>0 else 0
                            })

                        pf_stats = df_profile.groupby('agency_category').apply(calc_profile).reset_index()
                        valid_orgs = sorted(pf_stats['agency_category'].unique())
                        org_opts = ["선택 안함"] + valid_orgs

                        oc1, oc2, oc3 = st.columns(3)
                        sel_my = oc1.selectbox("🚩 우리 기관", org_opts, key="rd_my")
                        bench_opts1 = ["선택 안함"] + [o for o in valid_orgs if o != sel_my]
                        sel_b1 = oc2.selectbox("🔍 벤치마크 1", bench_opts1, key="rd_b1")
                        bench_opts2 = ["선택 안함"] + [o for o in valid_orgs if o != sel_my and o != sel_b1]
                        sel_b2 = oc3.selectbox("🔍 벤치마크 2", bench_opts2, key="rd_b2")

                        if sel_my != "선택 안함":
                            cats = ['건당단가','대내비중','징벌비중','고액비중','감액비중']
                            mx_vals = pf_stats[cats].max()
                            def norm_scores(org):
                                r = pf_stats[pf_stats['agency_category']==org].iloc[0]
                                sc = [(r[c]/mx_vals[c]*100) if mx_vals[c]>0 else 0 for c in cats]
                                rv = []
                                for c in cats:
                                    v = r[c]
                                    if c=='건당단가': rv.append(f"{v/1e6:,.1f}백만원")
                                    elif '비중' in c: rv.append(f"{v:.1f}%")
                                    else: rv.append(f"{v:,.0f}")
                                return sc, rv

                            figr = go.Figure()
                            figr.add_trace(go.Scatterpolar(r=[100]*(len(cats)+1),theta=cats+[cats[0]],mode='lines',line=dict(color='silver',width=2),hoverinfo='skip',showlegend=False))
                            colors = [('#FF4757','rgba(255,71,87,0.1)'),('#1E90FF','rgba(30,144,255,0.05)'),('#2ecc71','rgba(46,204,113,0.05)')]
                            radar_list = [sel_my]
                            if sel_b1 != "선택 안함": radar_list.append(sel_b1)
                            if sel_b2 != "선택 안함": radar_list.append(sel_b2)
                            for i, org in enumerate(radar_list):
                                sc, rv = norm_scores(org)
                                figr.add_trace(go.Scatterpolar(r=sc+[sc[0]],theta=cats+[cats[0]],fill='toself',name=org,
                                    fillcolor=colors[i][1],line=dict(color=colors[i][0],width=5 if i==0 else 3),
                                    text=rv+[rv[0]],hovertemplate='<b>%{theta}</b><br>점수:%{r:.1f}<br>실제:%{text}<extra></extra>'))
                            figr.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],showline=False,tickfont=dict(size=10,color="gray"),gridcolor='rgba(128,128,128,0.5)'),
                                angularaxis=dict(gridcolor='rgba(128,128,128,0.5)'),gridshape='linear',bgcolor='rgba(0,0,0,0)'),
                                paper_bgcolor='rgba(0,0,0,0)',height=500,margin=dict(l=50,r=50,t=30,b=50),legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center"))
                            st.plotly_chart(figr, use_container_width=True)

                            st.info("""
                            💡 **5축 범례** (CV 기반 선별)
                            - **건당단가:** 총금액÷총건수, 1건 평균 재무 강도
                            - **대내비중:** 임직원/기관 내부 대상 처분 비율
                            - **징벌비중:** 과징금/과태료 등 징벌적 처분 비율
                            - **고액비중:** 상위 25% 금액 기준 고액 처분 비율
                            - **감액비중:** 환수/감액/공제 등 재정 보전 조치 비율
                            """)
                        else:
                            st.info("👆 '우리 기관'을 선택하면 프로파일링이 시작됩니다.")

                # ★ Macro AI 추천 키워드
                my_df = df[df['agency_category'] == my_org]
                if not my_df.empty:
                    render_ai_search_cards(my_df, df, context_label=my_org, section_key="macro_ai")


# =============================================================
# EPIC-03 Micro: 처분 분석
# =============================================================
elif menu == "리스크 - 처분 분석 (Micro view)":
    render_nav_bar()
    st.title("리스크 관리 - 처분 분석 (Micro view)")
    st.caption('페르소나 시나리오: "이런 위반에 어떤 처분?" - 위반×처분 히트맵 + 비위유형 징계 + 재정처분')
    st.divider()

    if df.empty:
        st.error("데이터가 없습니다.")
    else:
        tab1, tab2, tab3 = st.tabs(["위반×처분 히트맵", "비위 유형별 징계 현황", "재정적 처분 분석"])

        # TAB 1 — disposition_level 기준 6그룹 + 드릴다운
        with tab1:
            st.subheader("1. 위반 유형 × 처분 수위 히트맵")
            vd = df['date'].dropna()
            mn_d = vd.min().date() if not vd.empty else datetime(2020,1,1).date()
            mx_d = vd.max().date() if not vd.empty else datetime(2024,12,31).date()
            year_opts=["전체"]+sorted(df['year'].dropna().astype(int).unique().tolist(),reverse=True)
            site_opts=["전체"]+sorted(df['site'].unique().tolist())
            rcat_opts=["전체"]+sorted(df['risk_category'].unique().tolist())

            # 1행: 기관 유형 → 기관 (상세)
            mf_r1c1, mf_r1c2 = st.columns(2)
            with mf_r1c1:
                all_oc_m1 = sorted(df['org_category'].dropna().unique().tolist())
                hm_oc = st.multiselect("🏛️ 기관 유형", ["전체"]+all_oc_m1, default=["전체"], key="hm_oc")
                hm_oc_sel = all_oc_m1 if "전체" in hm_oc else hm_oc
            with mf_r1c2:
                avail_orgs_m1 = sorted(df[df['org_category'].isin(hm_oc_sel)]['agency_category'].dropna().unique().tolist()) if hm_oc_sel else []
                hm_org = st.multiselect("🏢 기관 (상세)", avail_orgs_m1, default=[], key="hm_org_sel")
            # 2행: 기간 / 데이터 소스 / 위반 유형
            f1,f2,f3 = st.columns(3)
            yf=f1.selectbox("기간",year_opts,key="hm_y"); sf=f2.selectbox("데이터 소스",site_opts,key="hm_s"); rf=f3.selectbox("위반 유형",rcat_opts,key="hm_r")
            hdf=df.copy()
            if hm_oc_sel and len(hm_oc_sel) < len(all_oc_m1): hdf=hdf[hdf['org_category'].isin(hm_oc_sel)]
            if hm_org: hdf=hdf[hdf['agency_category'].isin(hm_org)]
            if yf!="전체": hdf=hdf[hdf['year']==yf]
            if sf!="전체": hdf=hdf[hdf['site']==sf]
            if rf!="전체": hdf=hdf[hdf['risk_category']==rf]
            if st.checkbox("'기타' 처분 제외",True,key="hm_ex"): hdf=hdf[hdf['disposition_level']!='기타']

            if hdf.empty:
                st.warning("데이터가 없습니다.")
            else:
                # KPI 요약 (히트맵 위)
                tc=hdf.shape[0]; hv=hdf[hdf['disposition_level']=='중징계'].shape[0]
                lt=hdf[hdf['disposition_level']=='경징계'].shape[0]
                ad=hdf[hdf['disposition_level'].isin(['시정','경고/주의','통보','현지조치'])].shape[0]
                k1,k2,k3,k4=st.columns(4)
                k1.metric("📊 총 건수",f"{tc:,}건")
                k2.metric("🔴 중징계",f"{hv:,}건 ({hv/tc*100:.1f}%)" if tc>0 else "0건")
                k3.metric("🟡 경징계",f"{lt:,}건 ({lt/tc*100:.1f}%)" if tc>0 else "0건")
                k4.metric("🟢 행정조치",f"{ad:,}건 ({ad/tc*100:.1f}%)" if tc>0 else "0건")

                st.markdown("<br>", unsafe_allow_html=True)

                # 메인 히트맵: disposition_level 기준 6그룹
                cross=hdf.groupby(['risk_category','disposition_level']).size().reset_index(name='건수')
                dof=[d for d in DISP_ORDER if d in cross['disposition_level'].unique()]
                cp=cross.pivot_table(index='risk_category',columns='disposition_level',values='건수',fill_value=0)
                oc=[c for c in dof if c in cp.columns]; ec=[c for c in cp.columns if c not in oc]; cp=cp[oc+ec]

                fhm=go.Figure(go.Heatmap(z=cp.values,x=cp.columns.tolist(),y=cp.index.tolist(),colorscale='Reds',
                    text=cp.values,texttemplate='%{text}',textfont=dict(size=12),
                    hovertemplate='<b>%{y}</b><br>처분그룹:%{x}<br>%{z}건<extra></extra>',colorbar=dict(title="건수")))
                fhm.update_layout(xaxis_title="처분 수위 그룹 (중징계 ← → 현지조치)",
                    height=max(400,len(cp)*50),margin=dict(l=10,r=10,t=30,b=10))
                fhm.update_yaxes(autorange="reversed")
                st.plotly_chart(fhm,use_container_width=True)

                # 세부 드릴다운: 그룹 선택 → 개별 disposition_level 히트맵
                sel_grp = st.selectbox("🔍 세부 처분 수위 확인 (그룹 선택)", ["선택하세요"]+dof, key="hm_drill_grp")
                if sel_grp != "선택하세요":
                    members = GROUP_MEMBERS.get(sel_grp, [])
                    sub_df = hdf[hdf['disposition_level'].isin(members)]
                    if sub_df.empty:
                        st.info(f"'{sel_grp}' 그룹에 해당하는 세부 데이터가 없습니다.")
                    else:
                        sub_cross = sub_df.groupby(['risk_category','disposition_level']).size().reset_index(name='건수')
                        sub_cp = sub_cross.pivot_table(index='risk_category',columns='disposition_level',values='건수',fill_value=0)
                        sub_cols = [c for c in members if c in sub_cp.columns]
                        sub_extra = [c for c in sub_cp.columns if c not in sub_cols]
                        sub_cp = sub_cp[sub_cols + sub_extra]

                        fig_sub = go.Figure(go.Heatmap(z=sub_cp.values,x=sub_cp.columns.tolist(),y=sub_cp.index.tolist(),
                            colorscale='Blues',text=sub_cp.values,texttemplate='%{text}',textfont=dict(size=12),
                            hovertemplate='<b>%{y}</b><br>처분:%{x}<br>%{z}건<extra></extra>',colorbar=dict(title="건수")))
                        fig_sub.update_layout(title=dict(text=f"📌 {sel_grp} 세부 분포",font=dict(size=14)),
                            height=max(300,len(sub_cp)*45),margin=dict(l=10,r=10,t=40,b=10))
                        fig_sub.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig_sub,use_container_width=True)

                with st.expander("💡 처분 수위 분류 기준 상세 보기"):
                    st.markdown(f"""
#### 히트맵 분류 체계
이 히트맵은 감사 지적 사항의 **위반 유형(Y축)** × **처분 수위 그룹(X축)** 교차 건수를 보여줍니다.

{DISPOSITION_DISCLAIMER}

#### 분류 규칙
1. **현지조치 우선**: '현지조치' 또는 '현지시정' 키워드가 있으면 즉시 현지조치로 분류
2. **최고 수위 채택**: 여러 처분 키워드가 동시 언급된 경우, 가장 높은 수위의 그룹을 채택
3. **미분류(기타)**: 어떤 키워드에도 매칭되지 않으면 '기타'로 분류

*(※ 이 분류 기준은 '리스크 - 기관 벤치마크 (Macro view)'의 리스크 점수 산정에 사용되는 처분강도 기준과 동일합니다.)*

**리스크 카테고리 분류 방식:**
리스크 카테고리와 처분 수준은 audit_v10 데이터셋에서 사전 분류된 값을 사용합니다.
키워드 클라우드는 TF-IDF 기반 키워드 빈도 분석으로 별도 산출됩니다.
                    """)

                st.divider()

                # 처분수위별 Stacked Bar
                st.subheader("2. 리스크 분야별 처분 구성")
                skd=hdf.groupby(['risk_category','disposition_level']).size().reset_index(name='건수')
                fsk=px.bar(skd,x='risk_category',y='건수',color='disposition_level',barmode='stack',
                    category_orders={'disposition_level':['중징계','경징계','시정','경고/주의','통보','현지조치','기타']},
                    color_discrete_map=DISP_COLOR_MAP,text='건수')
                fsk.update_layout(height=400,margin=dict(l=10,r=10,t=30,b=10),
                    legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center"))
                fsk.update_traces(textposition='inside',textfont=dict(size=10))
                st.plotly_chart(fsk,use_container_width=True)

                st.divider()

                # 사례 드릴다운 + 형평성 통계 통합
                st.subheader("3. 사례 드릴다운 및 형평성 분석")
                d1,d2=st.columns(2)
                sr=d1.selectbox("위반 유형",["전체"]+cp.index.tolist(),key="dd_r")
                sd=d2.selectbox("처분 수위 그룹",["전체"]+oc+ec,key="dd_d")
                dcases=hdf.copy()
                if sr!="전체": dcases=dcases[dcases['risk_category']==sr]
                if sd!="전체": dcases=dcases[dcases['disposition_level']==sd]
                filter_label = f"{sr} × {sd}"
                if dcases.empty:
                    st.info(f"'{filter_label}' 해당 사례 없음")
                else:
                    st.markdown(f"**검색 결과: {len(dcases)}건** ({filter_label})")

                    # 형평성 통계 블록
                    render_equity_stats(dcases, section_key="dd")

                # ★ Tab1 AI 추천 키워드
                render_ai_search_cards(hdf, df, context_label=f"{yf}년" if yf!="전체" else "", section_key="micro_t1_ai")

        # TAB 2: 비위 유형별 징계 현황
        with tab2:
            st.subheader("비위 유형별 징계 현황")
            st.caption("X축: 위반 유형(risk_category), Y축: 처분 강도(disposition_level) — Sankey + Grouped Bar")

            vt_c1, vt_c2, vt_c3, vt_c4, vt_c5 = st.columns(5)
            with vt_c1:
                all_oc_vt = sorted(df['org_category'].dropna().unique().tolist())
                vt_oc = st.multiselect("🏛️ 기관 유형", ["전체"]+all_oc_vt, default=["전체"], key="vt_oc")
                vt_oc_sel = all_oc_vt if "전체" in vt_oc else vt_oc
            with vt_c2:
                avail_orgs_vt = sorted(df[df['org_category'].isin(vt_oc_sel)]['agency_category'].dropna().unique().tolist()) if vt_oc_sel else []
                vt_org = st.multiselect("🏢 기관 (상세)", avail_orgs_vt, default=[], key="vt_org_sel")
            with vt_c3:
                vt_year_opts = ["전체"] + sorted(df['year'].dropna().astype(int).unique().tolist(), reverse=True)
                vt_yr = st.selectbox("기간", vt_year_opts, key="vt_yr")
            with vt_c4:
                vt_site_opts = ["전체"] + sorted(df['site'].unique().tolist())
                vt_site = st.selectbox("데이터 소스", vt_site_opts, key="vt_site")
            with vt_c5:
                vt_exclude = st.checkbox("'기타' 처분 제외", True, key="vt_ex")

            vt_df = df.copy()
            if vt_oc_sel and len(vt_oc_sel) < len(all_oc_vt): vt_df = vt_df[vt_df['org_category'].isin(vt_oc_sel)]
            if vt_org: vt_df = vt_df[vt_df['agency_category'].isin(vt_org)]
            if vt_yr != "전체": vt_df = vt_df[vt_df['year'] == vt_yr]
            if vt_site != "전체": vt_df = vt_df[vt_df['site'] == vt_site]
            if vt_exclude: vt_df = vt_df[vt_df['disposition_level'] != '기타']

            if vt_df.empty:
                st.warning("데이터가 없습니다.")
            else:
                st.markdown("#### 1. 위반 유형 → 처분 그룹 흐름도")
                sankey_data = vt_df.groupby(['risk_category', 'disposition_level']).size().reset_index(name='건수')
                risk_cats = sorted(sankey_data['risk_category'].unique().tolist())
                disp_groups = [d for d in DISP_ORDER if d in sankey_data['disposition_level'].unique()]
                all_nodes = risk_cats + disp_groups
                node_idx = {n: i for i, n in enumerate(all_nodes)}

                risk_colors = px.colors.qualitative.Set2[:len(risk_cats)]
                disp_colors_list = [DISP_COLOR_MAP.get(d, '#D3D3D3') for d in disp_groups]
                node_colors = risk_colors + disp_colors_list

                sources, targets, values, link_colors = [], [], [], []
                for _, row in sankey_data.iterrows():
                    if row['risk_category'] in node_idx and row['disposition_level'] in node_idx:
                        sources.append(node_idx[row['risk_category']])
                        targets.append(node_idx[row['disposition_level']])
                        values.append(row['건수'])
                        base = risk_colors[risk_cats.index(row['risk_category'])] if row['risk_category'] in risk_cats else '#ccc'
                        link_colors.append(base.replace(')', ',0.4)').replace('rgb', 'rgba') if 'rgb' in base else f"rgba(150,150,150,0.3)")

                fig_sankey = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(pad=20, thickness=20, label=all_nodes, color=node_colors,
                        hovertemplate='<b>%{label}</b><br>총 %{value}건<extra></extra>'),
                    link=dict(source=sources, target=targets, value=values, color=link_colors,
                        hovertemplate='%{source.label} → %{target.label}<br>%{value}건<extra></extra>')
                ))
                fig_sankey.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_sankey, use_container_width=True)

                st.divider()

                st.markdown("#### 2. 위반 유형별 처분 강도 분포")
                gb_data = vt_df.groupby(['risk_category', 'disposition_level']).size().reset_index(name='건수')
                fig_gb = px.bar(gb_data, x='risk_category', y='건수', color='disposition_level', barmode='group',
                    category_orders={'disposition_level': [d for d in DISP_ORDER if d in gb_data['disposition_level'].unique()]},
                    color_discrete_map=DISP_COLOR_MAP, text='건수')
                fig_gb.update_layout(xaxis_title="위반 유형", yaxis_title="건수", height=450,
                    margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center"))
                fig_gb.update_traces(textposition='auto',textfont=dict(size=9))
                st.plotly_chart(fig_gb, use_container_width=True)

                st.divider()
                st.markdown("#### 3. 위반 유형별 처분 통계 요약")
                vt_stats = vt_df.groupby('risk_category').agg(
                    총건수=('disposition_severity', 'count'),
                    평균강도=('disposition_severity', 'mean'),
                    최대강도=('disposition_severity', 'max'),
                    중징계=('disposition_level', lambda x: (x == '중징계').sum()),
                    경징계=('disposition_level', lambda x: (x == '경징계').sum()),
                    시정=('disposition_level', lambda x: (x == '시정').sum()),
                ).reset_index()
                vt_stats['중징계율(%)'] = (vt_stats['중징계'] / vt_stats['총건수'] * 100).round(1)
                vt_stats['평균강도'] = vt_stats['평균강도'].round(1)
                vt_stats = vt_stats.sort_values('평균강도', ascending=False)
                st.dataframe(vt_stats.rename(columns={'risk_category': '위반 유형'}), use_container_width=True, hide_index=True)

                # ★ Tab2 AI 추천 키워드
                render_ai_search_cards(vt_df, df, context_label="비위유형별", section_key="micro_t2_ai")

        # TAB 3 — 재정적 처분 분석
        with tab3:
            penalty_df = df[(df['penalty_type'].notnull())&(df['penalty_type']!='N/A')&(df['penalty_type']!='')].copy()
            if penalty_df.empty:
                st.error("유효한 벌금 데이터가 없습니다.")
            else:
                # 공통 필터
                vd2=penalty_df['date'].dropna()
                yr_opts2=["전체"]+list(range(vd2.max().year, vd2.min().year-1, -1)) if not vd2.empty else ["전체"]
                site_opts2=["전체"]+sorted(penalty_df['site'].unique().tolist())
                pen_opts2=["전체"]+sorted(penalty_df['penalty_type'].unique().tolist())
                tgt_opts2=["전체"]+sorted(penalty_df['penalty_target'].dropna().unique().tolist())
                cat_opts2=["전체"]+sorted(penalty_df['agency_category'].astype(str).unique().tolist()) if 'agency_category' in penalty_df.columns else ["전체"]

                # 1행: 기관 유형 → 기관 (상세)
                pen_oc1, pen_oc2 = st.columns(2)
                with pen_oc1:
                    all_oc_pen = sorted(penalty_df['org_category'].dropna().unique().tolist()) if 'org_category' in penalty_df.columns else []
                    pen_oc = st.multiselect("🏛️ 기관 유형", ["전체"]+all_oc_pen, default=["전체"], key="pen_oc")
                    pen_oc_sel = all_oc_pen if "전체" in pen_oc else pen_oc
                with pen_oc2:
                    avail_orgs_pen = sorted(penalty_df[penalty_df['org_category'].isin(pen_oc_sel)]['agency_category'].dropna().unique().tolist()) if pen_oc_sel else []
                    pen_org = st.multiselect("🏢 기관 (상세)", avail_orgs_pen, default=[], key="pen_org_sel")
                # 2행: 기간 / 데이터 소스 / 처분종류 / 대상
                pf1, pf2, pf3, pf4 = st.columns(4)
                py_yr = pf1.selectbox("기간", yr_opts2, key="pen_yr_global")
                py_site = pf2.selectbox("데이터 소스", site_opts2, key="pen_site_global")
                py_type = pf3.selectbox("처분종류", pen_opts2, key="pen_type_global")
                py_tgt = pf4.selectbox("대상", tgt_opts2, key="pen_tgt_global")

                pdf = penalty_df.copy()
                if pen_oc_sel and len(pen_oc_sel) < len(all_oc_pen): pdf = pdf[pdf['org_category'].isin(pen_oc_sel)]
                if pen_org: pdf = pdf[pdf['agency_category'].isin(pen_org)]
                if py_yr != "전체": pdf = pdf[pdf['year'] == py_yr]
                if py_site != "전체": pdf = pdf[pdf['site'] == py_site]
                if py_type != "전체": pdf = pdf[pdf['penalty_type'] == py_type]
                if py_tgt != "전체": pdf = pdf[pdf['penalty_target'] == py_tgt]

                if pdf.empty:
                    st.warning("선택 조건에 해당하는 데이터가 없습니다.")
                else:
                    # 전체 KPI
                    total_amt = pdf['penalty_amount_mill'].sum()
                    total_cnt = len(pdf)
                    avg_amt = total_amt / total_cnt if total_cnt > 0 else 0
                    max_single = pdf['penalty_amount_mill'].max()

                    pk1, pk2, pk3, pk4 = st.columns(4)
                    pk1.metric("💰 총 처분액", f"{total_amt:,.0f}백만원")
                    pk2.metric("📋 총 건수", f"{total_cnt:,}건")
                    pk3.metric("📊 건당 평균", f"{avg_amt:,.1f}백만원")
                    pk4.metric("🔝 최대 단건", f"{max_single:,.0f}백만원")

                    st.divider()

                    # Treemap + Bubble
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.subheader("1. 재정처분 구조")
                        tm_data = pdf.groupby(['penalty_type','penalty_target'])['penalty_amount_mill'].agg(['sum','count']).reset_index()
                        tm_data.columns = ['처분유형','대상','금액(백만원)','건수']
                        tm_data = tm_data[tm_data['금액(백만원)'] > 0]
                        if not tm_data.empty:
                            fig_tm = px.treemap(tm_data, path=['처분유형','대상'], values='금액(백만원)',
                                color='금액(백만원)', color_continuous_scale='RdYlGn_r',
                                hover_data={'건수':True})
                            fig_tm.update_layout(height=450, margin=dict(l=10,r=10,t=30,b=10),
                                coloraxis_colorbar=dict(title="백만원"))
                            fig_tm.update_traces(textinfo='label+value', texttemplate='%{label}<br>%{value:,.0f}백만원',
                                textfont=dict(size=10))
                            st.plotly_chart(fig_tm, use_container_width=True)
                        else:
                            st.info("Treemap 표시 가능한 데이터 없음")

                    with pc2:
                        st.subheader("2. 기관별 처분 규모")
                        org_pen = pdf.groupby('agency_category').agg(
                            총금액=('penalty_amount_mill','sum'),
                            건수=('penalty_amount_mill','count'),
                            평균=('penalty_amount_mill','mean'),
                            최대=('penalty_amount_mill','max')
                        ).reset_index()
                        org_pen = org_pen[org_pen['총금액'] > 0].sort_values('총금액', ascending=False).head(20)
                        if not org_pen.empty:
                            fig_bubble = px.scatter(org_pen, x='건수', y='평균',
                                size='총금액', hover_name='agency_category',
                                color='총금액', color_continuous_scale='Reds',
                                hover_data={'총금액':':.0f','건수':True,'평균':':.1f','최대':':.0f'},
                                size_max=50)
                            fig_bubble.update_layout(
                                xaxis_title="처분 건수", yaxis_title="건당 평균 (백만원)",
                                height=450, margin=dict(l=10,r=10,t=30,b=10),
                                coloraxis_colorbar=dict(title="총액(백만)")
                            )
                            st.plotly_chart(fig_bubble, use_container_width=True)
                        else:
                            st.info("데이터 없음")

                    st.divider()

                    pc3, pc4 = st.columns(2)
                    with pc3:
                        st.subheader("3. 연도별 추이 (금액+건수)")
                        if not pdf.empty:
                            ag={'penalty_amount_mill':'sum'}
                            ccn='idx'
                            if 'penalty_idx' in pdf.columns: ag['penalty_idx']='nunique'; ccn='penalty_idx'
                            else: ag['idx']='count'
                            td3=pdf.groupby('year').agg(ag).reset_index(); td3.rename(columns={ccn:'count'},inplace=True)
                            ft=go.Figure()
                            ft.add_trace(go.Bar(x=td3['year'],y=td3['penalty_amount_mill'],name='금액',
                                text=td3['penalty_amount_mill'],texttemplate='%{text:,.0f}',
                                textposition='auto',textfont=dict(size=9),
                                marker_color='#4ECDC4',yaxis='y'))
                            ft.add_trace(go.Scatter(x=td3['year'],y=td3['count'],name='건수',
                                mode='lines+markers',marker=dict(size=10,color='#FF6B6B'),
                                line=dict(width=3,color='#FF6B6B'),yaxis='y2'))
                            ft.update_layout(
                                xaxis=dict(title='연도',tickformat='d'),
                                yaxis=dict(title='백만원',side='left'),
                                yaxis2=dict(title='건수',side='right',overlaying='y',showgrid=False),
                                legend=dict(orientation="h",y=1.1,x=0.5,xanchor='center'),
                                height=420, margin=dict(l=10,r=10,t=30,b=10)
                            )
                            st.plotly_chart(ft,use_container_width=True)

                    with pc4:
                        st.subheader("4. 처분액 규모 분포")
                        if not pdf.empty:
                            bins=[0,100000,1000000,10000000,100000000,1000000000,float('inf')]
                            labels=['~10만','10만~100만','100만~1천만','1천만~1억','1억~10억','10억+']
                            pdf_copy = pdf.copy()
                            pdf_copy['rng']=pd.cut(pdf_copy['penalty_amount'],bins=bins,labels=labels,right=False)
                            ccn4='penalty_idx' if 'penalty_idx' in pdf_copy.columns else 'idx'
                            hd4=pdf_copy.groupby('rng')[ccn4].nunique().reset_index(name='count')
                            amt_by_rng = pdf_copy.groupby('rng')['penalty_amount_mill'].sum().reset_index(name='총액')
                            hd4 = hd4.merge(amt_by_rng, on='rng', how='left')
                            fh4=go.Figure()
                            fh4.add_trace(go.Bar(x=hd4['rng'],y=hd4['count'],name='건수',
                                text=hd4['count'],textposition='auto',textfont=dict(size=10),
                                marker_color='#FF6B6B',yaxis='y'))
                            fh4.add_trace(go.Bar(x=hd4['rng'],y=hd4['총액'],name='총액(백만)',
                                text=hd4['총액'].apply(lambda x: f"{x:,.0f}"),
                                textposition='auto',textfont=dict(size=9),
                                marker_color='#4ECDC4',yaxis='y2',opacity=0.6))
                            fh4.update_layout(
                                xaxis_title="금액 구간",
                                yaxis=dict(title='건수',side='left'),
                                yaxis2=dict(title='총액(백만원)',side='right',overlaying='y',showgrid=False),
                                barmode='group',
                                legend=dict(orientation="h",y=1.1,x=0.5,xanchor='center'),
                                height=420, margin=dict(l=10,r=10,t=30,b=10)
                            )
                            st.plotly_chart(fh4,use_container_width=True)

                    st.divider()

                    # 차트 5 & 6: 2열 배치
                    pc5, pc6 = st.columns(2)
                    with pc5:
                        st.subheader("5. 처분유형 × 대상 교차분석")
                        if not pdf.empty:
                            hd=pdf.groupby(['penalty_type','penalty_target']).size().reset_index(name='count')
                            fh2=px.density_heatmap(hd,x='penalty_target',y='penalty_type',z='count',
                                text_auto=True,color_continuous_scale='Reds')
                            fh2.update_layout(height=400,margin=dict(l=10,r=10,t=30,b=10))
                            fh2.update_yaxes(autorange="reversed")
                            st.plotly_chart(fh2,use_container_width=True)

                            ic=pdf[pdf['penalty_target'].str.contains("대내",na=False)].shape[0]
                            ec2=pdf[pdf['penalty_target'].str.contains("대외",na=False)].shape[0]
                            t2=ic+ec2; ir=(ic/t2*100) if t2>0 else 0; er=(ec2/t2*100) if t2>0 else 0
                            st.markdown(
                                f'<div style="display:flex;justify-content:center;gap:24px;text-align:center;margin-top:8px;">'
                                f'<div class="big-metric"><p class="metric-label">🏢 대내</p><p class="metric-value">{ic}건 <span class="metric-sub">({ir:.1f}%)</span></p></div>'
                                f'<div class="big-metric"><p class="metric-label">🏗️ 대외</p><p class="metric-value">{ec2}건 <span class="metric-sub">({er:.1f}%)</span></p></div>'
                                f'<div class="big-metric"><p class="metric-label">📊 총합</p><p class="metric-value">{t2}건</p></div>'
                                f'</div>', unsafe_allow_html=True)

                    with pc6:
                        st.subheader("6. Top 고액 처분 사례")
                        if not pdf.empty:
                            top_cases = pdf.nlargest(10, 'penalty_amount_mill')[
                                ['agency_category','penalty_type','penalty_target','penalty_amount_mill','date']
                            ].copy()
                            top_cases['date'] = top_cases['date'].dt.strftime('%Y-%m-%d')
                            top_cases['penalty_amount_mill'] = top_cases['penalty_amount_mill'].apply(lambda x: f"{x:,.0f}")
                            top_cases.columns = ['기관','처분유형','대상','금액(백만원)','일자']
                            top_cases = top_cases.reset_index(drop=True)
                            top_cases.index = top_cases.index + 1
                            top_cases.index.name = '순위'
                            st.dataframe(top_cases, use_container_width=True, height=400)

                    # ★ Tab3 AI 추천 키워드
                    render_ai_search_cards(pdf, df, context_label="재정처분", section_key="micro_t3_ai")
