import sys
import os
import streamlit as st
import pandas as pd
import json
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.agentic_rag_v2.modules.vector_retriever import VectorRetriever

@st.cache_resource
def load_retriever():
    return VectorRetriever()

retriever = load_retriever()

# ---------------------------------------------------------
# 0. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="GeniePick Dashboard",
    layout="wide"
)

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    try:
        with open('audit_v10.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ 'audit_v10.json' 파일이 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

    if 'penalty_amount' in df.columns:
        df['penalty_amount'] = pd.to_numeric(df['penalty_amount'], errors='coerce').fillna(0)
        df['penalty_amount_mill'] = df['penalty_amount'] / 1000000

    if 'penalty_type' not in df.columns:
        df['penalty_type'] = "N/A"

    return df

if 'df' not in st.session_state:
    with st.spinner('🚀 데이터 로딩 중...'):
        st.session_state['df'] = load_and_process_data()

df = st.session_state['df']

# ---------------------------------------------------------
# 2. 사이드바 네비게이션
# ---------------------------------------------------------
st.sidebar.title("PRISM Dashboard")
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "Home",
        "감사 트렌드",
        "리스크 관리 - 벤치마크",
        "리스크 관리 - 징계 및 처분 분석",
        "AI 분석 및 보고서 작성"
    ]
)
st.sidebar.divider()

# ---------------------------------------------------------
# 3. 화면 구현
# ---------------------------------------------------------

NODE_NAME_MAP = {
    "router": "질문 분석",
    "chat_worker": "대화 처리",
    "report_manager": "보고서 관리",
    "retrieve_sql": "SQL 검색",
    "field_selector": "필드 선택",
    "hybrid_retriever": "하이브리드 검색",
    "grade_documents": "문서 평가",
    "sop_retriever": "SOP 검색",
    "rewrite_query": "쿼리 재작성",
    "generate": "답변 생성",
    "verify_answer": "답변 검증",
    "summarize_conversation": "대화 요약",
}

PLACEHOLDER_MAP = {
    "사건 제목": "예: OO공사 공공기관 채용 비리 의혹",
    "감사 배경": "예: 내부 제보 접수로 인한 특정 감사 착수",
    "감사 목적": "예: 채용 절차의 공정성 검증 및 위반 사항 적발",
    "감사 방법": "예: 관련 서류 검토 및 관계자 대면 조사",
    "감사 기간": "예: 2023.11.01 ~ 2023.11.15",
    "대상 기관": "예: 한국철도공사",
    "문제점": "예: 채용 점수 조작 및 서류 위조 정황 발견",
}

if menu == "Home":
    st.title("🏠 Home")
    st.info("좌측 메뉴에서 원하는 섹션을 선택해주세요.")

elif menu == "감사 트렌드":
    st.title("📊 감사 트렌드")

elif menu == "리스크 관리 - 벤치마크":
    st.title("🛡️ 리스크 관리 - 벤치마크")

elif menu == "리스크 관리 - 징계 및 처분 분석":
    st.title("리스크 관리 - 징계 및 처분 분석")

# =========================================================
# Section 3. AI 분석
# =========================================================
elif menu == "AI 분석 및 보고서 작성":
    st.title("PRISM AI 분석")
    st.markdown("---")

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "thought_process" not in st.session_state:
        st.session_state["thought_process"] = {}  # {msg_idx: [steps]}
    if "show_report" not in st.session_state:
        st.session_state["show_report"] = False
    if "report_content" not in st.session_state:
        st.session_state["report_content"] = ""
    if "report_state" not in st.session_state:
        st.session_state["report_state"] = "idle"  # idle / checking / missing_info / generating / done
    if "missing_fields" not in st.session_state:
        st.session_state["missing_fields"] = []
    if "user_inputs" not in st.session_state:
        st.session_state["user_inputs"] = {}

    # 레이아웃: 채팅 | 보고서 패널
    if st.session_state["show_report"]:
        chat_col, report_col = st.columns([1, 1])
    else:
        chat_col = st.container()
        report_col = None

    # ── 채팅 영역 ──────────────────────────────────────────
    with chat_col:
        # 보고서 패널 토글 버튼
        col_title, col_btn = st.columns([8, 1])
        with col_btn:
            if st.button("📄", help="보고서 패널 열기/닫기"):
                st.session_state["show_report"] = not st.session_state["show_report"]
                st.rerun()

        # 대화 기록 표시
        for i, msg in enumerate(st.session_state["chat_history"]):
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    # 생각 과정 표시
                    thoughts = st.session_state["thought_process"].get(i, [])
                    if thoughts:
                        with st.expander(f"처리 과정 확인 ({len(thoughts)}단계)", expanded=False):
                            for step in thoughts:
                                node_name = NODE_NAME_MAP.get(step["node"], step["node"])
                                st.markdown(f"**{node_name}**: {step['content']}")
                    st.markdown(msg["content"])
        
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화"):
            st.session_state["chat_history"] = []
            st.session_state["thought_process"] = {}
            st.session_state["report_content"] = ""
            st.session_state["report_state"] = "idle"
            st.rerun()

        # 입력창 (화면 맨 아래 고정, 화살표 버튼 포함)
        user_input = st.chat_input("질문을 입력하세요...")

        if user_input:
            # 사용자 메시지 추가
            st.session_state["chat_history"].append({
                "role": "user",
                "content": user_input
            })
            msg_idx = len(st.session_state["chat_history"])

            with st.spinner("🚀 분석 중..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/chat",
                        json={
                            "query": user_input,
                            "history": st.session_state["chat_history"],
                            "session_id": "prism_session_01"
                        },
                        stream=True,
                        timeout=None
                    )

                    if response.status_code == 200:
                        answer_text = ""
                        thoughts = []

                        for line in response.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    content = decoded.replace("data: ", "")
                                    if content == "[DONE]":
                                        break
                                    try:
                                        json_data = json.loads(content)
                                        if json_data.get("type") == "status" and json_data.get("node"):
                                            thoughts.append({
                                                "node": json_data["node"],
                                                "content": json_data["content"]
                                            })
                                        elif json_data.get("type") == "answer":
                                            answer_text += json_data.get("content", "")
                                        elif json_data.get("type") == "command":
                                            if json_data.get("content") == "open_report":
                                                st.session_state["show_report"] = True
                                    except:
                                        pass

                        if answer_text:
                            st.session_state["chat_history"].append({
                                "role": "assistant",
                                "content": answer_text
                            })
                            st.session_state["thought_process"][msg_idx] = thoughts
                    else:
                        st.error(f"서버 응답 오류: {response.status_code}")

                except Exception as e:
                    st.error(f"연결 실패: {e}")

            st.rerun()

    # ── 보고서 패널 ────────────────────────────────────────
    if st.session_state["show_report"] and report_col:
        with report_col:
            st.markdown("### 📄 Audit Report")
            st.markdown("---")

            report_state = st.session_state["report_state"]

            # 상단 버튼
            col_refresh, col_close = st.columns([8, 1])
            with col_refresh:
                if st.button("🔄 보고서 작성 시작", use_container_width=True,
                             disabled=len(st.session_state["chat_history"]) == 0):
                    st.session_state["report_state"] = "checking"
                    st.rerun()
            with col_close:
                if st.button("✕"):
                    st.session_state["show_report"] = False
                    st.rerun()

            # 상태별 UI
            if report_state == "idle":
                st.info("대화 내용을 바탕으로 보고서를 작성합니다.\n\n위 버튼을 눌러 시작하세요.")

            elif report_state == "checking":
                with st.spinner("필수 정보를 확인하는 중..."):
                    try:
                        history = [{"role": m["role"], "content": m["content"]}
                                   for m in st.session_state["chat_history"]]
                        res = requests.post(
                            "http://localhost:8000/check_report_readiness",
                            json={"query": "Check Readiness", "history": history, "session_id": "auditor_session_01"},
                        )
                        data = res.json()
                        if data.get("status") == "missing_info":
                            st.session_state["missing_fields"] = data.get("missing_fields", [])
                            st.session_state["report_state"] = "missing_info"
                        else:
                            st.session_state["report_state"] = "generating"
                        st.rerun()
                    except Exception as e:
                        st.session_state["report_state"] = "generating"
                        st.rerun()

            elif report_state == "missing_info":
                st.warning("⚠️ 완성도 높은 보고서를 위해 추가 정보를 입력해주세요.")
                for field in st.session_state["missing_fields"]:
                    placeholder = PLACEHOLDER_MAP.get(field, f"{field} 입력...")
                    st.session_state["user_inputs"][field] = st.text_input(
                        field, placeholder=placeholder,
                        value=st.session_state["user_inputs"].get(field, ""),
                        key=f"input_{field}"
                    )
                if st.button("입력 완료 및 보고서 생성", type="primary", use_container_width=True):
                    st.session_state["report_state"] = "generating"
                    st.rerun()

            elif report_state == "generating":
                with st.spinner("📝 보고서 초안을 작성 중입니다..."):
                    try:
                        history = [{"role": m["role"], "content": m["content"]}
                                   for m in st.session_state["chat_history"]]
                        res = requests.post(
                            "http://localhost:8000/generate_report",
                            json={
                                "query": "Generate Report",
                                "history": history,
                                "session_id": "auditor_session_01",
                                "additional_info": st.session_state["user_inputs"]
                            },
                        )
                        data = res.json()
                        st.session_state["report_content"] = data.get("report", "")
                        st.session_state["report_state"] = "done"
                        st.rerun()
                    except Exception as e:
                        st.error(f"보고서 생성 오류: {e}")
                        st.session_state["report_state"] = "idle"

            elif report_state == "done":
                # 복사 버튼
                if st.button("📋 클립보드에 복사", use_container_width=True):
                    st.code(st.session_state["report_content"])
                    st.success("위 내용을 복사해주세요!")

                # 편집 모드
                edited = st.text_area(
                    "보고서 편집",
                    value=st.session_state["report_content"],
                    height=600,
                    key="report_edit"
                )
                if edited != st.session_state["report_content"]:
                    if st.button("💾 저장", use_container_width=True):
                        st.session_state["report_content"] = edited
                        st.success("저장되었습니다!")

                # 미리보기
                with st.expander("📖 미리보기", expanded=True):
                    st.markdown(st.session_state["report_content"])