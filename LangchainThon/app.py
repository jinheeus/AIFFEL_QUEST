# Streamlit (역할 D)

import streamlit as st
from config import settings
from rag.pipeline import build_all
from utils.text import count_chars, detect_repeated_keywords
from utils.logging import add_example



# ----------------------
# 페이지 설정 & 상단 타이틀
# ----------------------
st.set_page_config(page_title="JD 기반 자소서 코칭", layout="wide")

# 세션 상태 초기화 (★ 가장 먼저 한 번만)
if "blocks" not in st.session_state:
    st.session_state.blocks = [{"title": "", "text": "", "target_chars": 0}]
if "report" not in st.session_state:
    st.session_state.report = None
if "essays_raw" not in st.session_state:          # 원본데이터저장
    st.session_state.essays_raw = []
if "show_final" not in st.session_state:         # 최종 수정본 영역 표시 여부
    st.session_state.show_final = False

# 왼쪽상단로고 ###############################
st.sidebar.image("pass_logo.png", width=220)
st.sidebar.markdown("---")  # 로고 아래 구분선
############################################

st.title("JD 기반 자소서 코칭")
st.markdown(
    "<p style='color:#5a2d00; font-size:18px;'>채용공고(JD) 기반 자소서 분석 및 첨삭 리포트</p>",
    unsafe_allow_html=True
)


# ----------------------
# 리포트 렌더링 헬퍼 함수들
# ----------------------

# # 한글 라벨로 변환(1/3)
SCORE_LABELS = {
    "overall": "종합 점수",
    "jd_match": "JD 적합도",
    "consistency": "문항 간 일관성",
    "differentiation": "차별화",
    "clarity": "가독성",
    "specificity": "구체성",
    "structure": "구조/논리",
    "impact": "임팩트/설득력",
}

def _iter_scores_generic(scores):
    """dict이든 Pydantic 모델이든 공통으로 점수 항목을 꺼내기 위한 유틸."""
    if scores is None:
        return []

    # dict 형태인 경우
    if isinstance(scores, dict):
        items = scores.items()
    # Pydantic 모델인 경우(model_dump 사용)
    elif hasattr(scores, "model_dump"):
        data = scores.model_dump()
        items = data.items()
    else:
        return []

    # None / 빈 값은 걸러냄
    return [(k, v) for k, v in items if v not in (None, "")]


def render_jd_structured(jd):
    """JD 구조화 결과를 지원자가 보기 좋게 렌더링."""
    st.subheader("3. JD 분석 결과")

    if getattr(jd, "role_summary", None):
        st.markdown("**포지션 요약**")
        st.write(jd.role_summary)

    if getattr(jd, "responsibilities", None):
        if jd.responsibilities:
            st.markdown("**주요 업무**")
            for r in jd.responsibilities:
                st.write(f"- {r}")

    if getattr(jd, "requirements", None):
        if jd.requirements:
            st.markdown("**필수 자격 요건**")
            for r in jd.requirements:
                st.write(f"- {r}")

    if getattr(jd, "preferred", None):
        if jd.preferred:
            st.markdown("**우대 사항**")
            for p in jd.preferred:
                st.write(f"- {p}")

    if getattr(jd, "core_competencies", None):
        if jd.core_competencies:
            st.markdown("**핵심 역량 키워드**")
            st.write(", ".join(jd.core_competencies))

    if getattr(jd, "tech_stack", None):
        if jd.tech_stack:
            st.markdown("**기술 스택**")
            st.write(", ".join(jd.tech_stack))


def render_overall(report):
    """전체 요약 영역 렌더링."""
    st.subheader("4. 전체 요약")

    scores = getattr(report, "overall_scores", None)
    score_items = _iter_scores_generic(scores)

    if score_items:
        st.markdown("**전체 점수 요약**")
        for k, v in score_items:
            label = SCORE_LABELS.get(k, k)  # 한글 라벨로 변환(2/3)
            st.write(f"- {label}: {v}/100점")
            # 또는 프로그레스 바로 시각화
            # st.progress(v / 10, text=f"{k}: {v}/10점")

    st.markdown("**Top 3 개선 포인트**")
    for x in getattr(report, "overall_top3", []) or []:
        st.write(f"- {x}")

    if getattr(report, "disclaimer", None):
        st.caption(report.disclaimer)


def render_evidence(report):
    """검색된 근거/유사 사례 요약 렌더링."""
    st.subheader("5. 참고한 유사 사례 요약")

    summaries = getattr(report, "retrieved_evidence_summary", []) or []
    if not summaries:
        st.caption("이번 리포트에서 참고한 유사 사례가 없거나 매우 적습니다.")
        return

    with st.expander("AI가 참고한 합격/유사 사례 보기", expanded=False):
        for s in summaries:
            st.write("- ", s)


# def render_per_question(report):
#     """문항별 첨삭 리포트 렌더링."""
#     st.subheader("6. 문항별 첨삭 리포트")

#     per_q = getattr(report, "per_question", []) or []
#     if not per_q:
#         st.caption("문항별 리포트가 없습니다.")
#         return
 
#     for idx, q in enumerate(per_q, start=1):
#         # q가 딕셔너리로 들어오는 경우까지 방어적으로 처리
#         if isinstance(q, dict):
#             title = q.get("question_title") or f"문항 {idx}"
#             scores = q.get("scores")
#             top_improvements = q.get("top_improvements", [])
#             highlights_banned = q.get("highlights_banned", [])
#             edits = q.get("edits", [])
#         else:
#             title = getattr(q, "question_title", None) or f"문항 {idx}"
#             scores = getattr(q, "scores", None)
#             top_improvements = getattr(q, "top_improvements", []) or []
#             highlights_banned = getattr(q, "highlights_banned", []) or []
#             edits = getattr(q, "edits", []) or []

def render_per_question(report):
    """문항별 첨삭 리포트 렌더링."""
    st.subheader("6. 문항별 첨삭 리포트")

    per_q = getattr(report, "per_question", []) or []
    if not per_q:
        st.caption("문항별 리포트가 없습니다.")
        return
    
    for idx, q in enumerate(per_q, start=1):
        # Union에서 str 제거했으므로 단순화
        title = q.question_title or f"문항 {idx}"
        scores = q.scores
        top_improvements = q.top_improvements or []
        highlights_banned = q.highlights_banned or []
        edits = q.edits or []

        with st.expander(f"{idx}. {title}", expanded=(idx == 1)):

            # 점수 요약
            score_items = _iter_scores_generic(scores)
            if score_items:
                st.markdown("**점수 요약**")
                for k, v in score_items:
                    label = SCORE_LABELS.get(k, k)  # 한글 라벨로 변환(3/3)
                    st.write(f"- {label}: {v}점")

            # Top 개선 포인트
            if top_improvements:
                st.markdown("**Top 개선 포인트**")
                for x in top_improvements:
                    st.write(f"- {x}")

            # 금지 패턴
            if highlights_banned:
                st.markdown("**금지 패턴 하이라이트**")
                for h in highlights_banned[:10]:
                    st.warning(h)

            # 문장별 수정 제안
            if edits:
                st.markdown("**문장별 수정 제안 (원문 → 수정 → 이유)**")
                for e in edits[:8]:
                    # dict 또는 객체 모두 처리
                    if isinstance(e, dict):
                        original = e.get("original", "")
                        revised = e.get("revised", "")
                        rationale = e.get("rationale", "")
                        evidence = e.get("evidence", [])
                    else:
                        original = getattr(e, "original", "")
                        revised = getattr(e, "revised", "")
                        rationale = getattr(e, "rationale", "")
                        evidence = getattr(e, "evidence", []) or []

                    st.markdown(f"- 원문: {original}")
                    st.markdown(f"  - 수정: {revised}")
                    st.caption(f"  - 이유: {rationale}")

                    if evidence:
                        # Evidence 객체 처리
                        evidence_strs = []
                        for e in evidence[:3]:
                            # Evidence 객체인 경우
                            if hasattr(e, 'source_type'):
                                icon = {"jd": "📋", "user_text": "✍️", "similar_case": "⭐"}.get(e.source_type, "•")
                                evidence_strs.append(f"{icon} {e.content[:60]}...")
                            # 만약 아직 str인 경우 (하위 호환)
                            else:
                                evidence_strs.append(str(e)[:60])
                        
                        st.caption("  - 참고 근거: " + " | ".join(evidence_strs))

                    # if evidence:
                    #     st.caption("  - 참고 근거: " + " | ".join(evidence[:3]))
            else:
                st.caption("이 문항은 문장 단위 수정 제안 없이 요약 위주로 제공되었습니다.")

def apply_edits_to_text(text: str, edits) -> str:
    """SentenceEdit 리스트를 원문 텍스트에 적용해서 수정본을 만든다."""
    if not text or not edits:
        return text

    new_text = text

    for e in edits:
        # dict / 객체 둘 다 처리
        if isinstance(e, dict):
            original = e.get("original", "")
            revised = e.get("revised", "")
        else:
            original = getattr(e, "original", "")
            revised = getattr(e, "revised", "")

        if not original or not revised:
            continue

        # 가장 단순한 방식: 포함되어 있으면 통째로 교체
        if original in new_text:
            new_text = new_text.replace(original, revised)

    return new_text

# ----------------------
# 사이드바
# ----------------------
with st.sidebar:
    st.header("분석 옵션")
    settings.use_mmr = st.checkbox("검색 결과 다양화 사용", value=False)
    thr_on = st.checkbox("유사도 기준 적용", value=False)
    settings.score_threshold = 0.25 if thr_on else None
    settings.top_k = st.slider("참조 문서 개수", 2, 8, 4)

    st.divider()
    st.header("지원자 정보(선택)")
    user_job = st.text_input("지원 직무 (예: 데이터 분석)", value="")
    user_stack = st.text_input("보유 기술·경험 (예: Python, SQL, ML, 금융)", value="")


# ----------------------
# 1. JD 입력
# ----------------------
st.subheader("1. 채용공고(JD) 입력")
jd_mode = st.radio(
    "입력 방식",
    ["텍스트 직접 입력", "PDF 업로드"],
    horizontal=True,
    label_visibility="collapsed",
)

jd_text = ""
if jd_mode == "텍스트 직접 입력":
    jd_text = st.text_area(
        "JD 내용",
        height=200,
        placeholder="채용공고(JD) 텍스트를 입력하세요.",
        label_visibility="collapsed",
    )
else:
    jd_file = st.file_uploader("JD PDF 업로드", type=["pdf"])
    if jd_file is not None:
        # MVP: 파일에서 텍스트 추출은 별도 구현 가능(시간 없으면 JD 텍스트 붙여넣기만 데모)
        st.warning("MVP에서는 JD PDF 파싱은 생략 가능. 시간이 되면 PyPDF로 텍스트 추출 붙이세요.")


# ----------------------
# 2. 자소서 입력
# ----------------------
st.subheader("2. 자소서 입력")
if "blocks" not in st.session_state:
    # st.session_state.blocks = [{"title": "문항 1", "text": ""}]
    st.session_state.blocks = [{"title": "", "text": "", "target_chars": 0}]  # 변경: 제목 칸에 실제 문항 질문을 입력

colA, colB = st.columns([1, 1])
with colA:
    if st.button("+ 문항 추가"):
        n = len(st.session_state.blocks) + 1
        # st.session_state.blocks.append({"title": f"문항 {n}", "text": ""})
        st.session_state.blocks.append(
            {"title": "", "text": "", "target_chars": 0}
        )  # 새 문항도 제목(질문) 빈칸으로

with colB:
    if st.button("모든 문항 초기화"):
        # st.session_state.blocks = [{"title": "문항 1", "text": "", "target_chars": 0}]
        st.session_state.blocks = [
            {"title": "", "text": "", "target_chars": 0}
        ]  # 초기화 시에도 동일 구조

# target_chars = st.number_input("목표 글자 수(선택)", min_value=0, value=0, step=50)

essays = []
for i, b in enumerate(st.session_state.blocks):
    # 제목(질문)이 비어 있으면 기본 라벨만 보여줌
    heading = b["title"] if b.get("title") else f"문항 {i+1}"
    st.markdown(f"**{i+1}. {heading}**")

    # 문항 질문 아래에 목표 글자 수, 오른쪽에 자소서 내용이 오도록 2컬럼 구성
    col_left, col_right = st.columns([2, 5])

    # 문항 질문 + 목표 글자 수 (왼쪽에 세로 배치)
    with col_left:
        q = st.text_input(
            f"문항 질문 {i+1}",
            key=f"q_{i}",
            value=b.get("title", ""),
            label_visibility="collapsed",
        )

        target = st.number_input(
            "목표 글자 수",
            min_value=0,
            value=b.get("target_chars", 0),
            step=50,
            key=f"target_{i}",
        )

    # 자소서 답변 (오른쪽 전체)
    with col_right:
        t = st.text_area(
            f"내용 (문항 {i+1})",
            key=f"text_{i}",
            height=160,
            value=b.get("text", ""),
        )

    # 세션 상태에 저장
    st.session_state.blocks[i]["title"] = q
    st.session_state.blocks[i]["text"] = t
    st.session_state.blocks[i]["target_chars"] = target

    # 글자 수 표시
    cur = count_chars(t)
    if target > 0:
        st.caption(f"글자 수(공백 제외): {cur} / 목표 {target}")
    else:
        st.caption(f"글자 수(공백 제외): {cur}")

    essays.append(
        {
            "question_title": q,
            "text": t,
            "target_chars": target,
        }
    )

# P1: 중복 내용 경고(간이)
all_texts = [e["text"] for e in essays if e["text"].strip()]
if len(all_texts) >= 2:
    dup = detect_repeated_keywords(all_texts, top_n=10)
    if dup:
        st.info(
            "중복 키워드(간이 감지): "
            + ", ".join([f"{w}({c})" for w, c in dup[:6]])
        )

st.divider()
run = st.button("첨삭 리포트 생성")

if run:
    if not jd_text.strip():
        st.error("JD 텍스트가 필요합니다. (MVP 데모 기준)")
        st.stop()

    if not any(e["text"].strip() for e in essays):
        st.error("자소서 문항 텍스트를 최소 1개 입력해 주세요.")
        st.stop()

    with st.spinner("리포트 생성 중..."):
        report = build_all(
            jd_text=jd_text,
            essays=essays,
            user_job=user_job,
            user_stack=user_stack,
        )
        
    # 🔹 LangSmith Dataset에 이번 케이스를 logging (옵션: dataset_id가 있을 때만)
    if settings.langsmith_dataset_id:
        try:
            add_example(
                dataset_id=settings.langsmith_dataset_id,
                jd_text=jd_text,
                essays=essays,  # [{"question_title":..., "text":..., "target_chars":...}, ...]
                user_job=user_job,
                user_stack=user_stack,
                options={
                    "use_mmr": settings.use_mmr,
                    "top_k": settings.top_k,
                    "score_threshold": settings.score_threshold,
                },
                # 지금은 RAG 원본 문서까지 넘기지는 않고, 필요하면 나중에 확장
                retrieved_docs=None,
                retrieved_patterns=None,
                # FullReport(Pydantic) → dict 로 변환해서 저장
                report_json=report.model_dump(),
            )
        except Exception as e:
            # 로깅 실패해도 서비스는 계속 돌아가게
            st.warning(f"LangSmith 로깅 중 오류가 발생했지만, 리포트 생성에는 영향 없습니다: {e}")



    # 생성한 리포트를 세션에 저장해 두면,
    # 이후 다른 버튼을 눌러도 사라지지 않음
    st.session_state.report = report
    st.session_state.essays_raw = essays      #  원본 자소서도 같이 저장

# ----------- 여기부터는 'report'가 있으면 항상 결과 영역을 보여줌 -----------

report = st.session_state.report

if report is not None:
    st.success("리포트 생성 완료")

    # 가독성 높은 렌더링
    render_jd_structured(report.jd_structured)
    render_overall(report)
    render_evidence(report)
    render_per_question(report)

    # # P1: “수정본 적용” 버튼(뼈대)
    # st.divider()
    # if st.button("수정본 v1 생성(P1 버튼 - 현재는 뼈대)"):
    #     # TODO: 여기에서 report.per_question[*].edits 를 모아서
    #     # 문항별/전체 수정본을 합성하는 로직을 추가하면 됨
    #     st.info("여기서 'edits'를 적용해 문항별 최종 수정본을 생성하는 기능을 붙이면 됩니다.")

    st.divider()

    # 1) 버튼을 누르면 플래그만 켜기
    if st.button("최종 수정본 v1 보기"):
        st.session_state.show_final = True

    # 2) 플래그가 켜져 있으면 항상 이 영역을 렌더링
    if st.session_state.show_final:
        essays_raw = st.session_state.get("essays_raw", [])

        if not essays_raw:
            st.info("원본 자소서 텍스트가 없어, 현재 세션에서는 수정본을 만들 수 없습니다. 다시 리포트를 생성해 주세요.")
        else:
            st.subheader("7. 문항별 최종 수정본(v1)")

            all_original_chunks = []
            all_revised_chunks = []

            for i, base in enumerate(essays_raw):
                title = base.get("question_title") or f"문항 {i+1}"
                original_text = base.get("text", "")

                # 대응되는 문항 리포트 찾기 (제목 기준으로 매칭)
                qrep = None
                for cand in getattr(report, "per_question", []) or []:
                    if isinstance(cand, str):
                        continue
                    if getattr(cand, "question_title", "").strip() == title.strip():
                        qrep = cand
                        break

                edits = getattr(qrep, "edits", []) if qrep else []
                revised_text = apply_edits_to_text(original_text, edits)

                with st.expander(f"{i+1}. {title}", expanded=(i == 0)):
                    col_orig, col_rev = st.columns(2)

                    with col_orig:
                        st.markdown("**원본**")
                        st.text_area(
                            f"원본 ({i+1})",
                            value=original_text,
                            key=f"final_orig_{i}",
                            height=180,
                        )

                    with col_rev:
                        st.markdown("**수정본(v1)**")
                        st.text_area(
                            f"수정본 ({i+1})",
                            value=revised_text,
                            key=f"final_rev_{i}",
                            height=180,
                        )

                # 다운로드용 텍스트 누적
                all_original_chunks.append(f"[{title}]")
                all_original_chunks.append(original_text)
                all_original_chunks.append("")

                all_revised_chunks.append(f"[{title}]")
                all_revised_chunks.append(revised_text)
                all_revised_chunks.append("")

            original_bundle = "\n".join(all_original_chunks)
            revised_bundle = "\n".join(all_revised_chunks)

            st.subheader("다운로드")

            st.download_button(
                label="원본 자소서 다운로드",
                data=original_bundle,
                file_name="original_essays.txt",
                mime="text/plain",
            )

            st.download_button(
                label="최종 수정본 다운로드",
                data=revised_bundle,
                file_name="revised_essays_v1.txt",
                mime="text/plain",
            )
