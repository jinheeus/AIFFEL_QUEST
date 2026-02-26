# LangSmith(옵션)
# coverletter-coach-rag/utils/logging.py

"""
LangSmith 평가 로깅/평가 유틸

[Dataset 컬럼 개념 매핑]
- input: 사용자가 준 입력 (JD/에세이/옵션) -> "재실행 가능"한 재료
- context: RAG가 찾아온 근거 (retrieved docs/patterns)
- prediction: 모델이 만든 최종 결과 (FullReport JSON)

[평가]
- relevance (LLM-as-Judge): 예측이 input/context에 정합한가?
- helpfulness (LLM-as-Judge): 사용자가 고칠 수 있을 만큼 실행가능한가?
- groundedness (hallucination): prediction의 evidence/인용이 input/context에 실제 존재하는가?
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langsmith import Client
from langsmith.evaluation import evaluate

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


client = Client()


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def create_langsmith_dataset(name: str, description: str):
    """LangSmith dataset 생성"""
    return client.create_dataset(dataset_name=name, description=description)


def add_example(
    dataset_id: str,
    *,
    # -------------------- input --------------------
    jd_text: str,
    essays: List[Dict[str, Any]],  # [{"question_title":..., "text":...}, ...]
    user_job: str = "",
    user_stack: str = "",
    options: Optional[Dict[str, Any]] = None,  # {"top_k":..., "use_mmr":..., "score_threshold":...}

    # -------------------- context --------------------
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    retrieved_patterns: Optional[List[str]] = None,

    # -------------------- prediction --------------------
    report_json: Dict[str, Any] | str = "",
):
    """
    Dataset row 1개 추가.

    - inputs: input (JD/에세이/옵션)
    - outputs: prediction (리포트) + context (retrieval 근거를 outputs 쪽에 같이 넣어둠)
      -> 이유: LangSmith evaluator에서 run/example 어디서든 쉽게 꺼내 쓰려고.
    """
    if options is None:
        options = {}

    # report_json이 dict면 문자열로 저장
    if isinstance(report_json, dict):
        report_text = json.dumps(report_json, ensure_ascii=False)
    else:
        report_text = str(report_json)

    outputs = {
        # prediction
        "prediction_report": report_text,

        # context (RAG 근거)
        "context_retrieved_docs": retrieved_docs or [],
        "context_retrieved_patterns": retrieved_patterns or [],
    }

    inputs = {
        # input
        "jd_text": jd_text,
        "essays": essays,
        "user_job": user_job,
        "user_stack": user_stack,
        "options": options,
    }

    client.create_example(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset_id,
    )


# -----------------------------------------------------------------------------
# Utility: normalize + simple fuzzy containment
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s or ""
    s = s.replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains(haystack: str, needle: str, min_len: int = 8) -> bool:
    """아주 가벼운 1차 검증(문자열 포함). 너무 짧은 needle은 오탐 많아서 제외."""
    haystack = _norm(haystack)
    needle = _norm(needle)
    if len(needle) < min_len:
        return False
    return needle in haystack


def _build_context_text(example_outputs: Dict[str, Any]) -> str:
    """
    retrieved_docs/patterns를 groundedness 평가용 텍스트로 합쳐둠
    """
    docs = example_outputs.get("context_retrieved_docs") or []
    patterns = example_outputs.get("context_retrieved_patterns") or []

    doc_texts = []
    for d in docs:
        # chunk_text / page_content 등 키가 다를 수 있어서 넉넉히 처리
        chunk = d.get("chunk_text") or d.get("page_content") or d.get("text") or ""
        meta = d.get("metadata") or {}
        doc_id = d.get("doc_id") or meta.get("doc_id") or meta.get("source") or ""
        score = d.get("score", "")
        header = f"[DOC] id={doc_id} score={score}".strip()
        doc_texts.append(header + "\n" + str(chunk))

    patterns_text = "\n".join([f"[PATTERN] {p}" for p in patterns])

    return _norm("\n\n".join(doc_texts + [patterns_text]))


# -----------------------------------------------------------------------------
# LLM-as-Judge evaluators (relevance / helpfulness)
# -----------------------------------------------------------------------------
def _judge_llm():
    # settings.py를 쓰고 있으면 거기 연결해도 됨.
    # 여기서는 최소한으로 구성.
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator for a resume/coverletter coaching RAG system. "
     "Score ONLY based on the provided INPUT/CONTEXT/PREDICTION. "
     "Return JSON only."),
    ("user",
     """
[INPUT]
JD:
{jd_text}

Essays(JSON):
{essays_json}

User profile:
- user_job: {user_job}
- user_stack: {user_stack}
Options(JSON):
{options_json}

[CONTEXT] (retrieved evidence)
{context_text}

[PREDICTION] (the generated report JSON)
{prediction_text}

Task:
1) Give a 'relevance' score (0-5) and short rationale.
   - Does the report's claims/feedback align with JD and essays?
   - Does it avoid introducing requirements not in JD?
   - If it cites "accepted patterns", do they exist in CONTEXT?
2) Give a 'helpfulness' score (0-5) and short rationale.
   - Are the suggestions actionable (specific edits, clear priorities)?
   - Does it include concrete rewrite guidance and evidence?
   - Is it not vague or generic?

Return JSON exactly:
{{
  "relevance_score": 0,
  "relevance_rationale": "",
  "helpfulness_score": 0,
  "helpfulness_rationale": ""
}}
""")
])


def llm_judge_evaluator(run, example) -> Dict[str, Any]:
    """
    LangSmith evaluate()에 넣을 evaluator (callable)
    - example.inputs: input
    - example.outputs: context + prediction (우리는 dataset에 그렇게 저장해둠)

    LangSmith가 요구하는 형식:
    - 최소한 {"key": str, "score": float} 구조 필요
    - 나머지 부가 정보는 comment 문자열에 합쳐서 넣음
    """
    ex_in = example.inputs
    ex_out = example.outputs

    jd_text = ex_in.get("jd_text", "")
    essays_json = json.dumps(ex_in.get("essays", []), ensure_ascii=False)
    user_job = ex_in.get("user_job", "")
    user_stack = ex_in.get("user_stack", "")
    options_json = json.dumps(ex_in.get("options", {}), ensure_ascii=False)

    context_text = _build_context_text(ex_out)
    prediction_text = ex_out.get("prediction_report", "")

    llm = _judge_llm()
    msg = _JUDGE_PROMPT.format_messages(
        jd_text=jd_text,
        essays_json=essays_json,
        user_job=user_job,
        user_stack=user_stack,
        options_json=options_json,
        context_text=context_text,
        prediction_text=prediction_text,
    )
    resp = llm.invoke(msg).content

    try:
        data = json.loads(resp)
    except Exception:
        # LLM이 JSON을 깨먹은 경우 대비(평가 실패로 남김)
        return {
            "key": "llm_judge",
            "score": 0.0,
            "comment": f"Invalid JSON from judge: {resp[:500]}",
        }

    # 0~5 점수 두 개를 0~1 범위의 단일 score로 통합
    rel = float(data.get("relevance_score", 0) or 0)
    help_ = float(data.get("helpfulness_score", 0) or 0)
    combined = (rel + help_) / (2 * 5)  # 0~1 사이

    comment = (
        f"relevance={rel}, helpfulness={help_}; "
        f"relevance_rationale={data.get('relevance_rationale', '')}; "
        f"helpfulness_rationale={data.get('helpfulness_rationale', '')}"
    )

    # LangSmith가 이해할 수 있는 EvaluationResult 형태
    return {
        "key": "llm_judge",
        "score": float(combined),
        "comment": comment,
    }


# -----------------------------------------------------------------------------
# Groundedness evaluator (hallucination check)
# -----------------------------------------------------------------------------
def groundedness_evaluator(run, example) -> Dict[str, Any]:
    """
    Hallucination(groundedness) 체크:
    - prediction_report 안의 improvement_suggestions[].evidence[].content 가
      input(JD/essays) 또는 context(retrieved docs/patterns)에 존재하는지 1차 검증.

    점수:
    - evidence 항목 중 '근거 존재' 비율로 0~1 score 산정
    """
    ex_in = example.inputs
    ex_out = example.outputs

    jd_text = _norm(ex_in.get("jd_text", ""))
    # essays = ex_in.get("essays", [])
    # essays_text = _norm("\n".join([str(e.get("text", "")) for e in essays]))
    jd_text = _norm(ex_in.get("jd_text", ""))

    essays = ex_in.get("essays", [])
    # essays가 문자열 하나일 수도 있고, 리스트일 수도 있음
    if isinstance(essays, str):
        essays = [essays]

    essay_text_list = []
    for e in essays:
        if isinstance(e, dict):
            essay_text_list.append(str(e.get("text", "")))
        else:
            # 문자열/기타 타입은 그냥 str로 캐스팅
            essay_text_list.append(str(e))

    essays_text = _norm("\n".join(essay_text_list))
# <-- essays 수정 코드 삽입

    context_text = _build_context_text(ex_out)

    prediction_text = ex_out.get("prediction_report", "")
    try:
        pred = json.loads(prediction_text) if isinstance(prediction_text, str) else prediction_text
    except Exception:
        return {
            "key": "groundedness",
            "score": 0.0,
            "comment": "prediction_report is not valid JSON",
        }

    sugg = pred.get("improvement_suggestions", []) or []
    evidence_items = []
    for s in sugg:
        ev_list = s.get("evidence", []) or []
        for ev in ev_list:
            ev_type = str(ev.get("type", "")).strip()
            content = str(ev.get("content", "")).strip()
            if content:
                evidence_items.append((ev_type, content))

    if not evidence_items:
        # evidence가 없으면 groundedness 측면에서 위험하므로 0점
        return {
            "key": "groundedness",
            "score": 0.0,
            "comment": "No evidence items found in prediction (high hallucination risk).",
        }

    supported = 0
    unsupported_samples = []

    for ev_type, content in evidence_items:
        ok = False
        if ev_type.upper() == "JD":
            ok = _contains(jd_text, content)
        elif ev_type in ("지원자문장", "지원자 문장", "APPLICANT", "ESSAY"):
            ok = _contains(essays_text, content)
        elif ev_type in ("합격사례", "합격 사례", "CONTEXT", "RAG"):
            ok = _contains(context_text, content)
        else:
            # 타입이 애매하면 context에서라도 찾게 함(보수적으로)
            ok = _contains(jd_text, content) or _contains(essays_text, content) or _contains(context_text, content)

        if ok:
            supported += 1
        else:
            if len(unsupported_samples) < 5:
                unsupported_samples.append(f"[{ev_type}] {content[:120]}")

    score = supported / max(1, len(evidence_items))

    comment = (
        f"supported={supported}/{len(evidence_items)} "
        f"unsupported_samples={unsupported_samples}"
    )

    return {
        "key": "groundedness",
        "score": float(score),  # 0~1
        "comment": comment,
    }

# -----------------------------------------------------------------------------
# overall_gap evaluator (human vs model overall score)
# -----------------------------------------------------------------------------
def overall_gap_evaluator(run, example) -> Dict[str, Any]:
    """
    사람 점수(target_overall)와 모델 점수(report.overall_scores.overall)의 차이를
    0~1 스케일의 overall_gap으로 계산.

    전제:
    - example.outputs["target_overall"] : 사람이 매긴 기준 점수 (0~100)
    - run.outputs["report"]             : app_fn이 반환한 FullReport dict 또는 JSON 문자열
    """
    ex_out = example.outputs or {}
    human_score = ex_out.get("target_overall", None)

    if human_score is None:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": "No target_overall label in example.outputs; cannot compute gap.",
        }

    try:
        human_score = float(human_score)
    except Exception:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": f"target_overall is not numeric: {human_score!r}",
        }

    run_out = run.outputs or {}
    report = run_out.get("report")
    if report is None:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": "run.outputs['report'] not found.",
        }

    # report가 문자열이면 JSON 파싱
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except Exception:
            return {
                "key": "overall_gap",
                "score": 0.0,
                "comment": f"report is not valid JSON: {report[:200]}",
            }

    try:
        model_overall = report.get("overall_scores", {}).get("overall", None)
    except AttributeError:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": "report has no 'overall_scores.overall' field.",
        }

    if model_overall is None:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": "overall_scores.overall is None; cannot compute gap.",
        }

    try:
        model_overall = float(model_overall)
    except Exception:
        return {
            "key": "overall_gap",
            "score": 0.0,
            "comment": f"overall_scores.overall is not numeric: {model_overall!r}",
        }

    # gap = 1 - |AI - Human| / 100  (0~1, 1에 가까울수록 사람과 비슷)
    gap = 1.0 - abs(model_overall - human_score) / 100.0
    gap = max(0.0, min(1.0, gap))

    comment = f"human={human_score}, model={model_overall}, overall_gap={gap:.3f}"

    return {
        "key": "overall_gap",
        "score": float(gap),
        "comment": comment,
    }

# -----------------------------------------------------------------------------
# Run evaluation
# -----------------------------------------------------------------------------
def run_langsmith_eval(
    runs,
    dataset,
):
    """
    runs: LangSmith traced runs or callable that produces runs (프로젝트 구성에 따라 다름)
    dataset: create_langsmith_dataset로 만든 dataset 객체 or dataset_name

    evaluators:
    - overall_gap_evaluator: 사람 점수와 모델 점수 차이
    - llm_judge_evaluator: relevance/helpfulness
    - groundedness_evaluator: hallucination(근거 존재 비율)
    """
    return evaluate(
        runs,
        data=dataset,
        evaluators=[
            overall_gap_evaluator,
            llm_judge_evaluator,
            groundedness_evaluator,
        ],
    )
