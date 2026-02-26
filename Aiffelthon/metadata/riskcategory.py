"""
하이브리드 분류 시스템 FINAL
================================================================================
전략:
  Step 1: 키워드 검색 (contents_summary 우선)
  AI 검증 (키워드 있을 때)
    - 일치 → 키워드 채택 (very_high)
    - 불일치 → AI 채택 (high)
    - AI 실패 → 키워드 그대로 (low)
  AI 직접 (키워드 없을 때)

출력:
  - 카테고리: 무조건 6개 중 하나
  - JSON (상세)
  - CSV (Looker Studio용)
  - 통합 JSON (Streamlit용)
================================================================================
"""

import json
import time
import os
import re
import csv
from datetime import datetime
from collections import defaultdict
from typing import Optional, Tuple
from dotenv import load_dotenv
from langchain_community.chat_models import ChatClovaX
from langchain_core.messages import HumanMessage

load_dotenv()

CLOVA_API_KEY = os.getenv("CLOVASTUDIO_API_KEY")
INPUT_FILE = "data_2_AURA_rev_title.json"

# 6대 카테고리 (변경 불가)
VALID_CATEGORIES = [
    "윤리/부패/비위",
    "인사/채용/복무",
    "정보보안/IT",
    "시설/안전/환경",
    "재무/회계/계약",
    "사업/운영/성과"
]

MATCHED_KEYWORDS = {
    "윤리/부패/비위": ["갑질", "개발정보", "개입", "공정성", "공직기강", "금품", "담합",
        "부동산투기", "부패", "불공정", "비리", "비위", "이해충돌", "청렴", "청탁", "향응", "횡령"],
    "인사/채용/복무": ["건강검진", "겸직", "근로기준법", "근무태도", "근무태만", "근태",
        "무단결근", "민원응대", "복무", "상벌규정", "성희롱", "소속직원관리", "승진",
        "업무용차량", "업무차량", "연차휴가", "온라인 교육", "유연근무제", "의원면직",
        "인사발령", "인사통보", "임용", "임직원행동강령", "재택근무", "조직", "채용",
        "채용절차", "출입자명부", "출장", "출퇴근차량", "취업규정", "퇴직", "품위유지의무", "현지조치"],
    "정보보안/IT": ["CCTV 영상", "PC보안", "개인정보", "무단반출", "보안업무규정",
        "보안취약점", "시스템장애", "유출", "정보기술", "정보보안", "해킹"],
    "시설/안전/환경": ["가동중지", "경보", "계측관리", "계측기", "계통사고", "광역편제역",
        "교통안전성", "긴급작업승인절차", "나들목", "단선", "도로시설", "보강방안",
        "부실시공", "사고조사", "산업안전보건법", "설계 오류", "소등", "실내공기질",
        "안전관리", "안전난간", "안전문", "안전사고", "안전점검", "안정성", "연약지반",
        "열차", "임의보수", "장비검수", "재난", "적정안전시설", "점검업무", "침수예방",
        "탐지기", "터널", "폐기물", "화재", "환경영향평가", "환경오염"],
    "재무/회계/계약": ["가산세", "감가상각", "검수", "경제성검토", "계약관리 부적정",
        "계약금액조정", "계약서", "계약업무", "계약위반", "계약체결", "고용부담금",
        "공탁금", "과다 지급", "과다계상", "과다산출", "과징금", "국가계약법", "금융",
        "기금", "기부금 집행", "납품", "단가적용", "매입부가세", "물품관리", "법인카드",
        "변상금", "보조금", "보증금 반환", "부가가치세액", "부당지급", "사업시설임대차계약서",
        "사용료 징수", "사용료연체", "산업안전보건관리비", "상품권", "선지급", "설치비용",
        "소멸시효", "소유권이전등기", "수의계약", "수익", "예산 집행지침", "예산낭비",
        "예산편성", "외부위탁", "외주가공비", "용역비용", "운송계약", "원천세",
        "임대료 산정", "입찰참가", "자산관리", "자산취득", "자재구매", "재무감사",
        "재산세 감면", "전기및통신요금", "정산", "지연이자", "지출", "채권",
        "토지보상금", "토지보상비", "특혜", "하자관리", "회계", "회계연도", "회계장부"],
    "사업/운영/성과": ["경제성 평가", "관리지침", "미이행", "방치", "보도자료",
        "사업관리", "성과", "실태", "연구", "예산절감", "위임전결", "위탁운영",
        "일상감사", "평가", "품질", "현장적용 역무", "후속업무"]
}

# ============================================
# 유틸리티
# ============================================

def clean_txt(t):
    if not t or str(t) == 'nan':
        return ""
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', str(t))

def normalize_category(cat: str) -> str:
    """
    AI 응답을 무조건 6개 카테고리 중 하나로 정규화
    
    처리:
    1. 번호 제거: "6. 사업/운영/성과" → "사업/운영/성과"
    2. 세분화 제거: "윤리/부패/비위 - 복무규정" → "윤리/부패/비위"
    3. 축약형 보완: "시설/안전" → "시설/안전/환경"
    4. 부분 매칭: 6개 중 포함되는 것 찾기
    5. 완전 실패: None 반환
    """
    if not cat:
        return None

    cat = cat.strip()

    # Step 1: 번호 제거
    cat = re.sub(r'^\d+[\.\)]\s*', '', cat).strip()

    # Step 2: 세분화 제거 (대시 이후)
    if ' - ' in cat:
        cat = cat.split(' - ')[0].strip()

    # Step 3: 쉼표 이후 제거
    if ',' in cat:
        cat = cat.split(',')[0].strip()

    # Step 4: 축약형 보완
    abbreviations = {
        "시설/안전": "시설/안전/환경",
        "재무/회계": "재무/회계/계약",
        "윤리/부패": "윤리/부패/비위",
        "인사/채용": "인사/채용/복무",
        "사업/운영": "사업/운영/성과"
    }
    for short, full in abbreviations.items():
        if short in cat and full not in cat:
            cat = full
            break

    # Step 5: 정확히 6개 중 하나면 반환
    if cat in VALID_CATEGORIES:
        return cat

    # Step 6: 부분 매칭 (포함 관계 확인)
    for valid in VALID_CATEGORIES:
        if valid in cat or cat in valid:
            return valid

    # Step 7: 키워드 매칭
    cat_keywords = {
        "윤리/부패/비위": ["윤리", "부패", "비위", "횡령", "청탁"],
        "인사/채용/복무": ["인사", "채용", "복무", "근태"],
        "정보보안/IT": ["정보보안", "IT", "개인정보", "보안"],
        "시설/안전/환경": ["시설", "안전", "환경", "시공"],
        "재무/회계/계약": ["재무", "회계", "계약", "예산", "정산"],
        "사업/운영/성과": ["사업", "운영", "성과", "관리"]
    }
    for valid_cat, kws in cat_keywords.items():
        if any(kw in cat for kw in kws):
            return valid_cat

    return None  # 완전 실패

def extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except:
        cleaned = text.strip()
        cleaned = cleaned.replace("True", "true").replace("False", "false").replace("None", "null")
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group().replace("'", '"'))
            except:
                pass
    return None

# ============================================
# Step 1: 키워드 검색
# ============================================

def step1_keyword(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    contents_summary 우선 검색
    Returns: (category, keyword) or (None, None)
    """
    summary = item.get('contents_summary', {})

    # compare_classification.py와 동일한 순서
    search_targets = [
        summary.get('action', ''),    # 요약 조치사항 (1순위)
        summary.get('problems', ''),  # 요약 문제점 (2순위)
        item.get('title', ''),        # 제목 (3순위)
        item.get('action', ''),       # 원본 조치사항 (폴백)
        item.get('problem', '')       # 원본 문제점 (폴백)
    ]

    for text in search_targets:
        clean_text = clean_txt(text)
        if not clean_text:
            continue
        for category, keywords in MATCHED_KEYWORDS.items():
            for kw in keywords:
                if clean_txt(kw) in clean_text:
                    return (category, kw)

    return (None, None)

# ============================================
# Step 2: AI 검증
# ============================================

def step2_verify(item: dict, keyword_category: str, retry: int = 0) -> Optional[dict]:
    """
    키워드 결과를 AI로 검증
    Returns: 성공 시 dict, 실패 시 None
    """
    title = str(item.get('title', ''))
    summary = item.get('contents_summary', {})
    problems = str(summary.get('problems', item.get('problem', '')))[:500]
    action = str(summary.get('action', item.get('action', '')))[:400]

    prompt = f"""공공기관 감사보고서를 아래 6개 카테고리 중 하나로 분류하세요.

[카테고리]
1. 윤리/부패/비위
2. 인사/채용/복무
3. 정보보안/IT
4. 시설/안전/환경
5. 재무/회계/계약
6. 사업/운영/성과

[판단 원칙]
- 조치사항을 가장 우선 확인
- 금액/비용/정산 관련 → 재무/회계/계약
- 안전 "사고" → 시설/안전/환경
- 안전 "비용" → 재무/회계/계약

[데이터]
제목: {title}
문제: {problems}
조치: {action}

[키워드 힌트]
키워드 분석 결과: "{keyword_category}"
→ 맥락상 맞으면 채택, 아니면 수정

아래 JSON 형식으로만 응답:
{{
  "category": "위 6개 중 정확히 하나",
  "confidence": "high/medium/low",
  "reason": "한 문장"
}}"""

    try:
        chat = ChatClovaX(
            model="HCX-003",
            ncp_clovastudio_api_key=CLOVA_API_KEY,
            temperature=0.1,
            max_tokens=300
        )
        resp = chat.invoke([HumanMessage(content=prompt)])
        result = extract_json(resp.content.strip())

        if result and 'category' in result:
            cat = normalize_category(result['category'])
            if cat:  # 정규화 성공
                return {
                    "category": cat,
                    "confidence": result.get('confidence', 'medium'),
                    "reason": result.get('reason', ''),
                    "success": True
                }
    except Exception as e:
        if retry < 2:
            time.sleep(2)
            return step2_verify(item, keyword_category, retry + 1)

    return None  # 실패

# ============================================
# Step 3: AI 직접
# ============================================

def step3_direct(item: dict, retry: int = 0) -> Optional[dict]:
    """
    키워드 없을 때 AI 직접 분류
    Returns: 성공 시 dict, 실패 시 None
    """
    title = str(item.get('title', ''))
    summary = item.get('contents_summary', {})
    problems = str(summary.get('problems', item.get('problem', '')))[:500]
    action = str(summary.get('action', item.get('action', '')))[:400]

    prompt = f"""공공기관 감사보고서를 아래 6개 카테고리 중 하나로 분류하세요.

[카테고리]
1. 윤리/부패/비위
2. 인사/채용/복무
3. 정보보안/IT
4. 시설/안전/환경
5. 재무/회계/계약
6. 사업/운영/성과

[판단 원칙]
- 조치사항을 가장 우선 확인
- 금액/비용/정산 관련 → 재무/회계/계약
- 안전 "사고" → 시설/안전/환경
- 안전 "비용" → 재무/회계/계약

[데이터]
제목: {title}
문제: {problems}
조치: {action}

아래 JSON 형식으로만 응답:
{{
  "category": "위 6개 중 정확히 하나",
  "confidence": "high/medium/low",
  "reason": "한 문장"
}}"""

    try:
        chat = ChatClovaX(
            model="HCX-003",
            ncp_clovastudio_api_key=CLOVA_API_KEY,
            temperature=0.1,
            max_tokens=300
        )
        resp = chat.invoke([HumanMessage(content=prompt)])
        result = extract_json(resp.content.strip())

        if result and 'category' in result:
            cat = normalize_category(result['category'])
            if cat:
                return {
                    "category": cat,
                    "confidence": result.get('confidence', 'medium'),
                    "reason": result.get('reason', ''),
                    "success": True
                }
    except Exception as e:
        if retry < 2:
            time.sleep(2)
            return step3_direct(item, retry + 1)

    return None  # 실패

# ============================================
# 메인 하이브리드 로직
# ============================================

def classify_hybrid(item: dict) -> dict:
    """
    하이브리드 분류

    키워드 검색
    AI 검증 (키워드 있을 때)
    AI 직접 (키워드 없을 때)
    """

    # Step 1: 키워드 검색
    keyword_cat, matched_kw = step1_keyword(item)

    if keyword_cat:
        # Step 2: AI 검증
        ai_result = step2_verify(item, keyword_cat)

        if ai_result:
            ai_cat = ai_result['category']

            if keyword_cat == ai_cat:
                # ✅ 키워드 == AI → 키워드 채택 (최고 신뢰)
                return {
                    "category": keyword_cat,
                    "method": "keyword_ai_verified",
                    "confidence": "very_high",
                    "keyword": matched_kw,
                    "ai_reason": ai_result.get('reason', '')
                }
            else:
                # ⚠️ 키워드 != AI → AI 채택
                return {
                    "category": ai_cat,
                    "method": "ai_corrected",
                    "confidence": ai_result.get('confidence', 'high'),
                    "keyword": matched_kw,
                    "keyword_suggested": keyword_cat,
                    "ai_reason": ai_result.get('reason', '')
                }
        else:
            # ❗ AI 실패 → 키워드 그대로
            return {
                "category": keyword_cat,
                "method": "keyword_only",
                "confidence": "low",
                "keyword": matched_kw
            }

    else:
        # Step 3: AI 직접
        ai_result = step3_direct(item)

        if ai_result:
            return {
                "category": ai_result['category'],
                "method": "ai_direct",
                "confidence": ai_result.get('confidence', 'medium'),
                "keyword": None,
                "ai_reason": ai_result.get('reason', '')
            }
        else:
            # AI도 실패 → 가장 빈도 높은 카테고리로 fallback
            return {
                "category": "사업/운영/성과",  # 가장 안전한 기본값
                "method": "ai_failed",
                "confidence": "very_low",
                "keyword": None
            }

# ============================================
# 저장 (JSON + CSV + 통합 JSON)
# ============================================

def save_results(results: list, original_data: list, stats: dict, elapsed: float, ts: str):

    # 1. 분류 결과 JSON
    output_json = f"hybrid_results_{ts}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {**stats, "time_seconds": elapsed, "timestamp": ts},
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {output_json}")

    # 2. CSV (Looker Studio / Excel)
    output_csv = f"hybrid_results_{ts}.csv"
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'idx', 'risk_category', 'confidence', 'method', 'keyword', 'ai_reason'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'idx': r['idx'],
                'risk_category': r['category'],
                'confidence': r.get('confidence', ''),
                'method': r.get('method', ''),
                'keyword': r.get('keyword', ''),
                'ai_reason': str(r.get('ai_reason', ''))[:100]
            })
    print(f"✅ CSV: {output_csv}  ← Looker Studio 업로드용")

    # 3. 통합 JSON (원본 + 분류 결과, Streamlit용)
    output_integrated = f"data_classified_{ts}.json"
    results_map = {r['idx']: r for r in results}

    integrated = []
    for item in original_data:
        idx = item.get('idx')
        cls = results_map.get(idx, {})
        integrated.append({
            **item,
            "classification": {
                "risk_category": cls.get('category', '미분류'),
                "confidence": cls.get('confidence', ''),
                "method": cls.get('method', ''),
                "keyword": cls.get('keyword', ''),
                "ai_reason": cls.get('ai_reason', ''),
                "classified_at": ts
            }
        })

    with open(output_integrated, 'w', encoding='utf-8') as f:
        json.dump(integrated, f, ensure_ascii=False, indent=2)
    print(f"✅ 통합 JSON: {output_integrated}  ← Streamlit 로드용")

# ============================================
# 실행
# ============================================

def run():
    print("=" * 70)
    print("🚀 하이브리드 분류 FINAL")
    print("=" * 70)
    print("키워드 검색 (contents_summary 우선)")
    print("AI 검증 (키워드 있을 때)")
    print("AI 직접 (키워드 없을 때)")
    print("출력: 무조건 6개 카테고리 중 하나")
    print("=" * 70)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n✅ 전체: {len(data)}건")

    # 키워드 커버리지 사전 확인
    print("\n🔍 키워드 커버리지 확인 (200건 샘플)...")
    hit = sum(1 for item in data[:200] if step1_keyword(item)[0])
    print(f"   매칭: {hit}/200건 ({hit/2:.0f}%)")

    choice = input("\n1. 전체 / 2. 샘플 100건: ").strip()

    if choice == "1":
        to_classify = data
    else:
        import random
        random.seed(42)
        to_classify = random.sample(data, min(100, len(data)))

    print(f"\n✅ {len(to_classify)}건 분류 시작\n")
    print("=" * 70)

    results = []
    stats = defaultdict(int)
    start = time.time()

    for i, item in enumerate(to_classify, 1):
        print(f"[{i:04d}/{len(to_classify)}] idx {item.get('idx'):<5} ", end="", flush=True)

        result = classify_hybrid(item)
        results.append({"idx": item.get('idx'), **result})

        method = result["method"]
        stats[method] += 1
        stats["ai_calls"] += 1 if method != "keyword_only" else 0

        # 출력
        icons = {
            "keyword_ai_verified": "✅",
            "ai_corrected":        "⚠️ ",
            "keyword_only":        "❗",
            "ai_direct":           "🤖",
            "ai_failed":           "💥"
        }
        icon = icons.get(method, "❓")

        if method == "keyword_ai_verified":
            print(f"{icon} 키워드({result['keyword']}) + AI 일치 → {result['category']}")
        elif method == "ai_corrected":
            print(f"{icon} 키워드({result.get('keyword_suggested')}) → AI 수정 → {result['category']}")
        elif method == "keyword_only":
            print(f"{icon} AI 실패, 키워드 사용 → {result['category']}")
        elif method == "ai_direct":
            print(f"{icon} AI 직접 → {result['category']}")
        else:
            print(f"{icon} → {result['category']}")

        # API 제한
        if method != "keyword_only":
            time.sleep(1.2)

        if i % 20 == 0:
            elapsed_now = time.time() - start
            print(f"\n   진행: {i}/{len(to_classify)} | AI: {stats['ai_calls']} | {elapsed_now:.0f}초\n")

    elapsed = time.time() - start
    total = len(results)
    stats['total'] = total
    stats['cost'] = stats['ai_calls'] * 0.66

    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 최종 결과")
    print("=" * 70)
    print(f"\n총: {total}건")

    method_labels = {
        "keyword_ai_verified": "✅ 키워드 + AI 일치",
        "ai_corrected":        "⚠️  AI 수정",
        "keyword_only":        "❗ AI 실패 (키워드 사용)",
        "ai_direct":           "🤖 AI 직접",
        "ai_failed":           "💥 완전 실패"
    }
    for method, label in method_labels.items():
        count = stats.get(method, 0)
        if count > 0:
            print(f"  {label}: {count}건 ({count/total*100:.1f}%)")

    print(f"\n💰 비용: ₩{stats['cost']:.0f}")
    print(f"⏱️  시간: {elapsed/60:.1f}분")

    # 카테고리 분포 (6개만 나와야 함)
    cat_dist = defaultdict(int)
    for r in results:
        cat_dist[r['category']] += 1

    print("\n📊 카테고리 분포:")
    for cat in VALID_CATEGORIES:
        count = cat_dist.get(cat, 0)
        bar = "█" * (count * 30 // total) if total > 0 else ""
        print(f"  {cat:<20}: {count:>5}건 ({count/total*100:>5.1f}%) {bar}")

    # 이상값 체크
    invalid = {k: v for k, v in cat_dist.items() if k not in VALID_CATEGORIES}
    if invalid:
        print(f"\n⚠️  이상값 발견: {invalid}")
    else:
        print("\n✅ 카테고리 이상값 없음 (6개 완벽)")

    # 저장
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("\n💾 저장 중...")
    save_results(results, to_classify, dict(stats), elapsed, ts)

    print("\n" + "=" * 70)
    print("✅ 완료!")
    print("=" * 70)
    print(f"\n대시보드 연동:")
    print(f"  Looker Studio → hybrid_results_{ts}.csv")
    print(f"  Streamlit     → data_classified_{ts}.json")

if __name__ == "__main__":
    run()
