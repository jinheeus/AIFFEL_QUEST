import requests
import json
import os
import uuid
import pandas as pd
import re
from dotenv import load_dotenv
from datetime import datetime

# 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

CLOVA_API_KEY = os.getenv("CLOVASTUDIO_API_KEY") or os.getenv("NCP_CLOVASTUDIO_API_KEY")

if not CLOVA_API_KEY:
    print("❌ [오류] .env 파일에서 API Key를 찾을 수 없습니다.")
    exit()

print(f"🔑 API Key 확인: {CLOVA_API_KEY[:5]}..." + "*"*10)

# 추출 대상 idx 리스트
TARGET_IDX = [1876, 2005, 2222, 2481, 2908, 2919, 2925, 2951, 2954, 2958, 3044, 3045, 
              3046, 3048, 3049, 3113, 3114, 3115, 3117, 3118, 3175, 3177, 3178, 3179, 
              3187, 3207, 3214, 3215, 3216, 3217, 3245, 3247, 3249, 3250, 3256, 3258, 
              3259, 3262, 3263, 3286, 3328, 3334, 3335, 3337, 3426, 3473, 3475, 3478, 
              3481, 3548, 3549, 3550, 3555, 3557, 3558, 3559, 3560, 3596, 3597, 3598, 
              3602, 3603, 3667, 3668, 3677, 3685, 3704, 3705, 3706, 3707, 3708, 3709, 
              3822, 3823, 4241, 4242]

# 매칭 키워드
DISPOSITION_KEYWORDS = ['통보', '주의', '시정', '경고', '개선', '고발', '징계', '견책', 
                        '감액', '회수', '경징계', '감봉', '정직', '중징계', '파면', '환수']

def build_inference_prompt_v2(action_text):
    """개선된 추론 프롬프트"""
    return f"""당신은 공공기관 감사 결과의 처분 수위를 판정하는 행정 전문가입니다. 
제시된 [Action] 문장은 여러 처분이 섞여 있거나 표준 단어가 명시되지 않은 복잡한 사례입니다.
아래 지침에 따라 가장 우선순위가 높은 'disposition_level'을 결정하세요.

**중요: 반드시 아래 [Action] 필드의 텍스트에서만 근거를 찾고, 문맥을 파악하여 판단하세요.**

### [분류 계층 및 정의 (우선순위 순)]
1. 중징계: 정직, 파면, 고발, 수사 의뢰 등 신분 박탈 및 법적 책임
2. 경징계: 감봉, 견책, 인사상 징계 기록이 남는 조치
3. 시정: 환수, 회수, 설계변경, 예산 제외 등 행정적/경제적 원상복구
4. 경고/주의: 관련자 엄중 경고
5. 통보: 관리 방안 마련, 개선 권고, 자율적 조치 요구
   - **"마련하시기 바랍니다", "~하시기 바랍니다" 등으로 끝나면 통보로 분류**

### [오분류 방지 규칙 - 매우 중요!]
⚠️ **다음 키워드는 해당 카테고리로 분류하지 마세요:**
1. **"설계지침", "설계변경", "시공관리", "개정", "조치" 단독 출현 → 시정이 아님**
   - 이런 단어들이 나왔다고 무조건 '시정'으로 분류하지 마세요
   - 실제 금전 환수나 원상복구가 명시된 경우에만 시정으로 분류
   
2. **"철저히", "관리를 철저히", "규정 준수", "회계관련 규정 준수" → 경고/주의가 아님**
   - "철저히 관리", "철저한 준수" 등은 통보 수준의 권고사항
   - 실제 교육 실시, 엄중 경고가 명시되지 않으면 경고/주의로 분류하지 마세요

### [복합 문장 판정 가이드라인]
- **반드시 [Action] 필드의 실제 텍스트만 참고**하여 판단하세요.
- 한 문장에 여러 처분이 있으면 가장 높은 '수위(Hierarchy)'를 선택합니다.
- 예: '환수(시정)'와 '통보'가 같이 있으면 → '시정'으로 분류
- 예: '정직(중징계)'과 '재발방지 대책(통보)'이 같이 있으면 → '중징계'로 분류
- 예: '마련하시기 바랍니다'로 끝나면 → '통보'로 분류

### [판정 예시 (Few-shot)]
- 입력: "AA건설사업단장은 터널공사용 CCTV를 안전사고예방 및 원격현장관리에 활용하는 방안을 마련하시기 바랍니다."
  -> 분류: 통보 
  -> 근거: "방안을 마련하시기 바랍니다"라는 표현으로 개선 권고 및 자율적 조치 요구

- 입력: "문책 및 시정 요구로 오납부 하수도료 환수 조치 이행, 관련기관에 통보 및 내부통제 강화"
  -> 분류: 시정 
  -> 근거: '문책'과 '통보'가 포함되어 있으나, 실질적인 금전적 복구인 '환수 조치'가 핵심

- 입력: "조치할 사항□□본부장으로하여금,근무성적평정을사규에따라철저히 하도록"통보"조치하고,업무를소홀히한관련자에게"경고"및"주의"를 요구하고자합니다.[통보1,경고1,주의1]"
  -> 분류: 경고/주의 
  -> 근거: '업무를소홀히한관련자','요구하고자합니다'는 통보보다는 경고/주의에 가까운 표현

- 입력: "본부장은 D에 대해 인사규정에 의한 감봉 조치를 내리도록 한다. 재발방지 교육 및 내부통제 강화도 포함한다."
  -> 분류: 경징계 
  -> 근거: '감봉 조치'는 인사상 불이익이 발생하는 경징계에 해당하며, 교육은 부수적 조치

- 입력: "BI에 대한 징계(정직) 처분을 권고하고 금품수수 위반으로 법원에 통보한다. 재발방지 대책을 수립한다."
  -> 분류: 중징계 
  -> 근거: '정직 처분' 및 '법원 통보'는 신분 박탈 및 법적 책임을 수반하는 가장 엄중한 처분

- 입력: "향후 유사사례 재발방지를 위한 관리방안을 수립하시기 바랍니다."
  -> 분류: 통보
  -> 근거: '수립하시기 바랍니다'로 끝나므로 개선 권고 및 자율적 조치 요구

- 입력: "설계지침을 준수하여 시공관리를 철저히 하시기 바랍니다."
  -> 분류: 통보
  -> 근거: '철저히'와 '설계지침' 키워드가 있으나 실제 환수나 교육 실시가 없고, '하시기 바랍니다'로 끝나므로 통보

- 입력: "회계 관련 규정을 철저히 준수하고 관리를 강화하시기 바랍니다."
  -> 분류: 통보
  -> 근거: '철저히' 키워드가 있으나 실제 경고나 교육이 명시되지 않고 권고 형식이므로 통보

### [대상 분석]
[Action]: "{action_text}"

**위 [Action] 필드의 실제 문맥을 정확히 분석하여 판단하세요.**

결과는 아래 JSON 형식으로만 답변하세요:
{{
    "extracted_word": "판단의 근거가 된 핵심 키워드 (반드시 [Action]에서 추출)",
    "disposition_level": "6개 카테고리 중 하나 (중징계/경징계/시정/경고,주의/통보)",
    "reason": "복합 처분 중 해당 레벨을 선택한 논리적 근거 ([Action]의 실제 텍스트 기반)"
}}"""

def call_hcx_model(model_name, action_text):
    """HCX 모델 호출"""
    
    prompt = build_inference_prompt_v2(action_text)
    
    if model_name == "HCX-003":
        url = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
        data = {
            "messages": [
                {"role": "system", "content": "당신은 감사 처분 수준을 추출하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            "maxTokens": 300,
            "temperature": 0.1,
            "includeAiFilters": True
        }
    else:  # HCX-005
        url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
        data = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }],
            "maxTokens": 300,
            "topP": 0.8,
            "topK": 0,
            "temperature": 0.1,
            "stop": [],
            "includeAiFilters": True,
            "seed": 0
        }
    
    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['result']['message']['content'].strip()
            
            # JSON 파싱 시도
            try:
                # JSON 마크다운 제거
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # 줄바꿈 및 불필요한 공백 제거
                content = content.replace('\n', ' ').replace('\r', ' ')
                # 연속된 공백을 하나로
                content = re.sub(r'\s+', ' ', content)
                    
                result = json.loads(content)
                return {
                    'extracted_word': result.get('extracted_word'),
                    'disposition_level': result.get('disposition_level'),
                    'reason': result.get('reason')
                }
            except Exception as e:
                # JSON 파싱 실패 시 정규식으로 추출 시도
                print(f"  ⚠️ JSON 파싱 실패, 정규식으로 재시도 중...")
                try:
                    # 더 유연한 패턴으로 추출
                    extracted_word_match = re.search(r'"extracted_word"\s*:\s*"([^"]+)"', content)
                    disposition_match = re.search(r'"disposition_level"\s*:\s*"([^"]+)"', content)
                    # reason은 마지막 }까지 모두 포함
                    reason_match = re.search(r'"reason"\s*:\s*"(.+?)\s*"\s*}', content, re.DOTALL)
                    
                    if not reason_match:
                        # 혹시 reason에 "가 포함된 경우를 위한 대안 패턴
                        reason_start = content.find('"reason"')
                        if reason_start != -1:
                            reason_start = content.find(':', reason_start) + 1
                            reason_start = content.find('"', reason_start) + 1
                            reason_end = content.rfind('"')
                            if reason_start < reason_end:
                                reason_text = content[reason_start:reason_end].strip()
                            else:
                                reason_text = "추출 불가"
                        else:
                            reason_text = "추출 불가"
                    else:
                        reason_text = reason_match.group(1)
                    
                    if extracted_word_match and disposition_match:
                        print(f"  ✓ 정규식 추출 성공!")
                        return {
                            'extracted_word': extracted_word_match.group(1),
                            'disposition_level': disposition_match.group(1),
                            'reason': reason_text
                        }
                except Exception as regex_error:
                    print(f"  ⚠️ 정규식 추출도 실패: {regex_error}")
                
                print(f"  원본 응답: {content[:300]}")
                return None
        else:
            print(f"  API 오류: {response.status_code}")
            return None
    except Exception as e:
        print(f"  호출 오류: {e}")
        return None

def process_extraction(df, model_name):
    """추출 프로세스 실행"""
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*60}")
    print(f"🤖 {model_name} 모델로 처분 수준 추출 시작")
    print(f"{'='*60}")
    
    # idx 컬럼 값으로 필터링 (DataFrame 인덱스가 아님!)
    target_df = df[df['idx'].isin(TARGET_IDX)].copy()
    
    print(f"📌 처리 대상: {len(target_df)}건 (idx 컬럼 기준)")
    
    for _, row in target_df.iterrows():
        idx = row['idx']  # 실제 idx 컬럼 값
        action_text = str(row['action']) if pd.notna(row['action']) else ""
        
        print(f"\n[idx:{idx}] 처리 중... (진행: {len(results)+1}/{len(target_df)})")
        
        if not action_text:
            results.append({
                'idx': idx,
                'original_action': action_text,
                'extracted_disposition': None,
                'confidence': None,
                'reasoning': "action 필드 비어있음",
                'status': 'FAIL'
            })
            fail_count += 1
            continue
        
        result = call_hcx_model(model_name, action_text)
        
        if result and result.get('disposition_level'):
            results.append({
                'idx': idx,
                'original_action': action_text,
                'extracted_word': result.get('extracted_word'),
                'disposition_level': result.get('disposition_level'),
                'reason': result.get('reason'),
                'status': 'SUCCESS'
            })
            success_count += 1
            print(f"  ✅ 추출 성공: {result.get('disposition_level')} (근거: {result.get('extracted_word')})")
        else:
            results.append({
                'idx': idx,
                'original_action': action_text,
                'extracted_word': None,
                'disposition_level': None,
                'reason': "매칭 실패",
                'status': 'FAIL'
            })
            fail_count += 1
            print(f"  ❌ 추출 실패")
    
    # 통계
    total = len(results)
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} 추출 결과")
    print(f"{'='*60}")
    print(f"총 처리: {total}건")
    print(f"성공: {success_count}건 ({success_rate:.1f}%)")
    print(f"실패: {fail_count}건 ({100-success_rate:.1f}%)")
    
    return pd.DataFrame(results), success_rate

def main():
    # audit_v4.json 파일 로드 (동일 폴더 위치)
    try:
        json_path = os.path.join(current_dir, 'audit_v4.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✅ 데이터 로드 완료: {len(df)}건")
    except FileNotFoundError:
        print("❌ audit_v4.json 파일을 찾을 수 없습니다.")
        print(f"예상 경로: {json_path}")
        return
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return
    
    # HCX-005 모델만 사용
    results_005, rate_005 = process_extraction(df, "HCX-005")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = current_dir  # 스크립트와 동일한 폴더에 저장
    
    output_005 = os.path.join(output_dir, f'extraction_HCX005_{timestamp}.csv')
    
    results_005.to_csv(output_005, index=False, encoding='utf-8-sig')
    
    print(f"\n📁 결과 파일 저장 완료:")
    print(f"   - {output_005}")
    
    # 최종 리포트
    print(f"\n{'='*60}")
    print("🏆 추출 결과")
    print(f"{'='*60}")
    print(f"HCX-005 성공률: {rate_005:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()