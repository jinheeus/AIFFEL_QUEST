import os
import requests
import json
import time
from dotenv import load_dotenv

# 1. .env 파일 로드
load_dotenv()

# 환경 변수 설정 (이미 입력하신 변수명으로 매칭하세요)
# 만약 Authorization 토큰 방식이라면 그에 맞게 헤더를 수정해야 합니다.
CLOVA_STUDIO_API_KEY = os.getenv('CLOVA_STUDIO_API_KEY')
# v3 엔드포인트 URL 예시 (사용자님의 콘솔 내 '테스트 앱' 또는 '서비스 앱' URL을 복사해 넣으세요)
# 보통 https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-005 형태입니다.
API_URL = os.getenv('CLOVA_STUDIO_API_URL') 

def classify_with_hcx(action_text):
    headers = {
        'X-NCP-CLOVASTUDIO-API-KEY': CLOVA_STUDIO_API_KEY,
        'Content-Type': 'application/json',
        # 만약 API Gateway가 없다면 보통 API KEY만으로 인증하거나 Bearer 토큰을 사용합니다.
        # 가이드에 따라 Authorization 헤더가 필요할 수 있습니다.
    }

    # v3 Chat Completions 형식 데이터
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': (
                    "너는 공공기관 감사 처분 수위 분류 전문가야. "
                    "입력된 '조치사항' 문구를 분석해서 오직 다음 5가지 중 하나로만 답변해.\n"
                    "분류 기준: [파면, 해임, 정직, 경고, 주의]\n\n"
                    "분류 원칙:\n"
                    "1. 문구 내에 직접적인 징계 명칭이 있으면 그에 따름.\n"
                    "2. '중징계'라는 표현만 있으면 '정직'으로 분류.\n"
                    "3. '경징계', '감봉', '견책'은 '경고'로 분류.\n"
                    "4. 특정인 징계가 아닌 '기관 통보', '시정', '개선' 등 행정 조치는 '주의'로 분류."
                )
            },
            {
                'role': 'user',
                'content': f"조치사항: {action_text}"
            }
        ],
        'maxTokens': 20,
        'temperature': 0.1,
        'topP': 0.8,
        'repeatPenalty': 1.2
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # API 레이트 리밋 준수를 위해 1초 대기
        time.sleep(1)
        
        if response.status_code == 200:
            res_json = response.json()
            # v3 응답 구조: result -> message -> content
            return res_json['result']['message']['content'].strip()
        else:
            return f"Error: {response.status_code}, {response.text}"
            
    except Exception as e:
        return f"Exception: {str(e)}"

# 2. 단일 데이터 테스트
test_action = "관련자 A를 인사규정에 따라 징계처분(정직)하고, 재발 방지를 위해 기관 경고를 시정하시기 바랍니다."
result = classify_with_hcx(test_action)

print(f"--- 테스트 결과 ---")
print(f"입력 문구: {test_action}")
print(f"분류 결과: {result}")