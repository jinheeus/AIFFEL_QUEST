import requests
import json
import os
import uuid
from dotenv import load_dotenv

# ==============================================================================
# 1. 환경 설정 및 API 키 확인
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
env_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=env_path)

CLOVA_API_KEY = os.getenv("CLOVASTUDIO_API_KEY")

print("=" * 60)
print("🚀 HyperCLOVA X 모델 전체 연결 진단 도구")
print("=" * 60)

if not CLOVA_API_KEY:
    print("❌ [오류] .env 파일에서 API Key를 찾을 수 없습니다.")
    exit()

print(f"🔑 API Key 확인: {CLOVA_API_KEY[:5]}..." + "*"*10)

# ==============================================================================
# 2. 모델별 라우팅 설정 (Routing Config)
# ==============================================================================
# 각 모델이 사용하는 URL과 프로토콜 버전을 정의합니다.
MODEL_CONFIGS = {
    # [Group A] Legacy 모델 (기존 v1 방식)
    "HCX-003":      {"type": "legacy", "url": "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"},
    "HCX-DASH-001": {"type": "legacy", "url": "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001"},

    # [Group B] HyperCLOVA X 신형 모델 (v3 방식)
    "HCX-005":      {"type": "v3",     "url": "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"},
    "HCX-007":      {"type": "v3",     "url": "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"},
    "HCX-DASH-002": {"type": "v3",     "url": "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002"},
}

# ==============================================================================
# 3. 연결 테스트 로직
# ==============================================================================
def check_connection(model_name, config):
    url = config["url"]
    model_type = config["type"]
    
    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # --- [Payload 구성: 모델 타입에 따라 다르게 조립] ---
    if model_type == "legacy":
        # 구형 방식 (v1)
        data = {
            "messages": [
                {"role": "system", "content": "시스템"},
                {"role": "user", "content": "연결 확인. 짧게 답해."}
            ],
            "maxTokens": 50,
            "temperature": 0.5,
            "includeAiFilters": True
        }
    
    elif model_type == "v3":
        # 신형 방식 (v3) - 엄격한 Body 규격 준수
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "연결 확인. 짧게 답해."
                        }
                    ]
                }
            ],
            "topP": 0.8,
            "topK": 0,
            "temperature": 0.1,
            "stop": [],
            "includeAiFilters": True,
            "seed": 0
        }
        # ⚠️ [중요] HCX-007의 'Invalid parameter: maxTokens' 에러 방지
        # 007을 제외한 나머지 v3 모델에는 maxTokens를 명시
        if model_name != "HCX-007":
            data["maxTokens"] = 50

    # --- [요청 전송] ---
    print(f"\n📡 [{model_name}] 연결 시도 ({model_type.upper()} Protocol)...")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            res_json = response.json()
            if 'result' in res_json:
                content = res_json['result']['message']['content'].strip()
                print("   ✅ [성공] 연결되었습니다!")
                print(f"      ㄴ 응답: \"{content}\"")
                return True
            else:
                print(f"   ⚠️ [주의] 200 OK이나 응답 구조가 다름: {res_json}")
                return False
        else:
            print(f"   ❌ [실패] HTTP {response.status_code}")
            # 에러 메시지가 너무 길면 잘라서 출력
            err_msg = response.text
            print(f"      ㄴ 에러: {err_msg if len(err_msg) < 200 else err_msg[:200] + '...'}")
            return False

    except Exception as e:
        print(f"   ❌ [시스템 에러] {e}")
        return False

# ==============================================================================
# 4. 메인 실행 루프
# ==============================================================================
if __name__ == "__main__":
    success_count = 0
    total_models = len(MODEL_CONFIGS)
    
    for model_name, config in MODEL_CONFIGS.items():
        if check_connection(model_name, config):
            success_count += 1
            
    print("\n" + "=" * 60)
    print(f"🏆 최종 진단 결과: {success_count} / {total_models} 모델 연결 성공")
    
    if success_count == total_models:
        print("🎉 모든 모델이 정상 작동합니다!")
    else:
        print("⚠️ 일부 모델 연결 실패. 위 로그의 에러 메시지를 확인하세요.")
    print("=" * 60)