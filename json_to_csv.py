import json
import pandas as pd

# 1. JSON 파일 경로
JSON_PATH = "audits.json"
CSV_PATH = "audits.csv"

# 2. JSON 로드
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. JSON -> 테이블 구조로 변환
df = pd.json_normalize(data)

# 4. CSV 저장 (한글 깨짐 방지)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"CSV 생성 완료: {CSV_PATH}")
print("컬럼 목록:")
print(df.columns.tolist())