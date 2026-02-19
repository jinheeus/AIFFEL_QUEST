import pandas as pd
import re

# ==============================================================================
# [Step 3 검증 Tracer] Final v5 로직(검증/환류 등) 적용 과정 추적
# ==============================================================================

def trace_step3_logic_final_v5(row):
    text = row['action_prep']
    doc_code = row['doc_code']
    idx = row.get('idx', row.name)
    
    trace_logs = []
    
    # 1. 금액 추출
    amount_iter = re.finditer(r'(\d+)원', text)
    
    for match in amount_iter:
        amount_str = match.group(1)
        amount_val = int(amount_str)
        
        if amount_val == 0: continue
            
        # 2. 문맥 확보 (v5와 동일한 범위 설정)
        start_idx = match.start()
        end_idx = match.end()
        
        ctx_start = max(0, start_idx - 100)
        context = text[ctx_start:start_idx] # Prefix
        
        full_context = text[max(0, start_idx - 50):min(len(text), end_idx + 50)]
        
        # Suffix (Rule 5 판단용 - 80자)
        suffix_text = text[end_idx:min(len(text), end_idx + 80)]
        trailing_context = suffix_text
        
        # --- [로그 데이터 구성] ---
        log_entry = {
            'idx': idx,
            'doc_code': doc_code,
            'amount': amount_val,
            'context': f"...{context}[{amount_val}원]{suffix_text[:40]}...", # 출력용은 짧게
            'rules_triggered': []
        }
        
        # [Rule 2] Type Check
        money_type = "기타"
        type_trigger = "None"
        
        if re.search(r'(과태료|과징금|이행강제금|벌금|가산금|벌점)', full_context):
            money_type = "징벌"; type_trigger = re.search(r'(과태료|과징금|이행강제금|벌금|가산금|벌점)', full_context).group()
        elif re.search(r'(변상|배상)', full_context):
            money_type = "배상"; type_trigger = re.search(r'(변상|배상)', full_context).group()
        elif re.search(r'(지연배상|이자|연체료)', full_context):
            money_type = "이자"; type_trigger = re.search(r'(지연배상|이자|연체료)', full_context).group()
        elif re.search(r'(시세차익|부당이득|초과지급|과다지급|오지급|잘못지급)', full_context):
            money_type = "원상복구(부당이득)"; type_trigger = re.search(r'(시세차익|부당이득|초과지급|과다지급|오지급|잘못지급)', full_context).group()
        elif re.search(r'(감액|공제|정산)', full_context):
            money_type = "사전조치"; type_trigger = re.search(r'(감액|공제|정산)', full_context).group()
        elif re.search(r'(회수|환수|반납)', full_context):
            money_type = "원상복구(일반)"; type_trigger = re.search(r'(회수|환수|반납)', full_context).group()
            
        log_entry['rules_triggered'].append(f"Rule 2 (Type): {money_type} (Trigger: '{type_trigger}')")
        
        # [Rule 3] Total/Diff
        is_total = False
        trigger = "None"
        if re.search(r'(합계|총계|소계|누계|총액)', context):
            is_total = True; trigger = re.search(r'(합계|총계|소계|누계|총액)', context).group()
        elif context.endswith('총'):
            is_total = True; trigger = "'총' (Suffix Match)"
        
        log_entry['rules_triggered'].append(f"Rule 3 (Total): {is_total} (Trigger: {trigger})")

        # [Rule 4] Target (정교화된 패턴)
        internal_pattern = r'(관련자|행위자|직원|담당자|소속|인사|인력|인원|본인)'
        external_pattern = r'(업체|시공사|계약상대자|수급인|도급|회사)'
        
        target_val = "미식별"
        trigger = "None"
        if re.search(internal_pattern, full_context):
            target_val = "대내"; trigger = re.search(internal_pattern, full_context).group()
        elif re.search(external_pattern, full_context):
            target_val = "대외"; trigger = re.search(external_pattern, full_context).group()
            
        log_entry['rules_triggered'].append(f"Rule 4 (Target): {target_val} (Trigger: {trigger})")
            
        # [Rule 5] Validity (v5 핵심: 검증/환류 등 미확정 패턴 강화)
        is_confirmed = True
        trigger = "None"
        
        # 1. 미확정 키워드 패턴
        unconfirmed_pattern = r'(방안|강구|검토|재산정|예정|환류|확인되지않는|적정성|타당성|준수여부|만족하는지|재검증|시스템검증)'
        
        if re.search(unconfirmed_pattern, trailing_context):
            is_confirmed = False
            trigger = re.search(unconfirmed_pattern, trailing_context).group()
            log_entry['rules_triggered'].append(f"Rule 5 (Validity): FALSE (Trigger: '{trigger}') -> 미확정/절차")
            
        # 2. '검증' 단독 사용 시 예외 처리 (검증결과/완료는 제외)
        elif '검증' in trailing_context and not re.search(r'(검증결과|검증완료)', trailing_context):
            is_confirmed = False
            trigger = "'검증' (Without 결과/완료)"
            log_entry['rules_triggered'].append(f"Rule 5 (Validity): FALSE (Trigger: {trigger}) -> 단순 검증 요청")
            
        else:
            log_entry['rules_triggered'].append("Rule 5 (Validity): TRUE (확정)")
            
        trace_logs.append(log_entry)
        
    return trace_logs

# 실행 및 로그 출력
if 'df_step2' in locals() and len(df_step2) > 0:
    print("🔄 [Step 3 검증 v5] Rule별 추적 (검증/환류 등 강화된 로직 적용)\n")
    
    # 100개 샘플링 (랜덤 시드 고정으로 결과 재현성 확보)
    n_samples = min(100, len(df_step2))
    sample_docs = df_step2.sample(n=n_samples, random_state=42)
    
    count = 0
    for _, row in sample_docs.iterrows():
        logs = trace_step3_logic_final_v5(row)
        for log in logs:
            count += 1
            print(f"📄 [idx: {log['idx']}] [Doc Code: {log['doc_code']}] 추출 금액: {log['amount']:,}원")
            print(f"   🔍 문맥: {log['context']}")
            print("   🛠️  [Rules Applied]")
            for rule_log in log['rules_triggered']:
                # Rule 5가 False(미확정)인 경우 강조 표시
                if "Rule 5" in rule_log and "FALSE" in rule_log:
                    print(f"      👉 {rule_log}  <-- 필터링됨")
                else:
                    print(f"      - {rule_log}")
            print("-" * 80)
            
    print(f"\n✅ 총 {count}개의 금액 추출 건에 대한 검증이 완료되었습니다.")
else:
    print("❌ 'df_step2'가 없습니다. Step 2 코드를 먼저 실행해주세요.")
