# Redis 메모리 아키텍처 (최종)

이 문서는 AURA 에이전트를 위한 최종 Redis 기반 영구 메모리 구현에 대해 설명합니다. 이 커스텀 구현은 LangGraph의 비동기 실행(`astream`)을 지원하고, 직렬화 불가능한(non-serializable) 런타임 객체로 인한 피클링(pickling) 에러를 해결하기 위해 작성되었습니다.

## 1. 핵심 구현: `RedisSaver`
- 위치: `common/memory/redis_checkpointer.py`
- 설명: LangGraph의 `BaseCheckpointSaver`를 상속받아 대화 상태를 Redis에 저장합니다.

## 2. 키 구조 (데이터 스키마)
우리는 체크포인트와 중간 결과를 저장하기 위해 구조화된 키 패턴을 사용합니다.

| 키 패턴 (Key Pattern) | 설명 | 만료 시간 (TTL) |
| :--- | :--- | :--- |
| `checkpoint:{thread_id}:{checkpoint_id}` | 특정 시점의 실제 상태(메시지, 컨텍스트)를 저장합니다. | **24시간** (86400초) |
| `checkpoint_latest:{thread_id}` | 해당 스레드의 가장 최신 `checkpoint_id`를 가리키는 포인터입니다. | **24시간** (86400초) |
| `writes:{thread_id}:{checkpoint_id}:{task_id}` | 최종 체크포인트가 저장되기 전, 중간 산출물(예: 도구 실행 결과)을 저장합니다. | **24시간** (86400초) |

- **`thread_id`**: **세션 ID**와 동일합니다. 고유한 사용자 대화를 식별합니다.
- **`checkpoint_id`**: 대화 그래프의 각 단계에 대한 고유 ID (주로 UUID 또는 타임스탬프 등).
- **`task_id`**: 특정 노드 실행(예: 검색, 답변 생성 등)을 식별합니다.

## 3. 주요 수정 사항 및 기능

### A. 피클 안전성 (직렬화 문제 해결)
**문제**: LangGraph는 `config` 딕셔너리에 런타임 객체(예: `stream_writer` 콜백)를 주입합니다. 이 객체들은 피클링(파일/Redis 저장)이 불가능하여 `AttributeError: Can't get local object...` 에러를 유발합니다.
**해결**: 저장하기 전에 `config` 객체를 엄격하게 필터링합니다.
```python
# 복구에 필요한 안전한 키만 유지
safe_keys = {"thread_id", "checkpoint_ns", "checkpoint_id"}
safe_config = {k: v for k, v in config.items() if k in safe_keys}
pickle.dumps((checkpoint, metadata, safe_config))
```

### B. 비동기 지원 (Asyncio Fix)
**문제**: 웹 앱에서는 `agent.astream()`(비동기 스트리밍)을 사용합니다. 기본 `RedisSaver`나 단순 구현체는 동기식 `get/put`만 지원하여 비동기 루프를 차단(크래시)합니다.
**해결**: `asyncio.to_thread`를 사용하여 블로킹 Redis 작업을 깔끔하게 감싸는 `aget_tuple`, `aput`, `aput_writes`를 구현했습니다.
```python
async def aput(...):
    return await asyncio.to_thread(self.put, ...)
```

### C. `put_writes` 누락 해결 (NotImplementedError Fix)
**문제**: LangGraph는 Generator로 넘어가기 전에 "중간 쓰기"(예: 리서처가 찾은 내용)를 저장해야 합니다. `aput_writes`가 없으면 그래프 실행 중간에 실패합니다.
**해결**: 튜플 리스트를 직렬화하여 Redis에 저장하는 `put_writes` 및 `aput_writes`를 구현했습니다.

### D. 생명주기 관리 (TTL)
**문제**: Redis 메모리는 비싸므로, 오래된 대화 데이터가 무한정 쌓이면 안 됩니다.
**해결**: 모든 키에 **24시간 만료 (TTL)**를 적용했습니다.
- 사용자가 대화할 때마다 해당 세션의 TTL이 갱신됩니다.
- 사용자가 24시간 동안 활동이 없으면 세션 메모리는 자동으로 삭제됩니다.

## 4. 작동 흐름 (Work Flow)
1. **사용자 요청**: `POST /chat` (with `session_id`)
2. **상태 로드**: `aget_tuple`이 `checkpoint:{session_id}:...`에서 최신 체크포인트를 가져옵니다.
3. **그래프 실행**: 에이전트가 노드를 실행합니다 (추론 -> 검증 -> 생성).
4. **중간 저장**: `aput_writes`를 통해 중간 결과를 저장합니다.
5. **상태 저장**: `aput`을 통해 최종 상태(사용자 입력 + AI 응답)를 저장합니다.
6. **응답**: 사용자에게 스트리밍됩니다.

---
**운영자 참고용**:
- 활성 세션 조회: `redis-cli keys "checkpoint_latest:*"`
- 세션 내용 검사: `redis-cli get "checkpoint:{session_id}:{checkpoint_id}"` (바이너리 피클 데이터이므로 확인하려면 파이썬 디코딩 필요).

## 5. 컨텍스트 관리 (Context Management)
**Router**는 단순한 KV 저장소를 넘어, **대화의 문맥(Pivot)**을 능동적으로 관리합니다.
- **Pivot Detection**: 사용자가 새로운 주제(New Topic)로 넘어갔는지, 이전 주제에 대한 후속 질문(Follow-up)인지 판별합니다.
- **Memory Clearing**: 새로운 주제로 판별될 경우, 이전 턴의 검색 결과(`persist_documents`)를 강제로 초기화하여, 과거 데이터가 새로운 검색(SQL 생성 등)을 방해하지 않도록 합니다.
- **Persistence**: 후속 질문일 경우, 이전 검색 결과를 그대로 유지하여 "첫번째꺼"와 같은 지시 대명사를 해결할 수 있게 합니다.
