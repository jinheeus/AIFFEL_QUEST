from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from model_factory import ModelFactory
from state import AgentState


def chat_worker(state: AgentState):
    print("--- [ChatWorker] Handling Chit-Chat ---")
    query = state["query"]

    # 단순 대화 프롬프트 정의 (Simple Chat Prompt)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI Assistant specializing in Korean Audit & Public Data.
        The user input is classified as "Chit-Chat" (casual conversation).

        Guidelines:
        1. IF the user says "Hello" (안녕) or asks "Who are you?":
           - Introduce yourself as "Audit & Public Data AI Assistant" (감사 및 공공데이터 AI 비서).
           - Mention you cover "Korea Board of Audit and Inspection" (BAI) and "Alio" (Public Data).
        
        2. IF the user mentions "BAI" or "Alio" or asks for their websites/addresses:
           - Provide the OFFICIAL URLs clearly:
             - Board of Audit and Inspection (BAI): [https://www.bai.go.kr](https://www.bai.go.kr)
             - Alio (Public Institution Info): [https://www.alio.go.kr](https://www.alio.go.kr)
        
        3. IF the user says "I see", "Okay", "Right" (e.g., "그렇구나", "그래", "알겠어"):
           - These are just interjections acknowledging your previous answer.
           - React politely (e.g., "네," "도움이 되어 기쁩니다.", "궁금한 점이 있으면 또 물어보세요.").
           - DO NOT treat "그렇구나" as a name or noun.
        
        4. IF the user says something else (e.g., "Good job", "No"):
           - Respond naturally and conversationally.
           - Be polite, concise, and helpful.
        
        User Input: {query}
        """
    )

    llm = ModelFactory.get_rag_model(level="light")  # 채팅용 경량 모델 사용
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": query})
        return {"answer": response}
    except Exception as e:
        print(f"[ChatWorker] 오류 발생: {e}")
        return {"answer": "죄송합니다. 일시적인 오류가 발생했습니다."}
