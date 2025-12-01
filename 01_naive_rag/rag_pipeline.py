from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config import Config

class RAGPipeline:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri": Config.MILVUS_URI,
                "token": Config.MILVUS_TOKEN
            },
            collection_name=Config.COLLECTION_NAME,
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
        
        # Baseline System Prompt
        self.system_prompt = """
너는 공공기관 감사보고서 기반 RAG 어시스턴트이다.

반드시 아래 원칙을 준수하라:

1. 너의 모든 답변은 RAG가 제공한 문서(context)에 근거해야 한다.
2. 문서에 명시되지 않은 내용은 절대로 추론하거나 상상하지 않는다.
3. 문서에 근거가 없으면 "제공된 문서만으로는 답변할 수 없습니다."라고 답한다.
4. 문서에 포함된 사실, 수치, 기관명, 조치 내용 등은 있는 그대로 전달한다.
5. 내용을 요약·정리할 수는 있지만 새로운 사실을 생성해서는 안 된다.
6. 문서를 인용할 때는 자연스럽게 문장 안에서 재서술한다.
7. 여러 문서를 제공받으면 서로 모순되는지 먼저 확인하고, 모순될 경우 문서별로 구분해 설명한다.
8. 감사 업무 특성상, 문서의 “기관 판단 및 조치 내용”을 최우선적으로 반영한다.
9. 질문이 모호하면 문서 범위 내에서 가장 명확한 답을 제공하되, 문서 외 해석은 하지 않는다.

너의 목표는 사용자가 문서를 정확히 이해할 수 있도록,
문서 내용을 기반으로만 정확하고 신뢰 가능한 해석을 제공하는 것이다.
"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{context}\n\ninput: {input}")
        ])
        
        self.rag_chain = (
            RunnableParallel({
                "context": self.retriever | self._format_docs,
                "input": RunnablePassthrough()
            })
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def run(self, query):
        print(f"Processing query: {query}")
        
        # For debugging/visibility, we can also retrieve separately to show docs
        # But the chain handles it. We will just invoke the chain.
        
        response = self.rag_chain.invoke(query)
        return response
