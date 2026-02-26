import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_naver import ChatClovaX

load_dotenv()

llm_openai = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0, 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm_hcx = ChatClovaX(
    model="HCX-DASH-002", 
    max_tokens=1024,
    temperature=0,
    api_key=os.getenv("HCX_API_KEY")
)