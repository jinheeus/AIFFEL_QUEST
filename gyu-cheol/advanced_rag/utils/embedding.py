import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_naver import ClovaXEmbeddings

load_dotenv()

embeddings_openai = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

embeddings_hcx = ClovaXEmbeddings(
    model="bge-m3",
    api_key=os.getenv("HCX_API_KEY")
)