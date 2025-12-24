import torch
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

encoder_model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3", 
    device=device
)