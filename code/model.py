import torch
import random
import numpy as np
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    do_sample=False,
    max_new_tokens=500,
    repetition_penalty=1.05,
    return_full_text=False,
)

qwen = HuggingFacePipeline(pipeline=hf_pipeline)
