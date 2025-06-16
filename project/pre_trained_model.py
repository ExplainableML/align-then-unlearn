from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Tuple


def load_pre_trained_llm(
    model_name: str, tokenizer_name: str, revision: str = "main", **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_pre_trained_text_embedding_model(
    model_name: str, tokenizer_name: str, **kwargs
) -> Tuple[AutoModel, AutoTokenizer]:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer
