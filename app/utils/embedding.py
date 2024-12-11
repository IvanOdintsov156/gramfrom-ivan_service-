import torch
from transformers import AutoTokenizer, AutoModel
import re
from functools import lru_cache

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


@lru_cache(maxsize=1000)
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def extract_numbers_and_abbreviations(text):
    uppercase_pattern = r"\b[A-Z][A-Z\-,\.\/]*\b"  # только заглавные
    number_pattern = r"\b\w*\d[\w\-,\.\/]*\b"  # содержат числа
    abbreviations = set(re.findall(rf"{uppercase_pattern}|{number_pattern}", text))
    return abbreviations


def compare_numbers(item_abbrs, record_abbrs):
    if item_abbrs == record_abbrs:
        return 1
    return 0.5
