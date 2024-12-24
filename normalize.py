import re
import pandas as pd
import pymorphy3
from transformers import pipeline
from config import MAX_LENGTH, spell_check_dict
from reorder_name import reorder_name  # Импортируем функцию reorder_name

# Инициализация
morph = pymorphy3.MorphAnalyzer()

try:
    grammar_pipeline = pipeline("text2text-generation", model="ai-forever/sage-fredt5-large")
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели: {e}")

def normalize_name(text):
    """Нормализует наименование товара."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Входной текст должен быть непустой строкой.")

    original_case = text
    modified = False

    # Грамматическая корректировка
    try:
        corrected_text = grammar_pipeline(text, max_length=MAX_LENGTH)[0]['generated_text']
        logger.info(f"Грамматическая корректировка: {corrected_text}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при грамматической корректировке: {e}")

    text = reorder_name(corrected_text, original_case)

    for misspelled, correct in spell_check_dict.items():
        text = text.replace(misspelled, correct)

    text = re.sub(r"(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)", r"\1*\2*\3", text)

    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])
