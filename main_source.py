from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import time

app = FastAPI()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка модели и токенизатора
model_name = "ai-forever/sage-fredt5-distilled-95m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Инициализация морфологического анализатора
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Исключения для слов, которые не требуют проверки на существительное
exclusions = {"направляющая", "санки", "салазки"}

class CheckRequest(BaseModel):
    name: str

class CheckResponse(BaseModel):
    original_name: str
    has_error: bool
    errors: list[dict]
    corrected_name: str
    statistics: dict

def correct_morphology(text):
    """Исправляет морфологические ошибки в тексте с помощью модели NLP."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def fix_leet(text):
    """Исправляет leet-замену в тексте, сохраняя капитализацию."""
    leet_map = {
        '4': 'A',
        '3': 'E',
        '1': 'I',
        '0': 'O',
        '5': 'S',
        '7': 'T'
    }
    return ''.join(leet_map.get(char.lower(), char) if char.islower() else leet_map.get(char.lower(), char) for char in text)

def check_name(name):
    """Проверяет наименование по правилам и возвращает результат."""
    result = {
        'original_name': name,
        'errors': [],
        'corrected_name': name
    }

    rule_times = {
        'fix_leet': 0,
        'clean_text': 0,
        'correct_morphology': 0,
        'check_length': 0,
        'check_start_with_noun': 0,
        'check_grammar': 0,
        'check_forbidden_chars': 0,
        'check_extra_spaces': 0,
        'check_letter_e': 0,
        'check_quotation_marks': 0,
        'check_latin_chars': 0
    }

    # Исправление leet-замен
    start_time = time.time()
    cleaned_name = fix_leet(name)
    rule_times['fix_leet'] = time.time() - start_time

    # Очистка текста от ненужных символов
    start_time = time.time()
    cleaned_name = re.sub(r'[@{}\|°×\'^~‰ü°µαβ≤≥©®ø_]', '', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    cleaned_name = cleaned_name.replace('ё', 'е')
    cleaned_name = re.sub(r'["“”«»]', '', cleaned_name)
    cleaned_name = re.sub(r'[A-Za-z]', '', cleaned_name)
    rule_times['clean_text'] = time.time() - start_time

    # Проверка грамматических ошибок
    start_time = time.time()
    if not re.match(r'^[А-Яа-я\s]+$', cleaned_name):
        result['errors'].append({'error': 'Грамматическая ошибка', 'corrected': cleaned_name})
    else:
        corrected_name = correct_morphology(cleaned_name)
        if cleaned_name != corrected_name:
            result['errors'].append({'error': 'Морфологическая ошибка', 'corrected': corrected_name})
            cleaned_name = corrected_name
    rule_times['correct_morphology'] = time.time() - start_time

    # Проверка длины наименования
    start_time = time.time()
    if len(cleaned_name) > 256:
        result['errors'].append({'error': 'Длина наименования превышает допустимую норму', 'corrected': cleaned_name[:256]})
        cleaned_name = cleaned_name[:256]
    rule_times['check_length'] = time.time() - start_time

    # Проверка на начало с существительного в единственном числе
    start_time = time.time()
    doc = Doc(cleaned_name)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tokens = [token for token in doc.tokens if token.text.lower() not in exclusions]

    if doc.tokens:
        first_token = doc.tokens[0]
        if not (first_token.pos == 'NOUN' and 'Sing' in first_token.feats):
            result['errors'].append({'error': 'Наименование должно начинаться с существительного в единственном числе', 'corrected': cleaned_name})
    rule_times['check_start_with_noun'] = time.time() - start_time

    # Проверка на наличие запрещенных символов
    start_time = time.time()
    if re.search(r'[@{}\|°×\'^~‰ü°µαβ≤≥©®ø_]', cleaned_name):
        result['errors'].append({'error': 'Наличие запрещенных символов', 'corrected': cleaned_name})
    rule_times['check_forbidden_chars'] = time.time() - start_time

    # Проверка на наличие лишних пробелов
    start_time = time.time()
    if re.search(r'\s{2,}', cleaned_name):
        result['errors'].append({'error': 'Наличие лишних пробелов', 'corrected': re.sub(r'\s{2,}', ' ', cleaned_name).strip()})
    rule_times['check_extra_spaces'] = time.time() - start_time

    # Проверка на использование буквы "ё"
    start_time = time.time()
    if 'ё' in cleaned_name:
        result['errors'].append({'error': 'Использование буквы "ё"', 'corrected': cleaned_name.replace('ё', 'е')})
    rule_times['check_letter_e'] = time.time() - start_time

    # Проверка на использование кавычек
    start_time = time.time()
    if re.search(r'["“”«»]', cleaned_name):
        result['errors'].append({'error': 'Использование кавычек', 'corrected': re.sub(r'["“”«»]', '', cleaned_name)})
    rule_times['check_quotation_marks'] = time.time() - start_time

    # Проверка на использование букв латинского алфавита
    start_time = time.time()
    if re.search(r'[A-Za-z]', cleaned_name):
        result['errors'].append({'error': 'Использование букв латинского алфавита', 'corrected': re.sub(r'[A-Za-z]', '', cleaned_name)})
    rule_times['check_latin_chars'] = time.time() - start_time

    result['corrected_name'] = cleaned_name
    return result, rule_times

@app.post("/check_name", response_model=CheckResponse)
def check_name_endpoint(request: CheckRequest):
    start_time = time.time()
    result, rule_times = check_name(request.name)
    total_time = time.time() - start_time

    # Логирование метрик
    statistics = {
        'Среднее время обработки одной записи': total_time,
        'Среднее время обработки правила fix_leet': rule_times['fix_leet'],
        'Среднее время обработки правила clean_text': rule_times['clean_text'],
        'Среднее время обработки правила correct_morphology': rule_times['correct_morphology'],
        'Среднее время обработки правила check_length': rule_times['check_length'],
        'Среднее время обработки правила check_start_with_noun': rule_times['check_start_with_noun'],
        'Среднее время обработки правила check_grammar': rule_times['check_grammar'],
        'Среднее время обработки правила check_forbidden_chars': rule_times['check_forbidden_chars'],
        'Среднее время обработки правила check_extra_spaces': rule_times['check_extra_spaces'],
        'Среднее время обработки правила check_letter_e': rule_times['check_letter_e'],
        'Среднее время обработки правила check_quotation_marks': rule_times['check_quotation_marks'],
        'Среднее время обработки правила check_latin_chars': rule_times['check_latin_chars']
    }

    has_error = bool(result['errors'])

    if has_error:
        logger.info(f"Обработка записи с ошибками: {request.name} - {total_time} сек")
    else:
        logger.info(f"Обработка записи без ошибок: {request.name} - {total_time} сек")

    return CheckResponse(
        original_name=result['original_name'],
        has_error=has_error,
        errors=result['errors'],
        corrected_name=result['corrected_name'],
        statistics=statistics
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)