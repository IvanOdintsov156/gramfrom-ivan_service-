import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize
import re
import pymorphy3
import time
import spacy
import pandas as pd
import random
from kafka_logger import log_event  # Импортируем модуль для логирования

# Загрузка необходимых ресурсов для nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('punkt_tab')  # Загрузка punkt_tab для русской токенизации


grammar_correction_times = []

# Инициализация токенизатора и модели
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
morph = pymorphy3.MorphAnalyzer()

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара."""
    original_case = text
    modified = False

    log_event("Запуск нормализации наименования.")  # Логирование начала нормализации

    # Проверка, состоит ли ввод только из размеров и кодов
    if re.match(r"^[\d\*\-a-zA-Z\s]+$", text) and not re.search(r"[а-яА-Я]", text):
        return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])

    # 1. Очистка текста и удаление нежелательных префиксов
    rule_start_time = time.time_ns()
    text = text.replace('ё', 'е')

    # Удаление префиксов на основе NLP
    doc = nlp(text)
    prefixes_to_remove = [token.text for token in doc if re.match(r'^![а-яА-Я]+$', token.text)]
    for prefix in prefixes_to_remove:
        text = text.replace(prefix, '').strip()

    # Удаление нежелательных символов
    text = re.sub(r"[@\{\}\|,°×\'^~‰üµαβ≤≥©®ø]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    logger.info(f"Rule 'cleanup' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    # 2. NLP-based restructuring and grammar correction
    doc = nlp(text)
    noun_phrase = ""
    other_parts = []

    for token in doc:
        if token.pos_ == "NOUN":
            noun_phrase = token.text
            for child in token.children:
                if child.dep_ in ["amod", "nmod"]:
                    child_parsed = morph.parse(child.text)[0]
                    token_parsed = morph.parse(token.text)
                    if token_parsed and token_parsed[0].tag.case:
                        child_inflected = child_parsed.inflect({token_parsed[0].tag.case})
                        noun_phrase += " " + child_inflected.word if child_inflected else " " + child.text
                    else:
                        noun_phrase += " " + child.text
            break

        other_parts = [token.text for token in doc if token.text not in noun_phrase.split()]
        other_parts = [part for part in other_parts if not re.match(r"^\d{1,2}-[а-яА-Яa-zA-Z]+\s?$", part)]

        text = noun_phrase + " " + " ".join(other_parts)

        text = re.sub(r"(\d+)\s+(\d+)\s+(\d+)", r"\1*\2*\3", text)
        text = re.sub(r"(\d+)\s+(\d+)", r"\1*\2", text)

    # 4. Коррекция грамматики
    rule_start_time = time.time_ns()
    words = word_tokenize(text, language='russian')
    corrected_words = []
    for word in words:
        if not re.match(r"^\d{2,3}\*\d{2,3}\*\d{1,2}-[а-яА-Яa-zA-Z]?$", word) and \
           not re.match(r"^\d{3,}-\d{3,}-\d{3,}-[а-яА-Яa-zA-Z]?", word):
            parsed_word = morph.parse(word)[0]
            if 'NOUN' in parsed_word.tag and 'sing' in parsed_word.tag:
                corrected_words.append(parsed_word.normal_form)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    text = " ".join(corrected_words)

    logger.info(f"Rule 'grammar_correction' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    # Валидация
    rule_start_time = time.time_ns()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,\-]+", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\.$", "", text)
    text = correct_grammar(text)

    logger.info(f"Rule 'validation' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    # Восстановление регистра
    rule_start_time = time.time_ns()
    normalized_text = ""
    original_index = 0
    for i, char in enumerate(text):
        if original_index < len(original_case) and char.lower() == original_case[original_index].lower():
            normalized_text += original_case[original_index]
            original_index += 1
        else:
            normalized_text += char

    logger.info(f"Rule 'case_restoration' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    modified = normalized_text != original_case

    return pd.Series([normalized_text, modified], index=['Наименование(нормализованное)', 'Modified'])

def correct_grammar(sentence):
    """Корректирует грамматику с использованием моделей."""
    tokenizer_sage = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-large")
    model_sage = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-large")

    inputs_sage = tokenizer_sage(sentence, max_length=None, padding="longest", truncation=False, return_tensors="pt")
    outputs_sage = model_sage.generate(**inputs_sage.to(model_sage.device), max_length=inputs_sage["input_ids"].size(1) * 1.5)
    corrected_sentence_sage = tokenizer_sage.batch_decode(outputs_sage, skip_special_tokens=True)[0]
    print(f"Результат после sage-fredt5-large: {corrected_sentence_sage}")

    return corrected_sentence_sage

def process_excel_column(excel_file, column_name):
    """Обрабатывает указанный столбец в Excel файле и применяет нормализацию."""
    df = pd.read_excel(excel_file)

    log_event("Начало обработки столбца Excel.")  # Логирование начала обработки столбца

    df[['Наименование(нормализованное)', 'Modified']] = df[column_name].apply(lambda x: pd.Series(normalize_name(x)))

    df['Ошибка'] = df['Modified'].apply(lambda x: 'ИСТИНА' if x else 'ЛОЖЬ')
    df = df.drop(columns=['Modified'])

    log_event("Завершение обработки столбца Excel.")  # Логирование завершения обработки

    return df[[column_name, 'Наименование(нормализованное)', 'Ошибка']]

def main():
    """Основная функция для выполнения обработки данных."""
    excel_file = '10записей.xlsx'
    column_name = 'Наименование(ненормализованное)'

    log_event("Начало обработки Excel файла.")  # Логирование начала обработки файла
    result_df = process_excel_column(excel_file, column_name)

    total_records = len(result_df)
    correct_original_names = result_df['Ошибка'][result_df['Ошибка'] == 'ИСТИНА'].count()
    percentage_original = (correct_original_names / total_records) * 100

    result_df.loc[len(result_df)] = [f"Процент корректно нормализованных записей (оригинальные): {percentage_original:.2f}%", "", ""]
    
    result_df.to_excel('normalized_names.xlsx', index=False)
    print(result_df)

    log_event("Завершение обработки Excel файла.")  # Логирование завершения обработки файла

if __name__ == "__main__":
    main()
