import torch
from razdel import tokenize, sentenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize
import re
import pymorphy3
import logging
import time
import spacy
import pandas as pd
import random
from kafka_logger import KafkaLogger  # Импортируем KafkaLogger

# Download necessary resources for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('punkt_tab')

# Kafka configuration
kafka_conf = {'bootstrap.servers': 'localhost:9092'}  # Укажите адрес вашего Kafka сервера
kafka_topic = 'logs_topic'  # Укажите название вашего топика
kafka_logger = KafkaLogger(kafka_conf, kafka_topic)  # Инициализация KafkaLogger

# Initialize tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m").to(device)
morph = pymorphy3.MorphAnalyzer()

# Load spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара."""
    original_case = text
    modified = False
    
    # 1. Очистка текста and remove unwanted prefixes
    start_time = time.time_ns()
    logger.info("Запуск процесса обработки данных.")
    kafka_logger.send_log("Запуск процесса обработки данных", 0)

    text = text.replace('ё', 'е')
    
    # NLP-based prefix removal
    doc = nlp(text)
    prefixes_to_remove = [token.text for token in doc if re.match(r'^![а-яА-Я]+$', token.text)]
    for prefix in prefixes_to_remove:
        text = text.replace(prefix, '').strip()

    # Remove unwanted characters except for brackets
    text = re.sub(r"[@\{\}\|,°×\'^~‰üµαβ≤≥©®ø]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    logger.info(f"Правило 'очистка' применено к {text} (время: {time.time_ns() - start_time} ns)")
    kafka_logger.send_log("Очистка", time.time_ns() - start_time)

    # 2. NLP-based restructuring and grammar correction
    rule_start_time = time.time_ns()
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

    # 4. Grammar Correction
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
    
    logger.info(f"Правило 'коррекция грамматики' применено к {text} (время: {time.time_ns() - rule_start_time} ns)")
    kafka_logger.send_log("Коррекция грамматики", time.time_ns() - rule_start_time)

    # Validation
    rule_start_time = time.time_ns()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,\-]+", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\.$", "", text)

    logger.info(f"Правило 'валидация' применено к {text} (время: {time.time_ns() - rule_start_time} ns)")
    kafka_logger.send_log("Валидация", time.time_ns() - rule_start_time)

    # Case Restoration
    normalized_text = ""
    original_index = 0
    for i, char in enumerate(text):
        if original_index < len(original_case) and char.lower() == original_case[original_index].lower():
            normalized_text += original_case[original_index]
            original_index += 1
        else:
            normalized_text += char

    modified = text != original_case
    
    logger.info(f"Завершение обработки записи: {text} (время: {time.time_ns() - start_time} ns)")
    kafka_logger.send_log("Завершение обработки записи", time.time_ns() - start_time)
    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])

def process_excel_column(excel_file, column_name):
    """Processes the specified column in the Excel file and applies normalization."""
    df = pd.read_excel(excel_file)

    # Apply normalization and get modification status
    df[['Наименование(нормализованное)', 'Modified']] = df[column_name].apply(lambda x: pd.Series(normalize_name(x)))

    # Set 'Ошибка' based on modification status
    df['Ошибка'] = df['Modified'].apply(lambda x: 'ИСТИНА' if x else 'ЛОЖЬ') 

    # Introduce artifacts and re-normalize
    df['Искаженное наименование'] = df['Наименование(нормализованное)'].apply(introduce_artifacts)
    df[['Наименование(нормализованное с искажениями)', 'Modified']] = df['Искаженное наименование'].apply(lambda x: pd.Series(normalize_name(x)))
    
    # Set 'Ошибка (искаженное)' to always be 'ЛОЖЬ' for distorted names
    df['Ошибка (искаженное)'] = 'ЛОЖЬ'  

    df = df.drop(columns=['Modified'])  # Remove the temporary 'Modified' column

    return df[[column_name, 'Наименование(нормализованное)', 'Ошибка', 'Искаженное наименование', 'Наименование(нормализованное с искажениями)', 'Ошибка (искаженное)']]

def introduce_artifacts(text):
    """Introduces random artifacts into the text."""
    artifacts = ["@", "#", "$", "%", "&", "*", "(", ")", "-", "_", "+", "=", "[", "]", "{", "}", "|", "\\", ";", ":", "'", '"', ",", "<", ">", ".", "/", "?"]
    num_artifacts = random.randint(1, 3)  # Introduce 1 to 3 artifacts
    
    for _ in range(num_artifacts):
        artifact = random.choice(artifacts)
        insert_position = random.randint(0, len(text))
        text = text[:insert_position] + artifact + text[insert_position:]
    
    return text

def main():
    """Main function to execute the data processing."""
    excel_file = 'data1.xlsx'
    column_name = 'Наименование(ненормализованное)'

    result_df = process_excel_column(excel_file, column_name)

    # Calculate percentage of correctly normalized records
    total_records = len(result_df)
    correct_original_names = result_df['Ошибка'][result_df['Ошибка'] == 'ИСТИНА'].count()
    correct_distorted_names = result_df['Ошибка (искаженное)'][result_df['Ошибка (искаженное)'] == 'ИСТИНА'].count()

    percentage_original = (correct_original_names / total_records) * 100
    percentage_distorted = (correct_distorted_names / total_records) * 100

    # Add percentage rows to the DataFrame
    result_df.loc[len(result_df)] = [f"Процент корректно нормализованных записей (оригинальные): {percentage_original:.2f}%", "", "", "", "", ""]
    result_df.loc[len(result_df)] = [f"Процент корректно нормализованных записей (искаженные): {percentage_distorted:.2f}%", "", "", "", "", ""]

    result_df.to_excel('normalized_names.xlsx', index=False)
    print(result_df)

if __name__ == "__main__":
    main()