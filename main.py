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
from kafka_logger import log_event  # Импортируем функцию логирования из kafka_logger

# Download necessary resources for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('punkt_tab')  # Download punkt_tab for Russian tokenization
grammar_correction_times = []

# Initialize tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m").to(device)
morph = pymorphy3.MorphAnalyzer()

# Load spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара, игнорируя единицы измерения."""
    original_case = text
    modified = False
    log_event("Запуск процесса обработки данных.")

    # Список единиц измерения, которые нужно игнорировать
    units_of_measurement = ["м", "кг", "см", "мм", "л", "г", "шт", "м²", "м³", "км", "мл", "гц", "т", "дм", "мг", "мкг"]

    # Проверка, состоит ли ввод только из размеров и кодов
    if re.match(r"^[\d\*\-a-zA-Z\s]+$", text) and not re.search(r"[а-яА-Я]", text):
        return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])

    # 1. Очистка текста и удаление нежелательных префиксов
    rule_start_time = time.time_ns()
    text = text.replace('ё', 'е')

    # NLP-based prefix removal
    doc = nlp(text)
    prefixes_to_remove = [token.text for token in doc if re.match(r'^![а-яА-Я]+$', token.text)]
    for prefix in prefixes_to_remove:
        text = text.replace(prefix, '').strip()

    # Удаление нежелательных символов, кроме скобок
    text = re.sub(r"[@\{\}\|,°×\'^~‰üµαβ≤≥©®ø]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    log_event(f"Правило 'очистка' применено к {text} (время: {time.time_ns() - rule_start_time} нс)")

    # 2. NLP-based restructuring and grammar correction
    doc = nlp(text)
    noun_phrase = ""
    other_parts = []

    for token in doc:
        if token.pos_ == "NOUN":
            noun_phrase = token.text
            for child in token.children:
                if child.dep_ in ["amod", "nmod"]:
                    # Обработка склонения: склоняем прилагательное под существительное
                    child_parsed = morph.parse(child.text)[0]
                    # Проверка, можно ли склонять токен
                    token_parsed = morph.parse(token.text)
                    if token_parsed and token_parsed[0].tag.case:
                        child_inflected = child_parsed.inflect({token_parsed[0].tag.case})
                        noun_phrase += " " + child_inflected.word if child_inflected else " " + child.text
                    else:
                        noun_phrase += " " + child.text
            break  # Остановиться после нахождения первого существительного

        # Извлечение других частей текста (размеры, код и т.д.)
        other_parts = [token.text for token in doc if token.text not in noun_phrase.split()]

        # Исключаем единицы измерения из других частей текста
        other_parts = [part for part in other_parts if part not in units_of_measurement]

        # Удаляем оригинальный шаблон (например, "12-я ") из других частей
        other_parts = [part for part in other_parts if not re.match(r"^\d{1,2}-[а-яА-Яa-zA-Z]+\s?$", part)]

        # Восстановление нормализованного имени
        text = noun_phrase + " " + " ".join(other_parts)

        # Специальная обработка для размеров: добавляем "*" между числами
        text = re.sub(r"(\d+)\s+(\d+)\s+(\d+)", r"\1*\2*\3", text)  # Добавляем '*' между тремя числами
        text = re.sub(r"(\d+)\s+(\d+)", r"\1*\2", text)  # Добавляем '*' между двумя числами

    # 4. Коррекция грамматики (пересмотрена для любого единственного существительного)
    rule_start_time = time.time_ns()
    words = word_tokenize(text, language='russian')
    corrected_words = []
    for word in words:
        # Исключаем размеры и коды
        if not re.match(r"^\d{2,3}\*\d{2,3}\*\d{1,2}-[а-яА-Яa-zA-Z]?$", word) and \
           not re.match(r"^\d{3,}-\d{3,}-\d{3,}-[а-яА-Яa-zA-Z]?", word):
            
            parsed_word = morph.parse(word)[0]
            # Корректируем любое единственное существительное
            if 'NOUN' in parsed_word.tag and 'sing' in parsed_word.tag:
                corrected_words.append(parsed_word.normal_form)  # Используем нормальную форму
            else:
                corrected_words.append(word)  # Оставляем другие слова без изменений
        else:
            corrected_words.append(word)  # Оставляем размеры и коды без изменений

    text = " ".join(corrected_words)

    log_event(f"Правило 'коррекция грамматики' применено к {text} (время: {time.time_ns() - rule_start_time} нс)")

    # Валидация: Удаление запрещенных символов, лишних пробелов и завершающих точек
    rule_start_time = time.time_ns()

    # Разрешаем только алфавитные, пробелы, запятые, точки, дефисы и кавычки для дюймов
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,\-]+", "", text)  # Удаляем другую пунктуацию

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Удаляем завершающую точку
    text = re.sub(r"\.$", "", text)

    log_event(f"Правило 'валидация' применено к {text} (время: {time.time_ns() - rule_start_time} нс)")

    text = correct_grammar(text)
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

    log_event(f"Правило 'восстановление регистра' применено к {text} (время: {time.time_ns() - rule_start_time} нс)")

    modified = text != original_case  # Устанавливаем modified в True, если текст изменился

    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])


def correct_grammar(sentence):
    start_time = time.time_ns()  
    inputs = tokenizer(sentence, max_length=None, padding="longest", truncation=False, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_length=inputs["input_ids"].size(1) * 1.5)
    end_time = time.time_ns()  # Get the end time in nanoseconds
    execution_time_ns = end_time - start_time  # Calculate the time difference
    grammar_correction_times.append(execution_time_ns)  # Append execution time to the list
    log_event(f"Коррекция грамматики заняла {execution_time_ns} нс")  # Log the time taken
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] 

def process_excel_column(excel_file, column_name):
    """Processes the specified column in the Excel file and applies normalization."""
    df = pd.read_excel(excel_file)

    # Apply normalization and get modification status
    df[['Наименование(нормализованное)', 'Modified']] = df[column_name].apply(lambda x: pd.Series(normalize_name(x)))

    # Set 'Ошибка' based on modification status (ИСТИНА if modified, ЛОЖЬ otherwise)
    df['Ошибка'] = df['Modified'].apply(lambda x: 'ИСТИНА' if x else 'ЛОЖЬ') 

    # Introduce artifacts and re-normalize
    df['Искаженное наименование'] = df['Наименование(нормализованное)'].apply(introduce_artifacts)
    df[['Наименование(нормализованное с искажениями)', 'Modified']] = df['Искаженное наименование'].apply(lambda x: pd.Series(normalize_name(x)))
    
    # Set 'Ошибка (искаженное)' based on modification status
    df['Ошибка (искаженное)'] = df['Modified'].apply(lambda x: 'ИСТИНА' if x else 'ЛОЖЬ')  

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
    excel_file = 'data1 (1).xlsx'
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

    # Calculate average time per record for grammar correction
    total_grammar_correction_time_ns = sum(grammar_correction_times)
    num_records_processed = len(grammar_correction_times)  # Each call to correct_grammar corresponds to one record

    if num_records_processed > 0:  # Avoid division by zero
        average_time_per_record_ns = total_grammar_correction_time_ns / num_records_processed
        result_df.loc[len(result_df)] = [f"Среднее время коррекции грамматики на запись: {average_time_per_record_ns:.2f} нс", "", "", "", "", ""]
        print(f"Среднее время коррекции грамматики на запись: {average_time_per_record_ns:.2f} нс")  
    else:
        print("Не было обработано записей для коррекции грамматики.")

    result_df.to_excel('normalized_names.xlsx', index=False)
    print(result_df)
    
    log_event("Завершение процесса обработки данных.")

if __name__ == "__main__":
    main()
