import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize
import re
import pymorphy3
import logging
import time
import spacy
import pandas as pd

# Download necessary resources for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('punkt_tab')  # Download punkt_tab for Russian tokenization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer and model
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
morph = pymorphy3.MorphAnalyzer()

# Load spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара."""
    original_case = text
    modified = False

    # 1. Очистка текста и удаление нежелательных символов
    text = text.replace('ё', 'е')
    doc = nlp(text)

    # Удаление нежелательных символов
    text = re.sub(r"[@\{\}\|,°×\'^~‰üµαβ≤≥©®ø]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    logger.info(f"Cleanup applied to {text}")

     # 3. Коррекция грамматики
    text = correct_grammar(text)


 # 2. NLP-based restructuring (modified) - Now more selective
    doc = nlp(text)

    # Check if the text starts with an adjective followed by a noun or a single noun
    starts_with_adj_noun = False
    if len(doc) >= 2 and doc[0].pos_ == "ADJ" and doc[1].pos_ == "NOUN":
        starts_with_adj_noun = True

    starts_with_noun = False
    if len(doc) > 0 and doc[0].pos_ == "NOUN":
        starts_with_noun = True

    # Restructure only in specific cases to avoid unnecessary changes
    if not starts_with_adj_noun and not starts_with_noun and not re.fullmatch(r"[\d\*\-a-zA-Z\s,\(\)]+", text) and any(token.pos_ == "NOUN" for token in doc):
        noun_phrase = ""
        other_parts = []

        # Find the main noun (head of the noun phrase)
        main_noun = None
        for token in doc:
            if token.pos_ == "NOUN" and token.dep_ == "ROOT":
                main_noun = token
                break

        # If a main noun is found, build the noun phrase
        if main_noun:
            noun_phrase = main_noun.text

            # Process children only if they are ADJ or NOUN modifiers
            relevant_children = [child for child in main_noun.children if child.dep_ in ["amod", "nmod"] and not child.like_num and child.pos_ in ["ADJ", "NOUN"]]

            if relevant_children:  # Proceed only if there are relevant children
                for child in relevant_children:
                    child_parsed = morph.parse(child.text)[0]

                    # Get case from spaCy and map to pymorphy3 format
                    spacy_case = main_noun.morph.get("Case")
                    pymorphy_case = {'Nom': 'nomn', 'Gen': 'gent', 'Dat': 'datv', 'Acc': 'accs', 'Ins': 'ablt', 'Loc': 'loct'}.get(spacy_case[0] if spacy_case else None)

                    # Inflect if pymorphy_case is valid and the child is inflectable
                    child_inflected = child_parsed.inflect({pymorphy_case}) if pymorphy_case else None

                    # Add the inflected child to the noun phrase, prepending or appending based on dependency
                    if child_inflected and child.dep_ == 'amod':  # Prepend adjectives
                        noun_phrase = child_inflected.word + " " + noun_phrase
                    elif child_inflected:  # Append other modifiers
                        noun_phrase += " " + child_inflected.word
                    else:
                         # Append original if not inflectable
                         noun_phrase += " " + child.text

        # Extract other parts, excluding numbers
        if noun_phrase:
            other_parts = [token.text for token in doc if token.text not in noun_phrase.split() and not token.like_num]
        else:
            other_parts = [token.text for token in doc if not token.like_num]

        # Reconstruct the normalized name, moving the noun phrase to the beginning
        text = noun_phrase + " " + " ".join(other_parts)

    modified = text != original_case

    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])


def correct_grammar(sentence):
    """Корректирует грамматику с помощью предобученной модели."""
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-large")

    inputs = tokenizer(sentence, max_length=None, padding="longest", truncation=False, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_length=inputs["input_ids"].size(1) * 1.5)
    corrected_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return corrected_sentence

def process_excel_column(excel_file, column_name):
    """Обрабатывает указанный столбец в Excel-файле и применяет нормализацию."""
    df = pd.read_excel(excel_file)

    # Применяем нормализацию и получаем статус изменений
    df[['Наименование(нормализованное)', 'Modified']] = df[column_name].apply(lambda x: pd.Series(normalize_name(x)))

    # Устанавливаем 'Ошибка' на основе статуса изменений
    df['Ошибка'] = df['Modified'].apply(lambda x: 'ИСТИНА' if x else 'ЛОЖЬ')

    df = df.drop(columns=['Modified'])  # Удаляем временный столбец 'Modified'

    return df[[column_name, 'Наименование(нормализованное)', 'Ошибка']]

def main():
    """Основная функция для выполнения обработки данных."""
    excel_file = '10записей.xlsx'
    column_name = 'Наименование(ненормализованное)'

    result_df = process_excel_column(excel_file, column_name)

    # Подсчет процента корректно нормализованных записей
    total_records = len(result_df)
    correct_original_names = result_df['Ошибка'].value_counts().get('ИСТИНА', 0)

    percentage_original = (correct_original_names / total_records) * 100 if total_records > 0 else 0

    # Добавляем строку с процентом в итоговый DataFrame
    result_df.loc[len(result_df)] = [f"Процент корректно нормализованных записей (оригинальные): {percentage_original:.2f}%", "", ""]

    result_df.to_excel('normalized_names.xlsx', index=False)
    print(result_df)

if __name__ == "__main__":
    main()