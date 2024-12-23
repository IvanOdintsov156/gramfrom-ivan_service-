import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.tokenize import word_tokenize
import re
import pymorphy3
import logging
import time
import spacy
import pandas as pd
from IPython.display import display, clear_output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.width', None)  # Auto-detect the width for the terminal
pd.set_option('display.max_colwidth', None) # Prevent column truncation
# Download necessary resources for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('punkt_tab')  # Download punkt_tab for Russian tokenization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer and model
device = "cpu"
MAX_LENGTH = 512
morph = pymorphy3.MorphAnalyzer()
# Spell check dictionary
spell_check_dict = {
         "Риле": "Реле",
         "кантакта": "контакта",
         "Релень": "Ремень",  # Add "Релень" to the dictionary
         "Хамут": "Хомут",
         "Релень": "Ремень"  
     }
# Load spaCy Russian model
nlp = spacy.load("ru_core_news_sm")
grammar_pipeline = pipeline("text2text-generation", model="ai-forever/sage-fredt5-large")
def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара."""
    original_case = text
    modified = False

    text = re.sub(r"(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)", r"\1*\2*\3", text)


    # 2. Grammar correction
    corrected_text = grammar_pipeline(text, max_length=max_length)[0]['generated_text']
    text = reorder_name(corrected_text, original_case)

    for misspelled, correct in spell_check_dict.items():
         text = text.replace(misspelled, correct)


    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])


def reorder_name(text, original_case):
    """Reorders the words in the text, preserving original case."""
    doc = nlp(text)
    noun_phrase = ""
    other_parts = []

    # 1. Identify the main noun phrase
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

    # 2. Gather other parts of the name
    other_parts = [token.text for token in doc if token.text not in noun_phrase.split()]

    # 3. Remove original patterns (e.g., "1-шт") if present
    other_parts = [part for part in other_parts if not re.match(r"^\d{1,2}-[а-яА-Яa-zA-Z]+\s?$", part)]

    # 4. Reconstruct the normalized name
    text = noun_phrase + " " + " ".join(other_parts)
    
    # 5. Capitalize only the first letter
    text = text[0].upper() + text[1:]
    
    return text # Return the reordered name


    

def process_excel_column(excel_file, column_name):
    """Обрабатывает указанный столбец в Excel-файле и применяет нормализацию."""
    df = pd.read_excel(excel_file)
    
    processed_rows = []

    for index, row in df.iterrows():
        normalized_data = normalize_name(row[column_name])
        
        processed_row = {
            column_name: row[column_name],
            'Наименование(нормализованное)': normalized_data['Наименование(нормализованное)'],
            'Ошибка': 'ИСТИНА' if normalized_data['Modified'] else 'ЛОЖЬ' 
        }
        
        processed_rows.append(processed_row)
        
        current_df = pd.DataFrame(processed_rows)
        
        # Clear previous output and display the updated table
        clear_output(wait=True) 
        display(current_df)
        
    result_df = current_df

    return result_df

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