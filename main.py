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
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m").to(device)
morph = pymorphy3.MorphAnalyzer()

# Load spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

def normalize_name(text, max_length=MAX_LENGTH):
    """Нормализует наименование товара."""
    original_case = text
    modified = False
    # Check if the input consists only of dimensions and codes
    if re.match(r"^[\d\*\-a-zA-Z\s]+$", text) and not re.search(r"[а-яА-Я]", text):
        return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])  


    # 1. Очистка текста and remove unwanted prefixes
    rule_start_time = time.time_ns()
    text = text.replace('ё', 'е')

    # NLP-based prefix removal
    doc = nlp(text)
    prefixes_to_remove = [token.text for token in doc if re.match(r'^![а-яА-Я]+\!$', token.text)]
    for prefix in prefixes_to_remove:
        text = text.replace(prefix, '').strip()

    # Remove unwanted characters except for brackets
    text = re.sub(r"[@\{\}\|,°×\'^~‰üµαβ≤≥©®ø]", "", text)  # Updated regex to exclude brackets
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
                    # Handle declension: inflect the adjective to match the noun's case
                    child_parsed = morph.parse(child.text)[0]
                    # Check if token.text is parsable and has a case attribute
                    token_parsed = morph.parse(token.text)
                    if token_parsed and token_parsed[0].tag.case:
                        child_inflected = child_parsed.inflect({token_parsed[0].tag.case})
                        noun_phrase += " " + child_inflected.word if child_inflected else " " + child.text
                    else:
                        # If token is not parsable or has no case, keep the original child text
                        noun_phrase += " " + child.text
            break  # Stop after finding the first noun phrase

        # Extract other parts of the text (dimensions, code, etc.)
        other_parts = [token.text for token in doc if token.text not in noun_phrase.split()]


         # Remove the original pattern (e.g., "12-я ") from other_parts
        other_parts = [part for part in other_parts if not re.match(r"^\d{1,2}-[а-яА-Яa-zA-Z]+\s?$", part)]

        # Reconstruct the normalized name
        text = noun_phrase + " " + " ".join(other_parts)

        # Special handling for dimensions: add "*" between numbers
        text = re.sub(r"(\d+)\s+(\d+)\s+(\d+)", r"\1*\2*\3", text)  # Add '*' between three numbers
        text = re.sub(r"(\d+)\s+(\d+)", r"\1*\2", text)  # Add '*' between two numbers

    # 4. Grammar Correction (revised for any singular noun)
    rule_start_time = time.time_ns()
    words = word_tokenize(text, language='russian')
    corrected_words = []
    for word in words:
        # Exclude dimensions and codes
        if not re.match(r"^\d{2,3}\*\d{2,3}\*\d{1,2}-[а-яА-Яa-zA-Z]?$", word) and \
           not re.match(r"^\d{3,}-\d{3,}-\d{3,}-[а-яА-Яa-zA-Z]?", word):
            
            parsed_word = morph.parse(word)[0]
            # Correct any singular noun
            if 'NOUN' in parsed_word.tag and 'sing' in parsed_word.tag:
                corrected_words.append(parsed_word.normal_form)  # Use normal form
            else:
                corrected_words.append(word)  # Keep other words as is
        else:
            corrected_words.append(word)  # Keep dimensions and codes as is
    
    text = " ".join(corrected_words)
    
    logger.info(f"Rule 'grammar_correction' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    # Validation: Remove prohibited characters, extra spaces, and trailing periods
    rule_start_time = time.time_ns()
    
    # Allow only alphanumeric, spaces, commas, periods, hyphens, and quotation marks for inches
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,\-]+", "", text)  # Remove other punctuation

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove trailing period
    text = re.sub(r"\.$", "", text)

    logger.info(f"Rule 'validation' applied to {text} (took {time.time_ns() - rule_start_time} ns)")

    # Case Restoration
    normalized_text = ""
    original_index = 0
    for i, char in enumerate(text):
        if original_index < len(original_case) and char.lower() == original_case[original_index].lower():
            normalized_text += original_case[original_index]
            original_index += 1
        else:
            normalized_text += char

    modified = text != original_case # Set modified to True if text has changed
    
    logger.info(f"Finished processing record: {text} (took {time.time_ns() - rule_start_time} ns)")
    return pd.Series([text, modified], index=['Наименование(нормализованное)', 'Modified'])




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