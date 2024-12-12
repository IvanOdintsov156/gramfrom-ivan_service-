from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
import time
import json
from datetime import datetime

app = FastAPI()
nltk.download('stopwords')
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka import avro

class LogMessage:
    def __init__(self, event_type: str, description: str):
        self.time = datetime.now()
        self.event_type = event_type
        self.description = description
        self.start_time = time.time()
        try:
            self.producer = SerializingProducer(
                {"bootstrap.servers": "localhost:9092", "key.serializer": StringSerializer("utf-8"), "value.serializer": avro_serializer}
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None  # Set to None if initialization fails

    def end(self) -> dict:
        duration = time.time() - self.start_time
        log_entry = {
            "time": self.time.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "duration": f"{duration:.3f}s"
        }
        self.send_log(log_entry)
        return log_entry

    def send_log(self, log_entry: dict):
        """Send log entry to Kafka consumer."""
        if self.producer:
            try:
                self.producer.produce(topic="log_topic", key=str(self.time), value=log_entry)
                self.producer.flush()
            except Exception as e:
                logger.error(f"Failed to send log entry to Kafka: {e}")

def avro_serializer(obj, ctx):
    """Avro serializer for confluent_kafka."""
    schema = avro.loads(
        """
        {
            "namespace": "log",
            "name": "log_entry",
            "type": "record",
            "fields": [
                {"name": "time", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "description", "type": "string"},
                {"name": "duration", "type": "string"}
            ]
        }
        """
    )
    return avro.serializer.encode_record_with_schema(schema, obj)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка моделей
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m", token="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")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m", token="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")

nlp = spacy.load("ru_core_news_md")  # Load spaCy for Russian
morph = pymorphy3.MorphAnalyzer()

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')

class CheckRequest(BaseModel):
    name: str

class CheckResponse(BaseModel):
    original_name: str
    corrected_name: str
    has_error: bool
    status: str

def log_event(event_type: str, description: str) -> LogMessage:
    """Создает новый объект логирования и возвращает его."""
    log_msg = LogMessage(event_type, description)
    logger.info(json.dumps(log_msg.__dict__, default=str))
    return log_msg

async def context_aware_leet_replacement(text: str) -> str:
    """Контекстно-зависимая версия с использованием маскированного языкового моделирования"""
    leet_map = {
        '0': 'о', '4': 'а', '@': 'а', '6': 'б', '8': 'в',
        '9': 'д', '3': 'е', '€': 'е', '}{': 'ж', '%': 'ж',
        '3': 'з', '1': 'и', '|': 'и', '!': 'и', 'l<': 'к',
        '/\\': 'л', '1': 'л', '/v\\': 'м', '|\\/|': 'м',
        '/\\/': 'н', '|\\|': 'н', '0': 'о', '|D': 'р',
        '|>': 'р', 'c': 'с', '$': 'с', '7': 'т', '`/': 'ч',
        '|_|': 'ш', '|_||': 'щ', '|o': 'ю', '9|': 'я'
    }

    def is_special_token(token: str) -> bool:
        patterns = [
            r'^\d+\.?\d*[а-яА-Я]+$',
            r'^\d+%$',
            r'^v\d+\.\d+$',
            r'^\d+/\d+$',
            r'^[a-zA-Z]+$'
        ]
        return any(bool(re.match(pattern, token)) for pattern in patterns)

    def process_token(token: str) -> str:
        if not token.strip() or is_special_token(token):
            return token

        # Базовая замена leet-символов using the leet_map
        result = token
        for leet, normal in leet_map.items():
            result = result.replace(leet, normal)

        return result

    # Разбиваем текст на токены и обрабатываем каждый
    tokens = re.findall(r'\S+|\s+', text)
    processed_text = ""

    for token in tokens:
        processed_text += process_token(token)

    return processed_text

async def normalize_name(text, max_length):
    """Нормализует наименование товара.""" 
    log_msg = log_event("NORMALIZATION_START", f"Начало нормализации: {text}")
    
    original_case = text  

    # 1. Очистка текста от непечатаемых символов и лишних пробелов
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Проверка на латинские наименования
    if re.fullmatch(r'[a-zA-Z0-9-]+', text):
        logger.info(json.dumps(log_msg.end()))
        return text  # Return as is if it's a Latin name
    
    # 3. Разделение на части: существительное с зависимыми, бренд, конструкции/единицы
    words = word_tokenize(text, language='russian')
    noun_phrase = []
    brand_name = []
    constructions_units = []
    other_words = []

    if words:
        current_brand_phrase = []  # Initialize current_brand_phrase here

        for i, word in enumerate(words):
            # Skip if already categorized
            if word in noun_phrase + brand_name + constructions_units + other_words:
                continue

            # Check for Latin brand names
            if re.fullmatch(r'[a-zA-Z]+', word) and len(word) > 2 and word.lower() not in nltk.corpus.stopwords.words('english'):
                current_brand_phrase.append(word)
            else:
                    # Words before the noun phrase
                    other_words.append(word)

        # Add any remaining brand phrase at the end
        if current_brand_phrase:
            brand_name.extend(current_brand_phrase)

    # 4. Замена leet-символов и обработка существительного с зависимыми (если нужно)
    noun_phrase_text = " ".join(noun_phrase)
    if noun_phrase_text:
        # Apply leet replacement only to the noun phrase
        noun_phrase_text = await context_aware_leet_replacement(noun_phrase_text)
        noun_phrase = noun_phrase_text.split()  # Update noun_phrase

        # Process with LLM if not a special case (excluding constructions/units)
        if not (re.fullmatch(r'[a-zA-Z0-9-]+', noun_phrase_text) or
                re.fullmatch(r'\d[wW]-[a-zA-Z0-9]+', noun_phrase_text) or
                re.fullmatch(r'\d+(\.\d+)?\s*(шт|кг|л|мл|м|см|мм|г)', noun_phrase_text, re.IGNORECASE)) and \
                noun_phrase_text not in constructions_units:

            inputs = tokenizer(noun_phrase_text, max_length=None, padding="longest", truncation=False, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # Modify LLM generation to prevent ending with a period
            outputs = model.generate(**inputs, max_length=int(inputs["input_ids"].size(1) * 1.5),
                                     eos_token_id=tokenizer.eos_token_id,  # Use EOS token to signal end of sequence
                                     num_beams=4, early_stopping=True,
                                     no_repeat_ngram_size=2)  # Prevent repetition of 2-grams

            processed_noun_phrase = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            processed_noun_phrase = processed_noun_phrase.rstrip('.')  # Remove trailing period if still present
            noun_phrase = processed_noun_phrase.split()

    # 5. Объединение результатов в нужном порядке и форматирование
    normalized_text = " ".join(noun_phrase + brand_name + constructions_units + other_words)
    normalized_text = normalized_text.strip()

    # 6. Ограничение длины и капитализация
    normalized_text = normalized_text[:max_length]  # Limit length

    # Preserve original capitalization
    final_text = ""
    original_index = 0
    for i, char in enumerate(normalized_text):
        if original_index < len(original_case) and char.lower() == original_case[original_index].lower():
            final_text += original_case[original_index]
            original_index += 1
        else:
            final_text += char

    # Remove trailing period (dot)
    final_text = final_text.rstrip('.')
    
    logger.info(json.dumps(log_msg.end()))
    return final_text

async def check_name(name):
    """Проверяет наименование по правилам и возвращает результат."""
    process_log = log_event("PROCESS_START", "Начало обработки данных")
    
    result = {
        'original_name': name,
        'errors': [],
        'corrected_name': name,
        'status': 'нормализована',
        'reason': ''
    }

    record_log = log_event("RECORD_START", "Начало обработки записи")
    corrected_name = await normalize_name(name, 256)
    logger.info(json.dumps(record_log.end()))

    # Проверка длины наименования
    rule_log = log_event("RULE_START", "Проверка длины наименования")
    if len(corrected_name) > 256:
        result['errors'].append({'error': 'Длина наименования превышает допустимую норму', 'corrected': corrected_name[:256]})
        corrected_name = corrected_name[:256]
    logger.info(json.dumps(rule_log.end()))

    # Проверка на наличие запрещенных символов
    rule_log = log_event("RULE_START", "Проверка запрещенных символов")
    if re.search(r'[^\w\sа-яА-Я]', corrected_name):
        result['errors'].append({'error': 'Наличие запрещенных символов', 'corrected': re.sub(r'[^\w\sа-яА-Я]', '', corrected_name)})
    logger.info(json.dumps(rule_log.end()))

    # Проверка на наличие лишних пробелов
    rule_log = log_event("RULE_START", "Проверка лишних пробелов")
    if re.search(r'\s{2,}', corrected_name):
        corrected_name = re.sub(r'\s{2,}', ' ', corrected_name).strip()
        result['errors'].append({'error': 'Наличие лишних пробелов', 'corrected': corrected_name})
    logger.info(json.dumps(rule_log.end()))

    # Проверка на использование буквы "ё"
    rule_log = log_event("RULE_START", "Проверка буквы 'ё'")
    if 'ё' in corrected_name:
        corrected_name = corrected_name.replace('ё', 'е')
        result['errors'].append({'error': 'Использование буквы "ё"', 'corrected': corrected_name})
    logger.info(json.dumps(rule_log.end()))

    # Проверка на использование кавычек
    rule_log = log_event("RULE_START", "Проверка кавычек")
    if re.search(r'[""«»]', corrected_name):
        corrected_name = re.sub(r'[""«»]', '', corrected_name)
        result['errors'].append({'error': 'Использование кавычек', 'corrected': corrected_name})
    logger.info(json.dumps(rule_log.end()))

    result['corrected_name'] = corrected_name
    
    logger.info(json.dumps(process_log.end()))
    return result

@app.post("/check_name", response_model=CheckResponse)
async def check_name_endpoint(request: CheckRequest):
    result = await check_name(request.name)
    has_error = bool(result['errors'])

    if has_error:
        logger.info(f"Обработка записи с ошибками: {request.name}")
    else:
        logger.info(f"Обработка записи без ошибок: {request.name}")

    return CheckResponse(
        original_name=result['original_name'],
        corrected_name=result['corrected_name'],
        has_error=has_error,
        status=result['status']
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
