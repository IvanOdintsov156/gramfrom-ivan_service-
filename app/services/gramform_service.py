import re
import logging
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
import time
from app.models.gramform_model import CheckResponse, CheckRequest
from app.utils.gram_utils import correct_morphology, fix_leet
from app.utils.logging import log_info, measure_time

# Инициализация морфологического анализатора
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Исключения для слов, которые не требуют проверки на существительное
exclusions = {"направляющая", "санки", "салазки"}

async def check_name(request: CheckRequest):
    start_time = time.time()
    result, rule_times = check_name_rules(request.name)
    total_time = measure_time(start_time)

    # Логирование метрик
    statistics = {
        'Среднее время обработки одной записи': total_time,
        'Среднее время обработки правила fix_leet': rule_times['fix_leet'],
        'Среднее время обработки правила correct_morphology': rule_times['correct_morphology'],
        'Среднее время обработки правила check_start_with_noun': rule_times['check_start_with_noun'],
        'Среднее время обработки правила check_grammar': rule_times['check_grammar'],
    }

    log_info(f"Обработка записи без ошибок: {request.name} - {total_time} сек")

    keys = result['errors'].keys()

    return CheckResponse(
        corrected_name=result['corrected_name'],
        gram_morph_rule=not bool("gram-morph-error" in keys),
        first_noun_rule = not bool("first-noun-error" in keys),
        description=result['errors'],
    )


def check_name_rules(name):
    #Проверяет наименование по правилам и возвращает результат.
    result = {
        'original_name': name,
        'errors': {},
        'corrected_name': name
    }

    rule_times = {
        'fix_leet': "0",
        'correct_morphology': "0",
        'check_start_with_noun': "0",
        'check_grammar': "0"
    }

    # Исправление leet-замен
    start_time = time.time()
    cleaned_name = fix_leet(name)
    rule_times['fix_leet'] = measure_time(start_time)

    # Проверка грамматических ошибок
    start_time = time.time()
    if not re.match(r'^[А-Яа-я\s]+$', cleaned_name):
        result['errors']['gram-morph-error'] = 'Грамматическая ошибка'
    else:
        corrected_name = correct_morphology(cleaned_name)
        if cleaned_name != corrected_name:
            result['errors']['gram-morph-error'] = 'Морфологическая ошибка'
            cleaned_name = corrected_name
    rule_times['correct_morphology'] = measure_time(start_time)

    # Проверка на начало с существительного в единственном числе
    start_time = time.time()
    doc = Doc(cleaned_name)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tokens = [token for token in doc.tokens if token.text.lower() not in exclusions]

    if doc.tokens:
        first_token = doc.tokens[0]
        if not (first_token.pos == 'NOUN' and 'Sing' in first_token.feats):
            result['errors']['first-noun-error'] = 'Наименование должно начинаться с существительного в единственном числе'
    rule_times['check_start_with_noun'] = measure_time(start_time)

    result['corrected_name'] = cleaned_name
    return result, rule_times