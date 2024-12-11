import torch
import time
from app.utils.embedding import (
    get_sentence_embedding,
    extract_numbers_and_abbreviations,
    compare_numbers,
)
from app.utils.json_data import load_data, save_data, load_model_data, save_dublicate
from app.utils.logging import log_info, measure_time
from app.models.similarity_models import RecordRequest, AddDataRequest
from fastapi import HTTPException
import re

downloaded_embeddings = load_model_data()
downloaded_records = load_data()


async def add_data(request: AddDataRequest):
    global downloaded_embeddings
    global downloaded_records
    start_time = time.time()  # Начало измерения времени
    stored_items = request.items
    if stored_items:
        downloaded_records = [record for record in stored_items]
        similar_texts = [f"{record['name']}" for record in stored_items]
        downloaded_embeddings = [get_sentence_embedding(text) for text in similar_texts]
        downloaded_embeddings = torch.stack(downloaded_embeddings).squeeze()

        if len(downloaded_embeddings.shape) < 2:
            downloaded_embeddings = downloaded_embeddings.unsqueeze(0)

        duplicates = find_downloaded_duplicates()
        if duplicates:
            save_dublicate(duplicates)
        else:
            log_info("Дубликаты не найдены.")

        torch.save(downloaded_embeddings, "downloaded_embeddings.pt")
        save_data(downloaded_records)

        total_records_processed = len(stored_items) # Количество обработанных записей
        log_info(f"Время, затраченное на add_data: ", measure_time(start_time))
        log_info(f"Среднее время обработки записи: ", measure_time(start_time, total_records_processed))

        return {"message": "Items added successfully", "total_items": total_records_processed}
    raise HTTPException(
        status_code=400,
        detail="Нет доступных записей для добавления.",
    )


async def check_similarity(request: RecordRequest):
    start_time = time.time()  # Начало измерения времени
    if downloaded_embeddings is None or not downloaded_records:
        raise HTTPException(
            status_code=400,
            detail="Нет доступных записей для сравнения. Пожалуйста, сначала добавьте записи.",
        )

    item_id = request.id
    item_name = request.name

    input_text = f"{item_name}"

    similarity_score, most_similar_records = find_similar_records(input_text)
    log_info(f"Время, затраченное на проверку сходства: ", measure_time(start_time))

    return {
        "similarity_score": similarity_score,
        "most_similar_records": most_similar_records,
    }


def find_similar_records(input_text):
    global downloaded_embeddings
    global downloaded_records

    input_embedding = get_sentence_embedding(input_text)
    input_embedding = input_embedding.squeeze()

    cosine_scores = torch.nn.functional.cosine_similarity(
        input_embedding, downloaded_embeddings
    )

    item_abbrs = extract_numbers_and_abbreviations(input_text)

    coefficients = []
    for record in downloaded_records:
        record_abbrs = extract_numbers_and_abbreviations(record["name"])
        coefficient = compare_numbers(item_abbrs, record_abbrs)
        coefficients.append(coefficient)

    # Применяем коэффициенты к cosine_scores
    adjusted_scores = cosine_scores * torch.tensor(coefficients)

    max_score = adjusted_scores.max().item()
    most_similar_records = [
        downloaded_records[i]
        for i in range(len(downloaded_records))
        if adjusted_scores[i] > 0.85
    ]

    return max_score, most_similar_records


def find_downloaded_duplicates(threshold=0.85):
    global downloaded_embeddings
    global downloaded_records
    start_time = time.time()  # Начало измерения времени
    duplicates = []
    num_embeddings = downloaded_embeddings.shape[0]

    extracted_data = [
        extract_numbers_and_abbreviations(record["name"])
        for record in downloaded_records
    ]

    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            cosine_score = torch.nn.functional.cosine_similarity(
                downloaded_embeddings[i].unsqueeze(0),
                downloaded_embeddings[j].unsqueeze(0),
            ).item()

            if cosine_score > threshold:
                item_abbrs = extracted_data[i]
                record_abbrs = extracted_data[j]

                if compare_numbers(item_abbrs, record_abbrs) == 1:
                    duplicates.append(
                        (downloaded_records[i], downloaded_records[j], cosine_score)
                    )

    log_info(
        f"Время, затраченное на прокатную проверку дубликатов: ", measure_time(start_time)
    )
    log_info(
        f"Среднее время обработки дубликатов: ", measure_time(start_time, len(duplicates))
    )
    return duplicates
