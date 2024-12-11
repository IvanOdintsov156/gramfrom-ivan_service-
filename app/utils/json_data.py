import json
import os
import torch

DATA_FILE = "stored_items.json"
DUBLICATE_FILE = "dublicate_items.json"


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            f.close()
            return data
    return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.close()


def save_dublicate(data):
    with open(DUBLICATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()


def load_model_data():
    try:
        similar_embeddings = torch.load("similar_embeddings.pt")
        return similar_embeddings
    except:
        return None
