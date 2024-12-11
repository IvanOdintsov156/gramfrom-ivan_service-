from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка модели и токенизатора
model_name = "ai-forever/sage-fredt5-distilled-95m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def correct_morphology(text):
    """Исправляет морфологические ошибки в тексте с помощью модели NLP."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def fix_leet(text):
    """Исправляет leet-замену в тексте, сохраняя капитализацию."""
    leet_map = {
        '4': 'A',
        '3': 'E',
        '1': 'I',
        '0': 'O',
        '5': 'S',
        '7': 'T'
    }
    return ''.join(leet_map.get(char.lower(), char) if char.islower() else leet_map.get(char.lower(), char) for char in text)