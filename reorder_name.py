import re
import spacy
import pymorphy3

# Инициализация
morph = pymorphy3.MorphAnalyzer()

try:
    nlp = spacy.load("ru_core_news_sm")
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели spaCy: {e}")

def reorder_name(text, original_case):
    """Reorders the words in the text, preserving original case."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Входной текст должен быть непустой строкой.")

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
    text = text[0].upper() + text[1:]

    return text
