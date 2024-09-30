# Olha Svezhentseva & Sandra Sánchez Páez
# Argument Mining Projekt
# WiSe 2022/2023

import en_core_web_sm
import spacy
from nltk import WordNetLemmatizer

en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words

nlp = en_core_web_sm.load()
lemmatizer = WordNetLemmatizer()


# nltk.download('stopwords')
# nltk.download('punkt')


def read_file(file_name: str) -> str:
    """Read raw text, return a str."""
    with open(file_name, encoding='utf-8') as file:
        return file.read()


def read_lines(file_name: str) -> list[str]:
    """Read lines from file, split by new line to keep paragraph unity (potential context)."""

    with open(file_name, encoding='utf-8') as file:
        text = file.read()
    text = text.strip().split('\n')
    return text


def lemmatize_sentence(sentence: str) -> list:
    """Return all lemmas in sentence."""
    doc = en(sentence)
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text."""

    clean_paragraph = ' '.join([word for word in text.split() if
                                word.lower() not in stopwords])
    return clean_paragraph


def parse_sentence(sentence: any) -> list[tuple[any, str, str]]:
    """Return tagged lemmas from sentence."""
    doc = en(sentence)
    tagged_lemmas = [(token, token.lemma_, token.pos_) for token in doc]
    return tagged_lemmas
