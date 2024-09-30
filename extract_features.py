# Olha Svezhentseva & Sandra Sánchez Páez
# Argument Mining Projekt
# WiSe 2022/2023

import en_core_web_sm
import pysentiment2 as ps

hiv4 = ps.HIV4()
nlp = en_core_web_sm.load()


def get_length(sentence: list[str]) -> int:
    """Return binary value for sentence length with threshold 4."""
    if len(sentence) > 4:
        return 1
    return 0


def check_opinion_verbs(sentence: list[str]) -> int:
    """Check if a sentence contains an opinion verb and return binary value."""
    opinion_verbs = ["claim", "state", "believe", "think", "suggest"]
    for lemma in sentence:
        if lemma in opinion_verbs:
            return 1
    return 0


def get_subjectivity_score(sentence: list[str]) -> int:
    """Compute and return subjectivity score of sentence."""
    scores = hiv4.get_score(sentence)

    return scores['Subjectivity']


def has_entities(sentence: str) -> int:
    """
    Iterate over sentence items to check whether it contains the type
    of entity that could be related to a claim ('ORG' or 'PERSON').
    Return binary value.
    """
    doc = nlp(str(sentence))
    entities = []
    for token in doc.ents:  # Get all entities present in sentence
        if token.label_ in ['ORG', 'PERSON']:  # If the entity is a number, person or org, save it
            entities.append(token.label_)
        else:
            pass
    if entities:
        return 1
    else:
        return 0
