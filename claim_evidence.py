# Olha Svezhentseva & Sandra Sánchez Páez
# Argument Mining Projekt
# WiSe 2022/2023


import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from nltk import sent_tokenize

from preprocess import read_lines, read_file


def group_articles(articles_info: str) -> defaultdict[Any, list]:
    """
    Create a dictionary of the form: {topic1: [article number, ....],
    topic2: [article number, ....]} so each topic is associated
    with one or more article.
    """
    topic2article = defaultdict(list)
    with open(articles_info) as file_in:
        next(file_in)
        for line in file_in.readlines():
            article_info = line.split("\t")
            topic2article[article_info[0]].append(article_info[2].strip())
    return topic2article


def read_articles(ARTICLES_DIR: str) -> dict[int, list]:
    """Save content of articles of the form {article_number: list of preprocessed_sentences of the article}."""
    article_content = {}
    for file_name in os.listdir(ARTICLES_DIR):
        txt_path = Path(ARTICLES_DIR, file_name)
        if str(txt_path)[-3:] == "txt":
            number = re.findall('\d+', str(txt_path))[-1]
            with open(txt_path, encoding="'utf-8'") as file_in:
                content = file_in.read()
                sents = sent_tokenize(content)
                article_content[number] = sents

    return article_content


def topic2claim(claim_file: str) -> defaultdict[str, dict[str, list]]:
    """
    Create dict of the form: {"topic":  {claim1: [], claim2: [], ...}}
    so that we see every topic and claims associated with it.
    The empty list will be filled later with corresponding evidences.
    """
    topic2claim = defaultdict(lambda: dict())

    with open(claim_file) as file_in:
        next(file_in)
        for line in file_in.readlines():
            claims_info = line.split("\t")
            topic = claims_info[0]
            claim_original = claims_info[1].strip()
            # Add also thew last column of the original file so we match it against all evidences
            claim_corrected = claims_info[2].strip()
            topic2claim[topic][claim_original, claim_corrected] = []
    return topic2claim


def connect_claim_evidence(evidence_file: str, topic2claim: dict) -> dict[str, dict[str, list]]:
    """
    Fill in claims it topic2claim dict, so that it is of the form
    {"topic":  {claim1: [EVIDENCE1, EVIDENCE2]}
    """
    with open(evidence_file) as file_in:
        for line in file_in.readlines():
            evidence_info = line.split("\t")
            topic = evidence_info[0]
            claim = evidence_info[1].strip()
            evidence = evidence_info[2]
            for t in topic2claim:
                for c in topic2claim[topic]:
                    if claim == c[0] or claim == c[1]:
                        topic2claim[topic][c].append(evidence)

    return topic2claim


def read_all_articles(directory: str, articles: list) -> list:
    """
    Read all articles in key,value pairs like this
    'This house would abolish the monarchy': ['430', '450', '419']
    """
    return [read_lines(directory + 'clean_' + article + '.txt') for article in articles]


def get_evidences_from_claims(claims: dict) -> list[str]:
    """
    Given the dict claim2evidence with empty lists as values for every claim,
    find the corresponding evidences and append them to the lists.
    """
    all_evidences = []
    for claim in claims:
        for evidence in claims[claim]:
            all_evidences.append(evidence)
    return all_evidences


def get_evidences_from_articles(articles_dir: str, topic2article: dict, claim2evidence: dict) -> dict:
    """
    Given topic2article and claim2evidence, return a dict of the form
    {art_number_1: evidence, context_to_evidence, art_number2: evidence, context_to_evidence}
    """
    found_evidences = {}
    for topic in topic2article:
        claims_to_topic = claim2evidence[topic]
        evidences_to_topic = get_evidences_from_claims(claims_to_topic)
        for article_number in topic2article[topic]:
            try:
                article_text = read_file(articles_dir + 'clean_' + article_number + '.txt')
                for evidence in evidences_to_topic:
                    if evidence.lower() in article_text.lower():
                        found_evidences[article_number] = evidence, get_context_from_article(evidence, article_text)
            except FileNotFoundError:
                pass
    return found_evidences


def get_context_from_article(keyword: str, article: str) -> list[str]:
    "Given a keyword or a whole evidence (str), return the paragraph context to it."
    return [p for p in article.split('\n') if keyword in p]


SAMPLE_ARTICLES_DIR = 'sample_articles/'

# WHOLE DATASET
ARTICLES_DIR = 'IBM_Debater_(R)_CE-EMNLP-2015.v3/articles/'

# So now we have 4 variables:
# articles_content -- article's number with its content
articles_content = read_articles(ARTICLES_DIR)
# topic2article = topics with associated articles' numbers
topic2article = group_articles('IBM_Debater_(R)_CE-EMNLP-2015.v3/articles.txt')  # 58 topics

# topic2claim = every topic and claims associated with it
topic2claim = topic2claim('IBM_Debater_(R)_CE-EMNLP-2015.v3/claims.txt')

# claim2evidence = topics with corresponding claims, each claim is supported with 0 or more evidences
claim2evidence = connect_claim_evidence('IBM_Debater_(R)_CE-EMNLP-2015.v3/evidence.txt', topic2claim)

# Get all data (paragraphs) from every article related to a topic so we can split it afterwards into train and test sets
raw_data = {k: read_all_articles(ARTICLES_DIR, v) for k, v in topic2article.items()}
# print(raw_data)

if __name__ == '__main__':
    sample_found_evidences = get_evidences_from_articles(SAMPLE_ARTICLES_DIR, topic2article, topic2claim)
    print(len(sample_found_evidences.items()))
    found_evidences = get_evidences_from_articles(SAMPLE_ARTICLES_DIR, topic2article, claim2evidence)
    print(len(found_evidences.items()))  # 390 (should be many more), 3891 claims
    print(found_evidences)
