# Olha Svezhentseva & Sandra Sánchez Páez
# Argument Mining Projekt
# WiSe 2022/2023


import os.path

import nltk as nltk
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

SAMPLE_ARTICLES_DIR = 'sample_articles/'

# WHOLE DATASET
ARTICLES_DIR = 'IBM_Debater_(R)_CE-EMNLP-2015.v3/articles/'


def transform_files_to_dataframes(articles_file: str, claims_file: str, evidences_file: str) -> tuple[
        DataFrame | TextFileReader, DataFrame | TextFileReader, DataFrame | TextFileReader]:
    """Read all .txt corpus files and transform them into dataframes."""
    # Get text of all articles
    articles_dataframe = pd.read_csv(articles_file, sep="	")
    # Get all claims
    claims_dataframe = pd.read_csv(claims_file, sep="	")
    # Get all evidences
    evidences_dataframe = pd.read_csv(evidences_file, sep="	")
    # Add names of columns to evidence dataframe
    evidences_dataframe.columns = ['Topic', 'Claim', 'Evidence', 'Evidence Type']
    return articles_dataframe, claims_dataframe, evidences_dataframe


def get_labelled_sentences_from_data(articles_file: str,
                                     claims_file: str,
                                     evidences_file: str) -> tuple[list[str], list[int]]:
    """
    Read all .txt article files from corpus and check for every
    sentence whether it contains a claim in it. If it does, label it with 1,
    else with 0.
    Return list of sentences and list of corresponding labels.
    """
    articles_dataframe, claims_dataframe, evidences_dataframe = transform_files_to_dataframes(
        articles_file, claims_file, evidences_file
    )

    directory = os.fsencode('IBM_Debater_(R)_CE-EMNLP-2015.v3/articles/')
    article_no_claims = []
    number_of_claims = 0
    X = []
    Y = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(os.path.join(directory, file), 'r') as txt_file:
            txt = txt_file.read().replace('\n', '')
            sentences = nltk.tokenize.sent_tokenize(txt)
            # Get topic from article id
            art_id = int(filename[6:-4])
            topic = articles_dataframe.loc[articles_dataframe['article Id'] == art_id, 'Topic']
            if len(topic) > 0:
                # Get all claims for this topic
                claims_to_topic = claims_dataframe.loc[
                    claims_dataframe['Topic'] == topic.item()]  # select rows from a df based on values in column
                list_claims_ori = claims_to_topic['Claim original text'].tolist()
                list_claims_cor = claims_to_topic['Claim corrected version'].tolist()
                for sentence in sentences:
                    X.append(sentence)
                    if any(s in sentence for s in list_claims_ori) or any(s in sentence for s in list_claims_cor):
                        # Label sentence as 'with claim'
                        Y.append(1)
                        number_of_claims += 1
                    else:
                        # Label sentence as 'without claim'
                        Y.append(0)
            else:
                article_no_claims.append(art_id)
                continue
    print(f"The number of raw sentences is {len(X)}.")
    print(f"The number of labels is {len(Y)}.")
    return X, Y
