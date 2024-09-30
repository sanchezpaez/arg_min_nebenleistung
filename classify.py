# Olha Svezhentseva & Sandra Sánchez Páez
# Argument Mining Projekt
# WiSe 2022/2023

import pickle
import random
from typing import Any

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from nltk import word_tokenize
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm

from extract_features import check_opinion_verbs, get_subjectivity_score, has_entities, get_length
from preprocess import lemmatize_sentence, remove_stopwords
from reformat_corpus import get_labelled_sentences_from_data

SEED = 10


def save_data(data: any, filename: any) -> None:
    """Save data into file_name (.pkl file)to save time."""
    with open(filename, mode="wb") as file:
        pickle.dump(data, file)
        print(f'Data saved in {filename}')


def load_data(filename: any) -> any:
    """Load pre-saved data."""
    with open(filename, "rb") as file:
        output = pickle.load(file)
    print(f'Loading  data  pre-saved as {filename}...')
    return output


def create_termdoc_matrix(train_sents, train_labels: list[int], dev_sents: list[str], dev_labels: list[int],
                          oversample=False):
    """
    Turn X_train and y_train into a tf-idf weighted term-document matrix
    scipy.sparse.csr.csr_matrix for both train and test set.
    Add extra features (sentence length, opinion verbs and subjectivity score)
    Returns X_train, X_dev, y_train, y_dev matrices.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if oversample:
        train_sents = train_sents[:, 0]
        y_train = train_labels
    else:
        y_train = np.array(train_labels)
    X_train = vectorizer.fit_transform(train_sents)
    y_dev = np.array(dev_labels)
    print(X_train.shape)
    X_dev = vectorizer.transform(dev_sents)
    print(X_dev.shape)

    # Add other features
    # Sentence length
    lengths_train = [get_length(word_tokenize(sentence)) for sentence in train_sents]
    lengths_dev = [get_length(word_tokenize(sentence)) for sentence in dev_sents]
    # Update matrices
    X_train = hstack((X_train, np.reshape(lengths_train, (len(lengths_train), 1))))
    X_dev = hstack((X_dev, np.reshape(lengths_dev, (len(lengths_dev), 1))))

    # check_opinion_verbs
    opinion_train = [check_opinion_verbs(word_tokenize(sentence)) for sentence in train_sents]
    opinion_dev = [check_opinion_verbs(word_tokenize(sentence)) for sentence in dev_sents]
    # Update matrices
    X_train = hstack((X_train, np.reshape(opinion_train, (len(opinion_train), 1))))
    X_dev = hstack((X_dev, np.reshape(opinion_dev, (len(opinion_dev), 1))))

    # get_subjectivity_score
    subj_train = [get_subjectivity_score(word_tokenize(sentence)) for sentence in train_sents]
    subj_dev = [get_subjectivity_score(word_tokenize(sentence)) for sentence in dev_sents]
    # Update matrices
    X_train = hstack((X_train, np.reshape(subj_train, (len(subj_train), 1))))
    X_dev = hstack((X_dev, np.reshape(subj_dev, (len(subj_dev), 1))))

    # entities
    ent_train = [has_entities(sentence) for sentence in train_sents]
    ent_dev = [has_entities(sentence) for sentence in dev_sents]
    # Update matrices
    X_train = hstack((X_train, np.reshape(ent_train, (len(ent_train), 1))))
    X_dev = hstack((X_dev, np.reshape(ent_dev, (len(ent_dev), 1))))

    print('Matrix with all features fitted')

    return X_train, X_dev, y_train, y_dev


def create_dataset_simple(sentences: list[str], labels: list[int]):
    """
    Take in list of sentences and corresponding list of labels,
    get indexes, and return shuffled data.
    """
    indexes = [i for i in range(len(sentences))]
    X, y, indexes = shuffle(sentences, labels, indexes, random_state=SEED)
    return X, y, indexes


def preprocess_sentences(sentences: list[str], filename: str, is_loaded: bool):
    """
    Preprocess list of sentences, by removing stopwrods and lemmatizing.
    If the sentences have already been preprocessed, load data.
    """
    if is_loaded:
        X_preprocessed = load_data(filename)
    else:
        X_preprocessed = []

        for sentence in tqdm(sentences):
            clean_sentence = remove_stopwords(sentence)
            lemmatized_sentence = lemmatize_sentence(clean_sentence)
            X_preprocessed.append(' '.join(lemmatized_sentence))

        print('Saving data...')
        save_data(X_preprocessed, filename)
        print(f"The number of preprocessed sentences is {len(X_preprocessed)}.")
        print(f"The preprocessed sentences have been saved and will be available as {filename}")

    return X_preprocessed


def train_model(X_train, y_train, X_val, y_val, model) -> tuple[Any, Any]:
    """
    Pass in train & validation sets and the desired classifier.
    Call classifier with specific parameters and weights, if so specified,
    and fit model on the train set. Then predict accuracy on the validation set.
    """
    classifier = None
    if model == 'MLP':
        classifier = MLPClassifier(verbose=True, random_state=SEED, max_iter=15)
    elif model == 'SVC':
        classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True, random_state=SEED)
    elif model == 'SVC_weights':
        weights = {0: 1.03, 1: 34.4}
        # Weight calculated by getting the probability, for example probability of class 0 is  41117 /(1230+41117),
        # the inverse probability is 1/ 41117 /(1230+41117), which is 1.03.
        classifier = SVC(C=10, gamma=0.75, kernel='rbf', probability=True, class_weight=weights, random_state=SEED)
    elif model == 'dummy':
        classifier = DummyClassifier(strategy='most_frequent')
    classifier.fit(X_train, y_train)
    binary_balanc_predictions = classifier.predict(X_val)
    binary_balanc_probs = classifier.predict_proba(X_val)
    print(f'Accuracy on the test set with {model} classifier: {classifier.score(X_val, y_val)}')
    return binary_balanc_predictions, binary_balanc_probs


def downsample(sentences, labels) -> tuple[list[list[Any] | Any], list[int | Any]]:
    """
    Taken an imbalanced set, get all instances of the minority class
    and only a few of the majority class,
    creating thereby a new, balanced set.
    """
    downsampled_sentences = []
    downsampled_labels = []

    for i in range(0, len(sentences)):
        if labels[i] == 1:
            downsampled_sentences.append(sentences[i])
            downsampled_labels.append(labels[i])

    not_claims = list(set(sentences) - set(downsampled_sentences))

    # we do not need indices of corresponding train_labels because we know that they are 0
    inputNumbers = range(0, len(not_claims))

    random_not_claims = random.sample(inputNumbers, 3000)
    for i in random_not_claims:
        downsampled_sentences.append(not_claims[i])
        downsampled_labels.append(0)
    print(f"The number of downsampled sentences is {len(downsampled_sentences)}.")
    print(f"The number of downsampled labels is {len(downsampled_labels)}.")
    return downsampled_sentences, downsampled_labels


def over_sample(X_train: list[list[str]], y_train: list[int]) -> tuple[Any, Any]:
    """
    Taken an imbalanced train set, transform the X and y data into arrays,
    and call RandomOverSampler with 'minority' strategy on them.
    """
    X = np.array(X_train)
    X = np.reshape(X, (len(X), 1))
    y = np.array(y_train)
    y = np.reshape(y, (len(y), 1))
    ros = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # Class distribution
    print(X_resampled.shape)
    print(y_resampled.shape)
    return X_resampled, y_resampled


def create_train_dev_set_splits(sentences: list[list[str]], labels: list[int], indexes: list[int]) -> tuple[
        Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Take in simple dataset, make train/dev/test splits and return all
    corresponding X and y values matrices.
    """
    X_sent_train, X_sent_test, y_train, y_test, orig_train, orig_test = train_test_split(
        sentences,
        labels,
        indexes,
        test_size=.2,
        random_state=SEED
    )
    X_train_set, X_dev_set, y_train_set, y_dev_set, orig_train, orig_dev = train_test_split(
        X_sent_train,
        y_train,
        orig_train,
        test_size=.3,
        random_state=SEED
    )
    return X_train_set, X_sent_test, X_dev_set, y_train_set, y_test, y_dev_set, orig_train, orig_dev, orig_test


if __name__ == '__main__':
    sentences, labels = get_labelled_sentences_from_data(
        'IBM_Debater_(R)_CE-EMNLP-2015.v3/articles.txt', 'IBM_Debater_(R)_CE-EMNLP-2015.v3/claims.txt',
        'IBM_Debater_(R)_CE-EMNLP-2015.v3/evidence.txt'
    )
    needs_downsample = False
    needs_oversample = False
    if needs_downsample:
        down_sentences, down_labels = downsample(sentences,
                                                 labels)
        preprocessed_sentences = preprocess_sentences(down_sentences, 'downsampled_preprocessed_sentences.pkl',
                                                      is_loaded=True)
        labels = down_labels
    else:
        preprocessed_sentences = preprocess_sentences(sentences, 'preprocessed_sentences.pkl',
                                                      is_loaded=True)
    sentences, labels, indexes = create_dataset_simple(preprocessed_sentences, labels)

    # Split in train and test (test set for evaluation in the end)
    X_train_set, X_sent_test, X_dev_set, y_train_set, y_test, y_dev_set, \
        orig_train, orig_dev, orig_test = create_train_dev_set_splits(sentences, labels, indexes)
    if needs_oversample:
        over_sents, over_labels = over_sample(X_train_set, y_train_set)
        save_data(over_labels, 'oversampled_labels')
        save_data(over_sents, 'oversampled_sentences.pkl')
        save_data(X_dev_set, 'X_dev_set.pkl')
        save_data(y_dev_set, 'y_dev_set.pkl')
        save_data(X_sent_test, 'X_sent_test.pkl')
        save_data(y_test, 'y_test.pkl')

        # With pre-saved data

        # over_sents = load_data('oversampled_sentences.pkl')
        # over_labels = load_data('oversampled_labels')
        # X_dev_set = load_data('X_dev_set.pkl')
        # y_dev_set = load_data('y_dev_set.pkl')
        # X_sent_test = load_data('X_sent_test.pkl')
        # y_test = load_data('y_test.pkl')

        X_train, X_dev, y_train, y_dev = create_termdoc_matrix(over_sents, over_labels, X_sent_test, y_test,
                                                               oversample=True)
    else:
        X_train, X_test, y_train, y_test = create_termdoc_matrix(X_train_set, y_train_set, X_sent_test, y_test,
                                                                 oversample=False)

    #  Save and load matrices to spare time

    # X_train, X_dev, y_train, y_dev = create_termdoc_matrix(over_sents, over_labels, X_dev_set, y_dev_set, oversample=True)
    # X_train, X_test, y_train, y_test = create_termdoc_matrix(over_sents, over_labels, X_sent_test, y_test, oversample=True)
    # save_data(X_train, 'TEST_ov_X_train.pkl')
    # save_data(X_test, 'TEST_ov_X_dev.pkl')
    # save_data(y_train, 'TEST_ov_y_train.pkl')
    # save_data(y_test, 'TEST_ov_y_dev.pkl')

    # save_data(X_train, 'ov_X_train.pkl')
    # save_data(X_dev, 'ov_X_dev.pkl')
    # save_data(y_train, 'ov_y_train.pkl')
    # save_data(y_dev, 'ov_y_dev.pkl')

    # WITH LOADED MATRIX

    # X_train = load_data('ov_X_train.pkl')
    # X_dev = load_data('ov_X_dev.pkl')
    # y_train = load_data('ov_y_train.pkl')
    # y_dev = load_data('ov_y_dev.pkl')
    # X_sent_test = load_data('X_sent_test.pkl')
    # y_test = load_data('y_test.pkl')

    # X_train = load_data('TEST_ov_X_train.pkl')
    # y_train = load_data('TEST_ov_y_train.pkl')
    # X_test = load_data('TEST_ov_X_dev.pkl')
    # y_test = load_data('TEST_ov_y_dev.pkl')

    # EVALUATE ORIGINAL TEST SET
    predictions_dummy, probabilities_dummy = train_model(X_train, y_train, X_test, y_test,
                                                         'dummy')
    predictions_mlp, probabilities_mlp = train_model(X_train, y_train, X_test, y_test,
                                                     'MLP')
    predictions_svc, probabilities_svc = train_model(X_train, y_train, X_test, y_test, 'SVC')
    predictions_svc_w, probabilities_svc_w = train_model(X_train, y_train, X_test, y_test,
                                                         'SVC_weights')
    print(classification_report(y_test, predictions_dummy, target_names=["Not claim", "Claim"], zero_division=1))
    print(classification_report(y_test, predictions_mlp, target_names=["Not claim", "Claim"]))
    print(classification_report(y_test, predictions_svc, target_names=["Not claim", "Claim"]))
    print(classification_report(y_test, predictions_svc_w, target_names=["Not claim", "Claim"]))
