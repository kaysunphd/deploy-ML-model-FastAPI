"""
Units test with conftest data
"""

import os
import pytest
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data


@pytest.fixture(scope='session')
def data():
    """
    Get source data

    Return
    ------
    df: pd.DataFrame
        Loaded clean data
    """

    df = pd.read_csv("../../data/cleaned_census.csv")

    return df


@pytest.fixture(scope='session')
def trained_model():
    """
    Get trained model

    Return
    ------
    model: sklearn model
        Loaded model
    """

    output_path = "../../model"
    filename = os.path.join(output_path, "trained_model.pkl")

    with open(filename, 'rb') as fp:
        model = pickle.load(fp)

    return model


@pytest.fixture(scope='session')
def trained_encoder():
    """
    Get encoder

    Return
    ------
    model: sklearn encoder
        Loaded encoder
    """

    output_path = "../../model"
    filename = os.path.join(output_path, "trained_encoder.pkl")

    with open(filename, 'rb') as fp:
        encoder = pickle.load(fp)

    return encoder


@pytest.fixture(scope='session')
def trained_lb():
    """
    Get labelbinzarier

    Return
    ------
    model: sklearn label binarizer
        Loaded binarizer
    """

    output_path = "../../model"
    filename = os.path.join(output_path, "trained_lb.pkl")

    with open(filename, 'rb') as fp:
        lb = pickle.load(fp)

    return lb


@pytest.fixture(scope='session')
def cat_features():
    """
    Get categorical features of dataset

    Return
    ------
    cat_features: list
                List of categorical feature names
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features


@pytest.fixture(scope='session')
def test_data(data, cat_features, trained_encoder, trained_lb):
    """
    Get trained model

    Input
    ------
    data: pd.DataFrame
        Source dataset
    cat_features: list of strings
        List of categorical features names

    Return
    ------
    model: sklearn model
        Loaded model
    """

    # train_test_split
    train, test = train_test_split(data, test_size=0.20, random_state=123, stratify=data['salary'])

    # process data to create test dataset
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=trained_lb
    )

    return X_test, y_test
