#!/usr/bin/env python
"""
Perform unit tests on pipeline
"""
import argparse
import logging
import wandb
import pandas as pd
from ml.model import inference, compute_model_metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_column_names(data: pd.DataFrame):
    """
    Check if categorical features in process_data is in cleaned data
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
    column_names = data.columns
    check =  all(name in column_names for name in cat_features)
    assert check == True


def test_inference(trained_model, test_data):
    """
    Check inference of trained model
    """

    X_test, y_test = test_data

    try:
        predictions = trained_model.predict(X_test)
    except RuntimeError as err:
        logger.error("Inference failed, {err}")
        raise err


def test_compute_metrics(trained_model, test_data):
    """
    Check compute of metrics
    """

    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)

    try:
        precise, recall, fbeta = compute_model_metrics(y_test, predictions)
    except Exception as err:
        logger.error("Performance metrics calculations failed, {err}")
        raise err
