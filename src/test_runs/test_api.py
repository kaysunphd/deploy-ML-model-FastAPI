#!/usr/bin/env python
"""
Perform unit tests on API
Author: Kay Sun
Date: September 20 2023
"""

import os
import json
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_get():
    """
    Check GET
    """

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Greetings! Welcome to our model deployment"


def test_inference():
    """
    Check inference
    """
    test_sample = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "hours_per_week": 40,
            "native_country": "United-States"
            }

    # sample_data = json.dumps(test_sample)
    response = client.post("/inference", json=test_sample)

    assert response.status_code == 200
    assert response.json()['prediction'][0] == 0


def test_inference_labels():
    """
    Check inference labels
    """
    test_sample = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "hours_per_week": 40,
            "native_country": "United-States"
            }

    response = client.post("/inference_labels", json=test_sample)

    assert response.status_code == 200
    assert response.json()['prediction'] == "<=50K"
