"""
RESTful API using FastAPI
Author: Kay Sun
Date: September 20 2023
"""

import os
from fastapi import FastAPI
from src.ml.model import make_inference
from src.ml.data import InputData

app = FastAPI()

@app.get("/")
async def welcome():
    return "Greetings! Welcome to our model deployment"


@app.post("/")
async def inference(input_data: InputData):
    input_data = input_data.dict()
    
    make_inference