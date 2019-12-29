#!/usr/bin/env python
# coding: utf-8

import base64
import json
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import requests

# APP

app = Flask(__name__)

@app.route('/lesimpson/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    image = request.files["image"]
    image = Image.open(BytesIO(image.read()))
    image = np.array(image)
    image = cv2.resize(image, (128, 128))

    payload = {
        "instances": [{'input': [image.tolist()]}]
    }

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/lesimpson:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])
