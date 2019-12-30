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

LABELS = [
'maggie_simpson',
'charles_montgomery_burns',
'patty_bouvier',
'ralph_wiggum',
'chief_wiggum',
'milhouse_van_houten',
'martin_prince',
'lenny_leonard',
'sideshow_bob',
'selma_bouvier',
'barney_gumble',
'moe_szyslak',
'carl_carlson',
'edna_krabappel',
'snake_jailbird',
'groundskeeper_willie',
'ned_flanders',
'abraham_grampa_simpson',
'krusty_the_clown',
'waylon_smithers',
'apu_nahasapeemapetilon',
'marge_simpson',
'comic_book_guy',
'nelson_muntz',
'mayor_quimby',
'kent_brockman',
'professor_john_frink',
'principal_skinner',
'bart_simpson',
'lisa_simpson',
'homer_simpson',
]

# APP

app = Flask(__name__)

@app.route('/lesimpson/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    image = request.files["image"]
    image = Image.open(BytesIO(image.read()))
    image = np.array(image)
    image = cv2.resize(image, (50, 50))

    payload = {
        "instances": [{'in': image.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/lesimpson:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))['predictions'][0]

    i = 0
    for p in pred:
        if p != 0:
            prediction = LABELS[i]
            break
        i += 1

    # Returning JSON response to the frontend
    return jsonify({"index": i, "pred": prediction})
