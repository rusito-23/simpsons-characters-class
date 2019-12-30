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
import onnxruntime

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

sess = onnxruntime.InferenceSession('../../model/lesimpson.onnx')

@app.route('/lesimpson/predict/', methods=['POST'])
def image_classifier():

    # Decoding and pre-processing base64 image
    image = request.files["image"]
    image = Image.open(BytesIO(image.read()))
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (50, 50))
    image = image / 255

    # Pass the image through the model
    _in = sess.get_inputs()[0]
    _out = sess.get_outputs()[0]

    out = sess.run([_out.name],
            {_in.name: np.array([image.astype(np.float32)])})[0]

    pred_index = np.argmax(out)
    prediction = LABELS[pred_index]

    # Returning JSON response to the frontend
    return jsonify({
        "prediction": prediction
    })
