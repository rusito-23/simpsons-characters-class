#!/bin/sh

docker run -d --name serving_base tensorflow/serving
docker cp model/lesimpson serving_base:/models/lesimpson
docker commit --change "ENV MODEL_NAME lesimpson" serving_base lesimpson_serve
docker kill serving_base
docker rm serving_base
docker run -p 8501:8501 --name lesimpson_serve lesimpson_serve
