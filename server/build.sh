#!/bin/sh

## BUILD ALL NEEDED DOCKER CONTAINERS ##

docker build -t lesimpson_serve:latest -f docker_files/serve/Dockerfile .
docker build -t lesimpson_api:latest -f docker_files/app/Dockerfile .
