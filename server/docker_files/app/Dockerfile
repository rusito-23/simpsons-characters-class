# Dockerfile to embed LeSimpson model serving within a Rest api

FROM python:3.7.5
COPY source/app /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
