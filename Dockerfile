FROM python:3.9-slim

WORKDIR /app

COPY models/ ./models/
COPY app/ ./app/
#COPY app/ .
#COPY requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt gunicorn
RUN pip install --no-cache-dir -r ./app/requirements.txt gunicorn
EXPOSE 5000

CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]

## Explain:
#FROM python:3.9-slim – lightweight base image.

#COPY models/ models/ – copy saved model into container.

#COPY app/ app/ – copy Flask app code.

#pip install – installs libs inside container.

#CMD ["gunicorn", ...] – gunicorn runs Flask app in a production WSGI server, with multiple workers for concurrency.