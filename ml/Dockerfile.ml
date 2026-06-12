FROM python:3.11-slim AS ml
WORKDIR /ml
COPY requirements.ml.txt .
RUN pip install --no-cache-dir -r requirements.ml.txt
COPY ml/ .
CMD ["uvicorn", "ml_server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]