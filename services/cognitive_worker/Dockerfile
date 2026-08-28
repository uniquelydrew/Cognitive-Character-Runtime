FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services /app/services
ENV PYTHONPATH=/app
CMD ["uvicorn", "services.cognitive_worker.app:app", "--host", "0.0.0.0", "--port", "8000"]
