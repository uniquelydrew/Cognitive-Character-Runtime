FROM python@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services /app/services
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
CMD ["uvicorn", "services.memory.app:app", "--host", "0.0.0.0", "--port", "8000"]
