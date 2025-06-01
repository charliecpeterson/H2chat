FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

ENV NLTK_DATA=/app/nltk_data

RUN useradd --create-home --shell /bin/bash app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY hoffman2_chatbot_web.py .
COPY templates ./templates
COPY hoffman2_index ./hoffman2_index
RUN mkdir -p $NLTK_DATA && chown -R app:app /app

RUN python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')"

USER app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]

