# Dockerfile for Hoffman2 Chatbot
# This Dockerfile creates a container image for the Flask-based chatbot application,
# including the step to build the static frontend.

# --- Stage 1: Base Image ---
FROM python:3.10-slim

# --- Environment Variables ---
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
# OPENAI_API_KEY will be loaded from the .env file for testing,
# but for production, it should be passed at runtime via Kubernetes Secrets or -e flag.

# --- Install System Dependencies (if any) ---
# Git is needed if GitPython in createdb.py were run here,
# but also good if any other tool might need it.
# For the Markdown library or other Python packages, build-essential might be needed
# if they have C extensions, though often not for pure Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# requirements.txt should now include 'Markdown' for build_frontend.py

# --- Copy Frontend Source Files ---
# Copy the entire 'frontend' directory which contains build_frontend.py,
# template.html, and the markdown/json content files.
COPY frontend ./frontend

# --- Build the Static Frontend ---
# Run the build script. This will generate static/index.html within the /app/static directory.
# The build_frontend.py script is designed to output to ../static/index.html
# relative to its location. Since it's in /app/frontend, it will create /app/static/index.html.
RUN python frontend/build_frontend.py

# --- Copy Backend Application Code ---
COPY app.py .
COPY hoffman2_chatbot_web.py .

# --- Copy .env file for TESTING PURPOSES ---
# IMPORTANT: This copies your .env file (containing secrets like OPENAI_API_KEY)
# directly into the Docker image. This is acceptable for local testing with a
# test/dev key but is NOT recommended for production environments.
# In production, OPENAI_API_KEY should be injected as an environment variable
# at runtime (e.g., via Kubernetes Secrets, Docker -e flag, etc.)
# and the .env file should NOT be part of the image.
COPY .env .

# --- Copy Pre-built LlamaIndex ---
# Ensure 'hoffman2_index' exists in the build context (same directory as Dockerfile).
COPY hoffman2_index ./hoffman2_index

# The 'static' directory (containing the generated index.html) is now created by
# the RUN python frontend/build_frontend.py command.
# Flask is configured with static_folder='static', so it will find it.

# --- Expose Port ---
EXPOSE 5000

# --- Run Command ---
# Use Gunicorn for production.
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

# For development/simplicity, using Flask's built-in server:
CMD ["python", "app.py"]

# --- Final Notes ---
# To build this image:
# 1. Ensure 'hoffman2_index' directory is present at the project root.
# 2. Ensure the 'frontend' directory with all its contents is present.
# 3. Ensure '.env' file is present at the project root (for testing).
# 4. Run: docker build -t hoffman2-chatbot:latest .
#
# To run this image (the .env file inside the image will be used):
# docker run -p 5000:5000 hoffman2-chatbot:latest
#
# For PRODUCTION, remove 'COPY .env .' and pass OPENAI_API_KEY as a runtime environment variable:
# docker run -p 5000:5000 -e OPENAI_API_KEY="your_production_api_key" hoffman2-chatbot:latest
