# Hoffman2 Cluster Chatbot

## Testing
pip install -r requirements.txt

python app.py

## Build docker image
docker build -t ghcr.io/ucla-oarc-hpc/aicluster/h2chat:latest .
docker push ghcr.io/ucla-oarc-hpc/aicluster/h2chat:latest

## Run docker image
docker run -p 5001:5000  h2chatbot:latest

