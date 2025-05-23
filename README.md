# Hoffman2 Cluster Chatbot

## Testing
pip install -r requirements.txt

python frontend/build_frontend.py

python app.py

## Build docker image
docker build -t h2chatbot:latest .

## Run docker image
docker run -p 5001:5000  h2chatbot:latest