# app.py
# Description: Flask web server for the Hoffman2 Chatbot.
# This application serves the HTML frontend and provides an API endpoint
# for users to ask questions to the LlamaIndex-powered chatbot.

import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename # For potential future file uploads, not used now but good practice

# Import functions from your existing chatbot script
# We'll assume hoffman2_chatbot.py is in the same directory
# and we'll modify it slightly.
from hoffman2_chatbot_web import load_environment, load_hoffman2_index, ask_hoffman2_web

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static') # Define static folder for CSS, JS, images

# --- Configuration ---
# It's good practice to configure your app, e.g., for secret keys or environment-specific settings.
# For now, we'll keep it simple.
# app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_development')
app.config['INDEX_PATH'] = "./hoffman2_index" # Path to your LlamaIndex

# --- Load Chatbot Components ---
# These will be loaded once when the Flask app starts.
query_engine = None

def initialize_chatbot():
    """
    Loads environment variables and the LlamaIndex query engine.
    This function is called when the Flask app starts.
    """
    global query_engine
    try:
        print("Initializing chatbot environment...")
        load_environment() # Loads OpenAI API Key from .env
        print("Environment loaded.")

        print(f"Loading Hoffman2 index from: {app.config['INDEX_PATH']}")
        query_engine = load_hoffman2_index(app.config['INDEX_PATH'])
        if query_engine:
            print("Chatbot query engine loaded successfully.")
        else:
            print("Failed to load chatbot query engine. The application might not work correctly.")
            # You might want to raise an error here or prevent the app from starting
            # if the index is critical and not found.
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure your .env file is set up correctly with OPENAI_API_KEY.")
        # Depending on severity, you might want to exit or disable chat functionality
    except Exception as e:
        print(f"An unexpected error occurred during chatbot initialization: {e}")
        # Log this error appropriately in a production environment

# Call initialization when the app is created
initialize_chatbot()

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Serves the main HTML page for the chatbot.
    Looks for index.html in the 'static' folder.
    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle questions from the user.
    Expects a JSON payload with a "question" field.
    Returns a JSON response with the "answer".
    """
    if not query_engine:
        return jsonify({"error": "Chatbot is not initialized. Please check server logs."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request JSON."}), 400

    question = data['question']
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "'question' must be a non-empty string."}), 400

    print(f"Received question: {question}")

    try:
        answer, query_time = ask_hoffman2_web(query_engine, question)
        print(f"Generated answer in {query_time:.2f}s: {answer}")
        return jsonify({"answer": answer, "query_time": query_time})
    except Exception as e:
        # Log the exception in a real application
        print(f"Error processing question '{question}': {e}")
        # Consider more specific error handling based on exception types
        return jsonify({"error": "An error occurred while processing your question."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # IMPORTANT: For development only.
    # In production, use a WSGI server like Gunicorn or uWSGI.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    # Ensure OPENAI_API_KEY is set in the environment where this app runs.
    app.run(debug=True, host='0.0.0.0', port=5000)