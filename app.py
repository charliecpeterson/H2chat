# app.py
# Description: Flask web server for the Hoffman2 Chatbot.
# This application serves the HTML frontend and provides an API endpoint
# for users to ask questions to the LlamaIndex-powered chatbot.

import os
from flask import Flask, request, jsonify, render_template # Added render_template
from werkzeug.utils import secure_filename

from hoffman2_chatbot_web import load_environment, load_hoffman2_index, ask_hoffman2_web

# --- Flask App Initialization ---
# Flask looks for templates in a folder named "templates" by default.
# Explicitly setting template_folder here for clarity.
# **Action Required**: Ensure you have a folder named "templates" in the same
# directory as this app.py file, and that template.html is inside it.
app = Flask(__name__, template_folder='templates')

# --- Configuration ---
app.config['INDEX_PATH'] = "./hoffman2_index"

# --- Load Chatbot Components ---
query_engine = None

def initialize_chatbot():
    global query_engine
    try:
        print("Initializing chatbot environment...")
        load_environment()
        print("Environment loaded.")

        print(f"Loading Hoffman2 index from: {app.config['INDEX_PATH']}")
        query_engine = load_hoffman2_index(app.config['INDEX_PATH'])
        if query_engine:
            print("Chatbot query engine loaded successfully.")
        else:
            print("Failed to load chatbot query engine. The application might not work correctly.")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure your .env file is set up correctly with OPENAI_API_KEY.")
    except FileNotFoundError as fnfe: # More specific error handling for missing index
        print(f"Initialization Error: {fnfe}")
        print(f"Please ensure the index directory exists at: {app.config['INDEX_PATH']}")
    except Exception as e:
        print(f"An unexpected error occurred during chatbot initialization: {e}")

initialize_chatbot()

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Serves the main HTML page for the chatbot using render_template.
    """
    # Define content for the template
    page_title = "Hoffman2 HPC Chatbot"
    header_description_html = "Ask me anything about using the Hoffman2 cluster! I can help with commands, job submission, software modules, and more."
    initial_bot_message_html = "Hello! I'm the Hoffman2 HPC Assistant. How can I help you navigate the cluster today?"
    input_placeholder = "Type your question about Hoffman2..."

    return render_template('template.html',
                           PAGE_TITLE=page_title,
                           HEADER_DESCRIPTION_HTML=header_description_html,
                           INITIAL_BOT_MESSAGE_HTML=initial_bot_message_html,
                           INPUT_PLACEHOLDER=input_placeholder)

@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle questions from the user.
    Expects a JSON payload with "question" and optionally "history".
    Returns a JSON response with the "answer".
    """
    if not query_engine:
        return jsonify({"error": "Chatbot is not initialized. Please check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    question = data.get('question')
    # Get history from the request, default to an empty list if not provided
    chat_history = data.get('history', [])

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "'question' must be a non-empty string."}), 400
    
    # Validate chat_history structure (optional but good practice)
    if not isinstance(chat_history, list):
        return jsonify({"error": "'history' must be a list."}), 400
    for item in chat_history:
        if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
            return jsonify({"error": "Invalid item in 'history'. Each item must be a dict with 'role' and 'content'."}), 400


    print(f"Received question: {question}")
    if chat_history:
        print(f"Received history with {len(chat_history)} entries.")

    try:
        # Pass the question and chat_history to the backend function
        answer, query_time = ask_hoffman2_web(query_engine, question, chat_history=chat_history)
        print(f"Generated answer in {query_time:.2f}s: {str(answer)[:100]}...") # Log snippet, ensure answer is string
        return jsonify({"answer": str(answer), "query_time": query_time}) # Ensure answer is string
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
        return jsonify({"error": "An error occurred while processing your question."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # For development: debug=True allows for hot-reloading and provides a debugger.
    # Host '0.0.0.0' makes the server accessible from other devices on the same network.
    app.run(debug=True, host='0.0.0.0', port=5000)
