# build_frontend.py
# Description: Generates static/index.html from template.html and content files
# located in the same directory as this script (i.e., the 'frontend' subdirectory).
# The output static/index.html will be placed one level up from this script's location.
# Run this script from within the 'frontend' directory to update the HTML.

import json
import markdown # Requires 'pip install Markdown'
import os

def main():
    """
    Main function to build the index.html file.
    """
    print("Building frontend: static/index.html...")

    # Define file paths
    # script_dir is the directory where build_frontend.py is located (e.g., AIH2DOCS/frontend/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Source files are in the same directory as this script
    config_file_path = os.path.join(script_dir, "page_elements.json")
    header_md_path = os.path.join(script_dir, "header_description.md")
    initial_message_md_path = os.path.join(script_dir, "initial_bot_message.md")
    template_html_path = os.path.join(script_dir, "template.html")
    
    # Project root is one level up from the script_dir (e.g., AIH2DOCS/)
    project_root_for_static = os.path.dirname(script_dir)
    
    # Output path for static/index.html should be in the project root's static folder
    output_static_dir = os.path.join(project_root_for_static, "static")
    output_html_path = os.path.join(output_static_dir, "index.html")

    # Ensure static directory exists at the project root level
    os.makedirs(output_static_dir, exist_ok=True)

    # 1. Load JSON config
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"Loaded configuration from {config_file_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}")
        config_data = {
            "page_title": "Hoffman2 HPC Chatbot (Default)",
            "input_placeholder": "Type your question..."
        }
        print("Using default configuration values.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_file_path}")
        return

    # 2. Load and convert Markdown for header description
    try:
        with open(header_md_path, 'r', encoding='utf-8') as f:
            header_md_content = f.read()
        header_html_content = markdown.markdown(header_md_content, extensions=['fenced_code', 'tables'])
        print(f"Converted header description from {header_md_path}")
    except FileNotFoundError:
        print(f"Warning: Header Markdown file not found at {header_md_path}. Using empty string.")
        header_html_content = "<p><em>Header content missing.</em></p>"

    # 3. Load and convert Markdown for initial bot message
    try:
        with open(initial_message_md_path, 'r', encoding='utf-8') as f:
            initial_message_md_content = f.read()
        initial_bot_message_html = markdown.markdown(initial_message_md_content, extensions=['fenced_code', 'tables'])
        print(f"Converted initial bot message from {initial_message_md_path}")
    except FileNotFoundError:
        print(f"Warning: Initial bot message Markdown file not found at {initial_message_md_path}. Using default message.")
        initial_bot_message_html = "<p>Hello! How can I help you today? (Default)</p>"

    # 4. Load HTML template
    try:
        with open(template_html_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print(f"Loaded HTML template from {template_html_path}")
    except FileNotFoundError:
        print(f"Error: HTML template file not found at {template_html_path}. Cannot build frontend.")
        return

    # 5. Replace placeholders in the template
    final_html_content = template_content.replace("{{PAGE_TITLE}}", config_data.get("page_title", "Chatbot"))
    final_html_content = final_html_content.replace("{{HEADER_DESCRIPTION_HTML}}", header_html_content)
    final_html_content = final_html_content.replace("{{INITIAL_BOT_MESSAGE_HTML}}", initial_bot_message_html)
    final_html_content = final_html_content.replace("{{INPUT_PLACEHOLDER}}", config_data.get("input_placeholder", "Type here..."))
    print("Replaced placeholders in template.")

    # 6. Write the final HTML to static/index.html
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html_content)
        print(f"Successfully generated {output_html_path}")
    except IOError as e:
        print(f"Error writing final HTML to {output_html_path}: {e}")

if __name__ == "__main__":
    main()

