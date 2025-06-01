import os
import shutil
import json
import requests
import git
import re
import tempfile
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path # Added import
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SimpleNodeParser, MarkdownNodeParser
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Define a constant for chunking large code blocks (in characters)
# This is a heuristic. Adjust if you still see token limit errors.
# OpenAI: ~4 chars/token for English. Code might be different.
# text-embedding-3-large limit is 8192 tokens.
# 7000 chars * (1 token / ~2-4 chars for code) = ~1750-3500 tokens.
# This should be well within limits for a single chunk, allowing parser some room.
MAX_CODE_CONTENT_CHAR_LENGTH = 7000

def load_environment():
    """Load environment variables and check for API key"""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAI(model="gpt-4o")
    Settings.embed_model = embed_model
    Settings.llm = llm

def get_all_links(base_url, max_pages=50):
    """
    Recursively crawl a website to find all internal links, up to max_pages
    """
    visited = set()
    to_visit = [base_url]
    domain = urlparse(base_url).netloc
    print(f"Starting crawler with base domain: {domain}")
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            print(f"Visiting {url}")
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"  Failed to get {url}: status code {response.status_code}")
                continue
            visited.add(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    if not any(full_url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.gz', '.tgz', '.ipynb']): # Added more extensions
                        to_visit.append(full_url)
        except Exception as e:
            print(f"  Error crawling {url}: {e}")
        print(f"Discovered {len(visited)} pages, {len(to_visit)} pages in queue")
    print(f"Crawling complete. Found {len(visited)} pages.")
    return list(visited)

def preserve_code_blocks(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for code_block in soup.select('pre, code, .code'): # .code might be too general
        wrapper = soup.new_tag('div')
        wrapper['class'] = 'preserved-code-block'
        code_block.wrap(wrapper)
        code_text = code_block.get_text()
        code_text = code_text.replace('`', '%%%BACKTICK%%%')
        code_text = re.sub(r'^(\s*)#', r'\1%%%HASH%%%', code_text, flags=re.MULTILINE)
        code_block.string = code_text
    return str(soup)

def restore_code_blocks(markdown_text):
    markdown_text = re.sub(r'%%%HASH%%%', '#', markdown_text)
    markdown_text = re.sub(r'%%%BACKTICK%%%', '`', markdown_text)
    return markdown_text

def html_to_markdown(html_content, url):
    from markdownify import markdownify as md
    from bs4 import BeautifulSoup
    preserved_html = preserve_code_blocks(html_content)
    soup = BeautifulSoup(preserved_html, 'html.parser')
    for element in soup.select('nav, footer, .navigation, .menu, #sidebar, .sidebar, .footer, .header, #header, #footer'): # Added more common selectors
        element.extract()
    markdown_text = md(str(soup))
    markdown_text = restore_code_blocks(markdown_text)
    markdown_text = f"# Source: {url}\n\n{markdown_text}"
    # This regex is a bit aggressive, might make non-bash blocks bash.
    # Consider a more nuanced approach if you have varied code block types from HTML.
    markdown_text = re.sub(r'```(\s*\n)', r'```text\1', markdown_text) # Default to text if no lang
    markdown_text = re.sub(r'```\n', '```text\n', markdown_text) # Ensure all have at least 'text'
    return markdown_text

def process_script_file(file_path, repo_name):
    """Process script files to properly format them as code blocks.
    Returns a list of Document objects."""
    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        extension = os.path.splitext(file_path)[1][1:].lower()
        language = extension or 'text' # Default to 'text' if no extension
        if extension == 'sh': language = 'bash'
        elif extension == 'py': language = 'python'
        # Add more language mappings if needed

        # If content is too large, split it into chunks
        if len(content) > MAX_CODE_CONTENT_CHAR_LENGTH:
            num_chunks = (len(content) + MAX_CODE_CONTENT_CHAR_LENGTH - 1) // MAX_CODE_CONTENT_CHAR_LENGTH
            for i in range(num_chunks):
                chunk_content = content[i*MAX_CODE_CONTENT_CHAR_LENGTH:(i+1)*MAX_CODE_CONTENT_CHAR_LENGTH]
                formatted_content = (
                    f"### {os.path.basename(file_path)} (Part {i+1}/{num_chunks}) START ###\n"
                    f"```{language}\n{chunk_content}\n```\n"
                    f"### {os.path.basename(file_path)} (Part {i+1}/{num_chunks}) STOP ###"
                )
                relative_path = os.path.basename(file_path)
                doc = Document(
                    text=formatted_content,
                    metadata={
                        "source": f"{repo_name}/{relative_path} (Part {i+1})",
                        "title": f"Script: {relative_path} (Part {i+1})",
                        "script": "true",
                        "language": language
                    }
                )
                docs.append(doc)
        else:
            formatted_content = (
                f"### {os.path.basename(file_path)} START ###\n"
                f"```{language}\n{content}\n```\n"
                f"### {os.path.basename(file_path)} STOP ###"
            )
            relative_path = os.path.basename(file_path)
            doc = Document(
                text=formatted_content,
                metadata={
                    "source": f"{repo_name}/{relative_path}",
                    "title": f"Script: {relative_path}",
                    "script": "true",
                    "language": language
                }
            )
            docs.append(doc)
        return docs
    except Exception as e:
        print(f"Error processing script file {file_path}: {e}")
        return [] # Return an empty list on error

def process_github_repo(repo_url, cleaned_documents):
    print(f"Processing GitHub repository: {repo_url}")
    repo_name = repo_url.split('/')[-1]

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Cloning {repo_url} to {temp_dir}...")
            git.Repo.clone_from(repo_url, temp_dir, depth=1) # Use depth=1 for faster clone if history not needed

            # Process markdown files
            markdown_files = list(Path(temp_dir).glob("**/*.md"))
            print(f"Found {len(markdown_files)} markdown files")
            for md_file in markdown_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) < 100: continue # Skip very short files

                    relative_path = md_file.relative_to(temp_dir)
                    doc = Document(
                        text=content,
                        metadata={
                            "source": f"{repo_name}/{relative_path}",
                            "repo": repo_url,
                            "title": f"{repo_name}: {relative_path}"
                        }
                    )
                    cleaned_documents.append(doc)
                    print(f"Added document from {relative_path}")
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")

            # Process notebooks
            notebook_files = list(Path(temp_dir).glob("**/*.ipynb"))
            print(f"Found {len(notebook_files)} notebook files")
            for nb_file in notebook_files:
                try:
                    import nbformat # Keep import here in case not always available
                    # import json # Already imported globally

                    with open(nb_file, 'r', encoding='utf-8') as f:
                        notebook = nbformat.read(f, as_version=4)
                    content_parts = []
                    for i, cell in enumerate(notebook.cells):
                        if cell.cell_type == 'markdown':
                            content_parts.append(f"\n\n## Notebook Markdown Cell {i+1}\n\n{cell.source}")
                        elif cell.cell_type == 'code':
                            source_code = cell.source
                            # If a single code cell is massive, break it into multiple markdown code blocks
                            if len(source_code) > MAX_CODE_CONTENT_CHAR_LENGTH:
                                num_code_chunks = (len(source_code) + MAX_CODE_CONTENT_CHAR_LENGTH - 1) // MAX_CODE_CONTENT_CHAR_LENGTH
                                for k in range(num_code_chunks):
                                    chunked_code_cell = source_code[k*MAX_CODE_CONTENT_CHAR_LENGTH:(k+1)*MAX_CODE_CONTENT_CHAR_LENGTH]
                                    content_parts.append(f"\n\n### Code Cell {i+1} (Part {k+1}/{num_code_chunks})\n```python\n{chunked_code_cell}\n```")
                            elif source_code.strip(): # Only add if there's actual code
                                content_parts.append(f"\n\n### Code Cell {i+1}\n```python\n{source_code}\n```")
                    
                    if not content_parts: continue # Skip empty notebooks

                    content = "\n\n".join(content_parts)
                    relative_path = nb_file.relative_to(temp_dir)
                    doc = Document(
                        text=content,
                        metadata={
                            "source": f"{repo_name}/{relative_path}",
                            "repo": repo_url,
                            "title": f"{repo_name}: {relative_path} (Notebook)"
                        }
                    )
                    cleaned_documents.append(doc)
                    print(f"Added notebook content from {relative_path}")
                except ImportError:
                    print("Skipping notebook processing - nbformat not available. Install with 'pip install nbformat'")
                except json.JSONDecodeError: # More specific error
                    print(f"Error parsing notebook {nb_file} - invalid JSON format")
                except Exception as e:
                    print(f"Error processing notebook {nb_file}: {e}")

            # Process script files
            script_extensions = ["*.sh", "*.bash", "*.py", "*.pl", "*.rb", "*.js"] # Add more script types
            script_files = []
            for ext in script_extensions:
                script_files.extend(list(Path(temp_dir).glob(f"**/{ext}")))
            print(f"Found {len(script_files)} script files")
            for script_file in script_files:
                # process_script_file now returns a list of documents
                docs_from_script = process_script_file(script_file, repo_name)
                if docs_from_script:
                    cleaned_documents.extend(docs_from_script) # Use extend for lists
                    print(f"Added script content from {script_file.name}")

            # README processing (simplified, as it's also caught by markdown_files)
            readme_file = Path(temp_dir) / 'README.md'
            if readme_file.exists():
                for doc in cleaned_documents:
                    if doc.metadata.get("source") == f"{repo_name}/README.md":
                        doc.metadata["title"] = f"Workshop Summary: {repo_name}"
                        doc.metadata["workshop"] = "true"
                        doc.text = f"# Workshop: {repo_name}\n\n{doc.text}" # Prepend title
                        print(f"Enhanced workshop summary for {repo_name} from README.md")
                        break
                else: # if README.md was too short and skipped, or to be sure
                    try:
                        with open(readme_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if len(content) >= 100: # ensure it's substantial if re-adding
                            doc = Document(
                                text=f"# Workshop: {repo_name}\n\n{content}",
                                metadata={
                                    "source": f"{repo_name}/README.md",
                                    "repo": repo_url,
                                    "title": f"Workshop Summary: {repo_name}",
                                    "workshop": "true"
                                }
                            )
                            cleaned_documents.append(doc)
                            print(f"Added workshop summary for {repo_name} (fallback)")
                    except Exception as e:
                         print(f"Error processing README for {repo_name} (fallback): {e}")


        except git.exc.GitCommandError as e:
            print(f"Git clone error for {repo_url}: {e}")
        except Exception as e:
            print(f"Error processing repository {repo_url}: {e}")


def main():
    load_environment()

    if os.path.exists("./hoffman2_index"):
        print("Removing existing index...")
        shutil.rmtree("./hoffman2_index")
    
    base_url = "https://www.hoffman2.idre.ucla.edu"
    print("Discovering pages by crawling the website...")
    urls_to_scrape = get_all_links(base_url, max_pages=30) # Reduced for testing, increase as needed
    
    print(f"Found {len(urls_to_scrape)} pages to scrape:")
    for url in urls_to_scrape: print(f"  - {url}")
    
    scraped_documents = []
    print("Scraping documentation from website...")
    for url in urls_to_scrape:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Ensure content is not too large before processing
                if len(response.content) > 5_000_000: # Skip extremely large HTML files (5MB)
                    print(f"Skipping {url} due to excessive size: {len(response.content)} bytes")
                    continue
                markdown_content = html_to_markdown(response.content, url)
                doc = Document(
                    text=markdown_content,
                    metadata={"source": url, "title": f"Page: {url.split('/')[-1] or urlparse(url).netloc}"}
                )
                scraped_documents.append(doc)
                print(f"Converted {url} to markdown")
            else:
                print(f"Failed to get {url}: status code {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"Timeout processing {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
                
    print(f"Loaded {len(scraped_documents)} documents from web scraping.")
    
    cleaned_documents = []
    for i, doc in enumerate(scraped_documents):
        try:
            # Basic cleaning: replace null bytes, ensure valid UTF-8
            text = doc.text.replace('\x00', '').encode('utf-8', errors='replace').decode('utf-8')
            # Further cleaning: multiple newlines, leading/trailing whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()
            if not text: # Skip if document becomes empty after cleaning
                print(f"Skipped document {i+1} from {doc.metadata.get('source')} as it became empty after cleaning.")
                continue
            cleaned_doc = Document(text=text, metadata=doc.metadata)
            cleaned_documents.append(cleaned_doc)
            # print(f"Cleaned document {i+1}/{len(scraped_documents)}") # Can be too verbose
        except Exception as e:
            print(f"Error cleaning document {i+1} from {doc.metadata.get('source')}: {e}")
    
    print(f"Processed {len(cleaned_documents)} documents from web scraping after cleaning.")
    
    workshop_repos = [
        "https://github.com/ucla-oarc-hpc/WS_LLMonHPC",
        "https://github.com/ucla-oarc-hpc/WS_MLonHPC",
        "https://github.com/ucla-oarc-hpc/WS_containers",
        # "https://github.com/ucla-oarc-hpc/WS_MakingContainers", # Often has large binaries
        "https://github.com/ucla-oarc-hpc/WS_VisualizationHPC",
        "https://github.com/ucla-oarc-hpc/WS_CompilingOnHPC",
        "https://github.com/ucla-oarc-hpc/WS_BigDataOnHPC",
        "https://github.com/ucla-oarc-hpc/H2-RStudio",
        "https://github.com/ucla-oarc-hpc/H2HH_rstudio",
        "https://github.com/ucla-oarc-hpc/H2HH_anaconda",
        "https://github.com/ucla-oarc-hpc/H2HH_Python-R",
    ]
    
    for repo_url in workshop_repos:
        process_github_repo(repo_url, cleaned_documents)
 
    beginner_docs_data = [ # Renamed to avoid conflict
        {
            "title": "Getting Started with Hoffman2",
            "text": """
            # Hoffman2 Basics
            Hoffman2 is UCLA's High-Performance Computing (HPC) cluster. Think of it as a very powerful
            computer system that you can access remotely to run computationally intensive tasks.
            ## Basic Steps to Use Hoffman2:
            1. **Logging in**: Use SSH to connect with your UCLA ID: `ssh your_username@hoffman2.idre.ucla.edu`
            2. **Transferring files**: Use SCP or SFTP to move files to/from Hoffman2. Example: `scp your_file your_username@dtn.hoffman2.idre.ucla.edu:~/destination` (DTN nodes are recommended for transfers).
            3. **Running jobs**: Always use job scheduling system (UGE - Univa Grid Engine). Don't run directly on login nodes. Example: `qsub my_job_script.sh`
            4. **Checking job status**: Example: `qstat -u your_username`
            """
        },
        {
            "title": "Common Hoffman2 Commands for New Users",
            "text": """
            # Essential Hoffman2 Commands
            ## File Management
            - `ls` - List files and directories
            - `cd path/to/directory` - Change directory
            - `pwd` - Print working directory (shows current location)
            - `mkdir new_directory_name` - Make a new directory
            - `rm file_name` - Remove a file (use with caution!)
            - `rmdir directory_name` - Remove an empty directory
            - `cp source_file destination_file_or_directory` - Copy files
            - `mv old_name_or_path new_name_or_path` - Move or rename files/directories
            ## Job Management (UGE)
            - `qsub script.sh` - Submit a job script
            - `qstat` - Show status of all jobs in the queue
            - `qstat -u your_username` - Show status of your jobs
            - `qdel job_id` - Delete a submitted or running job
            - `qmon` - (If X11 forwarding is enabled) A graphical interface for job submission and monitoring.
            ## Module Management (Lmod)
            - `module avail` or `module spider` - See all available software modules
            - `module avail software_name` or `module spider software_name` - Search for specific software
            - `module load software_name/version` - Load a specific software module (e.g., `module load python/3.9.6`)
            - `module list` - See currently loaded modules
            - `module unload software_name` - Unload a software module
            - `module purge` - Unload all currently loaded modules
            """
        },
        {
            "title": "Common Errors and Solutions on Hoffman2",
            "text": """
            # Common Hoffman2 Problems and Solutions
            ## Out of Memory Errors
            If your job fails with "out of memory" errors, or is killed by the system, you likely need to request more memory in your job script.
            Add this line to your UGE job script (e.g., for 4GB of memory per core/slot):
            ```bash
            #$ -l h_data=4G
            ```
            ## Job Running Too Long (Hitting Walltime)
            If your job is terminated because it reached the maximum allowed run time (walltime), request more time.
            Add this line (e.g., for 24 hours):
            ```bash
            #$ -l h_rt=24:00:00
            ```
            """
        }
    ]

    for i, doc_dict in enumerate(beginner_docs_data): # Uncommented and renamed
        supplementary_doc = Document(
            text=doc_dict["text"],
            metadata={"source": f"beginner_guide_{i+1}", "title": doc_dict["title"], "beginner_guide": "true"}
        )
        cleaned_documents.append(supplementary_doc)
        print(f"Added supplementary document: {doc_dict['title']}")

    if not cleaned_documents: # Check if list is empty
        print("Error: No documents were successfully processed or found. Cannot create index.")
        return

    print(f"Total documents to be indexed: {len(cleaned_documents)}")

    parser = MarkdownNodeParser(
        chunk_size=512, # This is in tokens for OpenAI model context
        chunk_overlap=50, # Tokens
        # markdown_header_splits=True # Default is True, explicitly stating if needed
    )

    print("Parsing documents into nodes...")
    nodes = parser.get_nodes_from_documents(cleaned_documents, show_progress=True)
    print(f"Created {len(nodes)} nodes")
    
    if not nodes: # Check if list is empty
        print("Error: No nodes were created from documents. Cannot create index.")
        return
    
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("Building the index (this may take a while)...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("Saving index to disk...")
    index.storage_context.persist("./hoffman2_index")
    print("Index created and saved successfully!")
    
    print("\n--- Testing Index ---")
    query_engine = index.as_query_engine(response_mode="compact") # "compact" is a good default
    test_questions = [
        "How do I log in to Hoffman2?",
        "How to run a python script on Hoffman2?",
        "What is qsub?",
        "Tell me about machine learning workshops"
    ]
    for test_question in test_questions:
        print(f"\nTesting with question: {test_question}")
        try:
            response = query_engine.query(test_question)
            print(f"Answer: {response.response}")
            # print(f"Source nodes: {response.source_nodes}") # For debugging
        except Exception as e:
            print(f"Error during test query for '{test_question}': {e}")
    print("\n--- Test Complete ---")
    print("\nHoffman2 Documentation Chatbot index is ready.")

if __name__ == "__main__":
    main()
