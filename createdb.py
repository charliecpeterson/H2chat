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
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SimpleNodeParser, MarkdownNodeParser
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

def load_environment():
    """Load environment variables and check for API key"""
    # Load variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    
    # Create embedding and LLM instances
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAI(model="gpt-4o")
    
    # Set up settings globally
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
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Resolve relative URLs
                full_url = urljoin(url, href)
                
                # Only follow links to the same domain
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    # Skip links to files (PDFs, etc)
                    if not any(full_url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip']):
                        to_visit.append(full_url)
        
        except Exception as e:
            print(f"  Error crawling {url}: {e}")
            
        print(f"Discovered {len(visited)} pages, {len(to_visit)} pages in queue")
    
    print(f"Crawling complete. Found {len(visited)} pages.")
    return list(visited)

def preserve_code_blocks(html_content):
    """
    Pre-process HTML to preserve code blocks, backticks and special characters
    before converting to Markdown
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all code blocks and pre blocks
    for code_block in soup.select('pre, code, .code'):
        # Mark the code block with a special wrapper
        wrapper = soup.new_tag('div')
        wrapper['class'] = 'preserved-code-block'
        code_block.wrap(wrapper)
        
        # Replace the content with a placeholder that won't be affected by markdown conversion
        code_text = code_block.get_text()
        # Replace backticks with a special marker
        code_text = code_text.replace('`', '%%%BACKTICK%%%')
        # Replace hash symbols with special marker at beginning of lines
        code_text = re.sub(r'^(\s*)#', r'\1%%%HASH%%%', code_text, flags=re.MULTILINE)
        code_block.string = code_text
    
    return str(soup)

def restore_code_blocks(markdown_text):
    """
    Restore code blocks with their original formatting after Markdown conversion
    """
    # Restore hash symbols at line beginnings (in code blocks)
    markdown_text = re.sub(r'%%%HASH%%%', '#', markdown_text)
    # Restore backticks
    markdown_text = re.sub(r'%%%BACKTICK%%%', '`', markdown_text)
    
    return markdown_text

def html_to_markdown(html_content, url):
    """Convert HTML to Markdown while preserving structure and code blocks"""
    from markdownify import markdownify as md
    from bs4 import BeautifulSoup
    
    # First preserve code blocks with special markers
    preserved_html = preserve_code_blocks(html_content)
    
    soup = BeautifulSoup(preserved_html, 'html.parser')
    
    # Remove navigation, footers, etc.
    for element in soup.select('nav, footer, .navigation, .menu, #sidebar'):
        element.extract()
    
    # Convert to markdown
    markdown_text = md(str(soup))
    
    # Restore special markers in code blocks
    markdown_text = restore_code_blocks(markdown_text)
    
    # Add source URL as metadata in the markdown
    markdown_text = f"# Source: {url}\n\n{markdown_text}"
    
    # Ensure code blocks use triple backticks
    markdown_text = re.sub(r'```\n', '```bash\n', markdown_text)
    
    return markdown_text

def process_script_file(file_path, repo_name):
    """Process script files to properly format them as code blocks"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get file extension to use as language hint
        extension = os.path.splitext(file_path)[1][1:].lower()
        if extension == 'sh':
            language = 'bash'
        elif extension == 'py':
            language = 'python'
        else:
            language = extension or 'text'
        
        # Format as proper markdown code block
        formatted_content = f"### {os.path.basename(file_path)} START ###\n```{language}\n{content}\n```\n### {os.path.basename(file_path)} STOP ###"
        
        relative_path = os.path.basename(file_path)
        return Document(
            text=formatted_content,
            metadata={
                "source": f"{repo_name}/{relative_path}",
                "title": f"Script: {relative_path}",
                "script": "true"
            }
        )
    except Exception as e:
        print(f"Error processing script file {file_path}: {e}")
        return None

def process_github_repo(repo_url, cleaned_documents):
    """Clone a GitHub repository and extract useful content for the knowledge base"""
    print(f"Processing GitHub repository: {repo_url}")
    
    # Extract repo name from URL for reference
    repo_name = repo_url.split('/')[-1]
    
    # Create a temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone the repository
            print(f"Cloning {repo_url} to {temp_dir}...")
            git.Repo.clone_from(repo_url, temp_dir, depth=1)
            
            # Process markdown files (which usually contain documentation)
            markdown_files = list(Path(temp_dir).glob("**/*.md"))
            print(f"Found {len(markdown_files)} markdown files")
            
            # Process notebooks
            notebook_files = list(Path(temp_dir).glob("**/*.ipynb"))
            print(f"Found {len(notebook_files)} notebook files")
            
            # Process script files (sh, bash, py)
            script_files = []
            script_files.extend(list(Path(temp_dir).glob("**/*.sh")))
            script_files.extend(list(Path(temp_dir).glob("**/*.bash")))
            script_files.extend(list(Path(temp_dir).glob("**/*.py")))
            print(f"Found {len(script_files)} script files")
            
            # Process markdown files
            for md_file in markdown_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip very short files or those that are likely just placeholders
                    if len(content) < 100:
                        continue
                    
                    # Create a document
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
            
            # Process notebooks (convert to markdown and extract code + text)
            for nb_file in notebook_files:
                try:
                    import nbformat
                    import json
                    
                    # Read the notebook
                    with open(nb_file, 'r', encoding='utf-8') as f:
                        try:
                            notebook = nbformat.read(f, as_version=4)
                            content_parts = []
                            
                            # Extract cells
                            for cell in notebook.cells:
                                if cell.cell_type == 'markdown':
                                    content_parts.append(cell.source)
                                elif cell.cell_type == 'code':
                                    content_parts.append(f"```python\n{cell.source}\n```")
                            
                            content = "\n\n".join(content_parts)
                            
                            # Create a document
                            relative_path = nb_file.relative_to(temp_dir)
                            doc = Document(
                                text=content,
                                metadata={
                                    "source": f"{repo_name}/{relative_path}",
                                    "repo": repo_url,
                                    "title": f"{repo_name}: {relative_path}"
                                }
                            )
                            cleaned_documents.append(doc)
                            print(f"Added notebook content from {relative_path}")
                        except json.JSONDecodeError:
                            print(f"Error parsing notebook {nb_file} - invalid JSON format")
                except ImportError:
                    print("Skipping notebook processing - nbformat not available")
                except Exception as e:
                    print(f"Error processing notebook {nb_file}: {e}")
            
            # Process script files with special handling
            for script_file in script_files:
                doc = process_script_file(script_file, repo_name)
                if doc:
                    cleaned_documents.append(doc)
                    print(f"Added script content from {script_file.name}")
            
            # Look for a README file and add special metadata
            readme_files = [f for f in markdown_files if f.name.lower() == 'readme.md']
            if readme_files:
                try:
                    with open(readme_files[0], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create a workshop summary document
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
                    print(f"Added workshop summary for {repo_name}")
                except Exception as e:
                    print(f"Error processing README for {repo_name}: {e}")
                
        except Exception as e:
            print(f"Error processing repository {repo_url}: {e}")


def main():
    # Load environment settings
    load_environment()

    # Remove existing index if it exists
    if os.path.exists("./hoffman2_index"):
        print("Removing existing index...")
        shutil.rmtree("./hoffman2_index")
    
    # Define the base URL
    base_url = "https://www.hoffman2.idre.ucla.edu"
    
    # Discover all pages by crawling
    print("Discovering pages by crawling the website...")
    urls_to_scrape = get_all_links(base_url, max_pages=30)
    
    print(f"Found {len(urls_to_scrape)} pages to scrape:")
    for url in urls_to_scrape:
        print(f"  - {url}")
    
    print("Scraping documentation from website...")
    try:
        # Use our custom HTML to Markdown conversion
        documents = []
        for url in urls_to_scrape:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    markdown_content = html_to_markdown(response.content, url)
                    doc = Document(
                        text=markdown_content,
                        metadata={"source": url, "title": f"Page: {url}"}
                    )
                    documents.append(doc)
                    print(f"Converted {url} to markdown")
                else:
                    print(f"Failed to get {url}: status code {response.status_code}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error during web scraping: {e}")
        return
    
    # Clean up documents and create new ones
    cleaned_documents = []
    for i, doc in enumerate(documents):
        try:
            # Clean the text and create a new document
            text = doc.text.encode('utf-8', errors='ignore').decode('utf-8')
            cleaned_doc = Document(text=text, metadata=doc.metadata)
            cleaned_documents.append(cleaned_doc)
            print(f"Cleaned document {i+1}/{len(documents)}")
        except Exception as e:
            print(f"Error cleaning document {i+1}: {e}")
    
    print(f"Cleaned {len(cleaned_documents)} documents")
    
    if len(cleaned_documents) == 0:
        print("Error: No documents were successfully cleaned. Cannot create index.")
        return

    workshop_repos = [
        "https://github.com/ucla-oarc-hpc/WS_LLMonHPC",
        "https://github.com/ucla-oarc-hpc/WS_MLonHPC",
        "https://github.com/ucla-oarc-hpc/WS_containers",
        "https://github.com/ucla-oarc-hpc/WS_MakingContainers",
        "https://github.com/ucla-oarc-hpc/WS_VisualizationHPC",
        "https://github.com/ucla-oarc-hpc/WS_CompilingOnHPC",
        "https://github.com/ucla-oarc-hpc/WS_BigDataOnHPC",
        "https://github.com/ucla-oarc-hpc/H2-RStudio",
        "https://github.com/ucla-oarc-hpc/H2HH_rstudio",
        "https://github.com/ucla-oarc-hpc/H2HH_anaconda",
        "https://github.com/ucla-oarc-hpc/H2HH_Python-R",
    ]
    
    # Process each workshop repository
    for repo_url in workshop_repos:
        process_github_repo(repo_url, cleaned_documents)
 
    # Add supplementary documents with beginner-friendly explanations
    beginner_docs = [
        {
            "title": "Getting Started with Hoffman2",
            "text": """
            # Hoffman2 Basics 
            
            Hoffman2 is UCLA's High-Performance Computing (HPC) cluster. Think of it as a very powerful 
            computer system that you can access remotely to run computationally intensive tasks.
            
            ## Basic Steps to Use Hoffman2:
            
            1. **Logging in**: Use SSH to connect with your UCLA ID
               ssh your_username@hoffman2.idre.ucla.edu
               
            2. **Transferring files**: Use SCP or SFTP to move files to/from Hoffman2
               scp your_file your_username@dtn.hoffman2.idre.ucla.edu:~/destination
               
            3. **Running jobs**: Always use job scheduling system (don't run directly on login nodes)
               qsub my_job_script.sh
               
            4. **Checking job status**:
               qstat -u your_username
            """
        },
        {
            "title": "Common Hoffman2 Commands for New Users",
            "text": """
            # Essential Hoffman2 Commands
            
            ## File Management
            - `ls` - List files
            - `cd` - Change directory
            - `mkdir` - Make directory
            - `rm` - Remove files
            - `mv` - Move files
            
            ## Job Management
            - `qsub script.sh` - Submit a job
            - `qstat -u username` - Check job status
            - `qdel job_id` - Delete a job
            
            ## Module Management
            - `modules_lookup -a` - See available software
            - `modules_lookup -f software_name` - Find specific software
            - `module load software_name` - Load software
            - `module list` - See loaded modules
            """
        },
        {
            "title": "Common Errors and Solutions on Hoffman2",
            "text": """
            # Common Hoffman2 Problems and Solutions
            
            ## Out of Memory Errors
            If your job fails with "out of memory" errors, request more memory:
            ```bash
            #$ -l h_data=4G  (for 4GB of memory)
            ```
            
            ## Job Running Too Long
            If your job hits the time limit, increase it:
            ```bash
            #$ -l h_rt=24:00:00  (for 24 hours)
            ```
            
            ## Software Not Found
            Always load modules before using software:
            ```bash
            module load python/3.9.6
            ```
            """
        }
    ]

    # Add these beginner documents to your collection
#    for i, doc_dict in enumerate(beginner_docs):
#        supplementary_doc = Document(
#            text=doc_dict["text"],
#            metadata={"source": f"beginner_guide_{i}", "title": doc_dict["title"]}
#        )
#        cleaned_documents.append(supplementary_doc)
#        print(f"Added supplementary document: {doc_dict['title']}")

    # Parse documents into nodes for indexing
    from llama_index.core.node_parser import MarkdownNodeParser
    
    parser = MarkdownNodeParser(
        chunk_size=512,
        chunk_overlap=50,
        markdown_header_splits=True  
    )

    nodes = parser.get_nodes_from_documents(cleaned_documents)
    print(f"Created {len(nodes)} nodes")
    
    if len(nodes) == 0:
        print("Error: No nodes were created from documents. Cannot create index.")
        return
    
    # Use SimpleVectorStore instead of FAISS
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index
    print("Building the index...")
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # Save the index to disk
    print("Saving index to disk...")
    index.storage_context.persist("./hoffman2_index")
    
    print("Index created and saved successfully!")
    
    # Test the index
    query_engine = index.as_query_engine(response_mode="compact")
    test_question = "How do I log in to Hoffman2?"
    print(f"\nTesting with question: {test_question}")
    try:
        response = query_engine.query(test_question)
        print(f"Answer: {response.response}")
        print("\nHoffman2 Documentation Chatbot is ready. You can now interact with it!")
    except Exception as e:
        print(f"Error during test query: {e}")

if __name__ == "__main__":
    main()