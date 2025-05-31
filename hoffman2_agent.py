#!/usr/bin/env python3
import click
import os
import subprocess
import json # Still useful for internal data, though LLM handles tool arg parsing
from openai import OpenAI as OpenAIClient # For direct client if needed, but LlamaIndex LLM is primary
from dotenv import load_dotenv, set_key

# LlamaIndex imports
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI # LlamaIndex's OpenAI LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentRunner
from llama_index.core.agent import ReActAgent # Alternative agent type

# For potentially nicer terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Confirm
    console = Console()
except ImportError:
    console = None
    class Confirm: # Basic fallback if rich is not installed
        @staticmethod
        def ask(prompt, default=False):
            response = input(f"{prompt} [{'y/N' if not default else 'Y/n'}] ").strip().lower()
            if not response: return default
            return response == 'y'

# --- Configuration ---
CONFIG_DIR = os.path.expanduser("~/.config/hpc_agent")
ENV_FILE = os.path.join(CONFIG_DIR, ".env")
DEFAULT_INDEX_PATH = "./hoffman2_index" 

# Global LlamaIndex RAG query engine (can be wrapped in a tool)
rag_query_engine = None
# Global API key store
_api_key_global = None


# --- Helper Functions ---
def ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_api_key_from_config():
    global _api_key_global
    ensure_config_dir()
    load_dotenv(dotenv_path=ENV_FILE)
    _api_key_global = os.getenv("OPENAI_API_KEY")
    return _api_key_global

def save_api_key_to_config(api_key_val):
    global _api_key_global
    ensure_config_dir()
    set_key(ENV_FILE, "OPENAI_API_KEY", api_key_val)
    _api_key_global = api_key_val
    output_msg = f"API key saved to {ENV_FILE}"
    if console: console.print(output_msg)
    else: print(output_msg)

def initialize_llama_index_settings(api_key):
    if not api_key:
        msg = "[bold red]Error: OpenAI API key not found for LlamaIndex settings.[/bold red]"
        if console: console.print(msg)
        else: print(msg[11:-12]) # Strip markdown for plain print
        raise click.Abort()
    try:
        Settings.llm = LlamaOpenAI(model="gpt-4o", api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
        if console: console.print("LlamaIndex OpenAI models configured.")
    except Exception as e:
        msg = f"[bold red]Failed to initialize LlamaIndex OpenAI models: {e}[/bold red]"
        if console: console.print(msg)
        else: print(msg[11:-12])
        raise click.Abort()

def initialize_rag_engine(index_path):
    global rag_query_engine
    if not (os.path.exists(index_path) and os.path.isdir(index_path)):
        msg = f"[yellow]Warning: Index directory not found at {index_path}. RAG features will be limited.[/yellow]"
        if console: console.print(msg)
        else: print(msg[9:-10])
        return

    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        rag_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        if console: console.print(f"RAG engine loaded from {index_path}.")
    except Exception as e:
        msg = f"[bold red]Failed to load RAG engine: {e}. Ensure 'createdb.py' ran successfully.[/bold red]"
        if console: console.print(msg)
        else: print(msg[11:-12])

def confirm_execution(command_str):
    prompt_text = f"Agent wants to run: [cyan]{command_str}[/cyan]\nProceed?"
    if console:
        return Confirm.ask(f"[bold yellow]{prompt_text}[/bold yellow]", default=False)
    else:
        user_input = input(f"Agent wants to run: {command_str}\nProceed? (y/N): ").strip().lower()
        return user_input == 'y'

def execute_local_command(command_parts):
    command_str = " ".join(command_parts)
    if not confirm_execution(command_str):
        return "User cancelled command execution." # Return as a string indicating cancellation

    try:
        process = subprocess.run(command_parts, capture_output=True, text=True, check=False)
        # Combine stdout and stderr for the LLM to process, along with retcode
        output = f"Return Code: {process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        return output
    except FileNotFoundError:
        return f"Error: Command '{command_parts[0]}' not found."
    except Exception as e:
        return f"Error executing command '{command_str}': {e}"

# --- Tool Function Definitions ---
# These functions will be wrapped by FunctionTool. They perform the actual actions.

def list_files_command(path: str = ".", long_format: bool = False, show_hidden: bool = False) -> str:
    """
    Lists files in a specified directory.
    Args:
        path (str): The directory path to list files from. Defaults to current directory.
        long_format (bool): If True, uses long format listing (ls -l). Defaults to False.
        show_hidden (bool): If True, shows hidden files (ls -a). Defaults to False.
    """
    if not os.path.isdir(path): # Basic validation
        return f"Error: Directory '{path}' not found or is not accessible."
    
    command = ["ls"]
    if long_format: command.append("-l")
    if show_hidden: command.append("-a")
    command.append(path)
    return execute_local_command(command)

def show_disk_usage_command(path: str = ".") -> str:
    """
    Shows disk usage for a specified file or directory.
    Args:
        path (str): The file or directory path. Defaults to current directory.
    """
    command = ["du", "-sh", path]
    return execute_local_command(command)

def check_job_status_command(job_id: str = None, user: str = None) -> str:
    """
    Checks job status on the HPC cluster.
    IMPORTANT: Replace 'your_hpc_queue_command' with the actual command (e.g., squeue, qstat).
    Args:
        job_id (str, optional): The specific job ID to check.
        user (str, optional): The user whose jobs to check. Defaults to the current user if neither is specified.
    """
    hpc_queue_command = "squeue" # EXAMPLE: Replace with your HPC's command (squeue for Slurm, qstat for SGE/PBS)
    
    command = [hpc_queue_command]
    current_user = os.getenv("USER")

    if job_id:
        # Adapt flags for your specific command, e.g., squeue -j <job_id>, qstat -j <job_id>
        if hpc_queue_command == "squeue": command.extend(["-j", job_id])
        elif hpc_queue_command == "qstat": command.extend(["-j", job_id]) # SGE/UGE
        else: return f"Job status command ({hpc_queue_command}) flags for job_id not configured in agent."
    elif user:
        if hpc_queue_command == "squeue": command.extend(["-u", user])
        elif hpc_queue_command == "qstat": command.extend(["-u", user]) # SGE/UGE
        else: return f"Job status command ({hpc_queue_command}) flags for user not configured in agent."
    elif current_user : # Default to current user if no specific user or job_id given
         if hpc_queue_command == "squeue": command.extend(["-u", current_user])
         elif hpc_queue_command == "qstat": command.extend(["-u", current_user]) # SGE/UGE
         else: return f"Job status command ({hpc_queue_command}) flags for current user not configured."
    else: # General queue status if no user context can be determined (less common for this tool)
        if hpc_queue_command == "qstat": pass # qstat without args might show all
        elif hpc_queue_command == "squeue": pass # squeue without args might show all
        else: return f"General job status for {hpc_queue_command} not configured."

    if command == [hpc_queue_command] and hpc_queue_command == "your_hpc_queue_command": # Unconfigured
         return "Job status tool needs 'your_hpc_queue_command' replaced with your HPC's specific queue command."
        
    return execute_local_command(command)

def query_hoffman2_documentation_command(query_text: str) -> str:
    """
    Queries the Hoffman2 HPC documentation for information. Use this for 'how-to' questions,
    understanding policies, or finding specific documentation.
    Args:
        query_text (str): The question or topic to search for in the documentation.
    """
    if not rag_query_engine:
        return "Documentation (RAG) engine is not initialized. Cannot search documentation."
    try:
        response = rag_query_engine.query(query_text)
        return str(response) # AgentRunner expects string output from tools
    except Exception as e:
        return f"Error querying documentation: {e}"

# --- CLI Setup ---
@click.group()
@click.option('--index-path', default=DEFAULT_INDEX_PATH, help=f"Path to LlamaIndex RAG index. Default: {DEFAULT_INDEX_PATH}")
@click.pass_context
def cli(ctx, index_path):
    """HPC AI Agent CLI, powered by LlamaIndex."""
    ctx.ensure_object(dict)
    ctx.obj['index_path'] = index_path
    
    api_key = load_api_key_from_config()
    if api_key:
        initialize_llama_index_settings(api_key)
        # RAG engine initialized on demand by the tool or if a command needs it directly
    elif ctx.invoked_subcommand != 'configure':
        msg = "[bold yellow]Warning: OpenAI API key not configured. Run 'hpc-agent configure'. Some features may not work.[/bold yellow]"
        if console: console.print(msg)
        else: print(msg[11:-12])

@cli.command()
def configure():
    """Configure the OpenAI API key."""
    api_key_val = click.prompt("Please enter your OpenAI API key", hide_input=True)
    save_api_key_to_config(api_key_val)
    initialize_llama_index_settings(api_key_val) # Initialize LlamaIndex settings after saving

@cli.command()
@click.argument('query_text', nargs=-1, required=True)
@click.pass_context
def ask(ctx, query_text):
    """Ask the HPC AI agent a question or give it a command."""
    full_query = " ".join(query_text)
    
    if not _api_key_global: # Check if API key is loaded
        msg = "[bold red]Agent not initialized. OpenAI API key missing. Please run 'hpc-agent configure'.[/bold red]"
        if console: console.print(msg)
        else: print(msg[11:-12])
        return

    # Initialize RAG engine here if not already done and path is valid
    # It's better to initialize it once if it's going to be used as a tool.
    if not rag_query_engine and ctx.obj.get('index_path'):
        initialize_rag_engine(ctx.obj['index_path'])

    # Define tools for the LlamaIndex Agent
    tools = [
        FunctionTool.from_defaults(
            fn=list_files_command,
            name="list_files",
            description="Lists files and directories in a specified path. Useful for exploring the file system."
        ),
        FunctionTool.from_defaults(
            fn=show_disk_usage_command,
            name="show_disk_usage",
            description="Shows the disk space usage for a given file or directory path."
        ),
        FunctionTool.from_defaults(
            fn=check_job_status_command,
            name="check_job_status",
            description="Checks the status of jobs on the HPC cluster using the system's queue command. Can check a specific job ID or a user's jobs."
        ),
        FunctionTool.from_defaults(
            fn=query_hoffman2_documentation_command,
            name="query_hoffman2_documentation",
            description="Retrieves information from the Hoffman2 HPC cluster's documentation. Use for questions about policies, software, 'how-to' guides, etc."
        )
    ]
    
    # LLM for the Agent (uses LlamaIndex Settings by default if not specified here)
    # The LlamaOpenAI from Settings will be used if llm param is not passed to AgentRunner
    # Ensure this LLM supports tool calling (e.g., gpt-3.5-turbo, gpt-4, gpt-4o)
    llm = LlamaOpenAI(model="gpt-4o", api_key=_api_key_global) # Explicitly use the right model

    system_prompt = (
        "You are an AI assistant for the Hoffman2 HPC cluster. Your goal is to help users by answering their questions "
        "and executing commands on their behalf when appropriate and confirmed by them. "
        "When a user asks for an action (like listing files, checking disk usage, or job status), use the available tools. "
        "If a question seems like it can be answered from documentation (e.g., 'how do I use X?', 'what is the policy for Y?'), "
        "use the 'query_hoffman2_documentation' tool. For other general queries, you can answer directly. "
        "Always be helpful and ensure commands are confirmed by the user via the tool's built-in confirmation step."
    )

    agent = AgentRunner.from_llm(
        tools=tools,
        llm=llm, # Uses the LLM specified here, which supports tool calling
        verbose=True, # Set to False for less console noise in production
        system_prompt=system_prompt
    )
    # Alternative: ReActAgent if you prefer that style, though AgentRunner with OpenAI function calling is often more direct
    # agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, system_prompt=system_prompt)


    if console:
        console.print(f"[dim]You asked:[/dim] {full_query}")
        with console.status("[bold green]Agent is thinking...", spinner="dots"):
            try:
                response = agent.chat(full_query)
            except Exception as e:
                console.print(f"[bold red]Error during agent execution: {e}[/bold red]")
                # Provide more debug info if verbose
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                return
        console.print(Markdown(str(response)))
    else:
        print(f"You asked: {full_query}\nAgent is thinking...")
        try:
            response = agent.chat(full_query)
        except Exception as e:
            print(f"Error during agent execution: {e}")
            import traceback
            print(traceback.format_exc())
            return
        print("\nAgent Response:")
        print(str(response))

if __name__ == '__main__':
    cli()