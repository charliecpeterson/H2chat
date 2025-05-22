#!/usr/bin/env python3
"""
Hoffman2 Interactive Agent

This tool provides an interactive agent that can help users navigate and use the Hoffman2 cluster
by analyzing their environment, job files, and providing context-aware assistance.

Usage:
  python hoffman2_agent.py --index-path PATH_TO_INDEX

Author: OARC-HPC Team @ UCLA
"""

import os
import sys
import time
import subprocess
import glob
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

def load_environment():
    """Load environment variables and check for API key"""
    # Load variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    
    # Create embedding and LLM instances
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o")
    
    # Set up settings globally
    Settings.embed_model = embed_model
    Settings.llm = llm

def load_hoffman2_index(index_path="./hoffman2_index"):
    """Load the Hoffman2 documentation index from disk"""
    print(f"Loading Hoffman2 documentation index from {index_path}...")
    
    # Check if the index exists
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}.")
        print("Please run createdb.py first to build the documentation index.")
        sys.exit(1)
    
    # Simple loading without any patching
    try:
        # Load the storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        # Create a query engine with customized parameters
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        
        return query_engine
    except Exception as e:
        print(f"Error loading index: {e}")
        print("The index files may be corrupted or using an incompatible encoding.")
        print("Consider rebuilding the index by running createdb.py again.")
        sys.exit(1)

class Hoffman2Agent:
    """An agent that can interact with the Hoffman2 cluster environment."""
    
    def __init__(self, query_engine):
        """Initialize agent with access to the knowledge base."""
        self.query_engine = query_engine
        self.user_home = os.path.expanduser("~")
        self.username = os.path.basename(self.user_home)
        self.is_on_hoffman2 = self._check_if_on_hoffman2()
        
    def _check_if_on_hoffman2(self):
        """Determine if we're running on Hoffman2 cluster using FQDN."""
        # Primary method: Check hostname -A output for hoffman2.idre.ucla.edu
        try:
            fqdn = subprocess.check_output(["hostname", "-A"], universal_newlines=True).strip()
            if "hoffman2.idre.ucla.edu" in fqdn:
                return True
        except Exception:
            # Fall back to other methods if hostname -A fails
            pass
            
        # Method 2: Check hostname pattern (for older nodes or misconfigured systems)
        try:
            hostname = subprocess.check_output(["hostname"], universal_newlines=True).strip()
            if any(pattern in hostname.lower() for pattern in ["hoffman2", "h2-", "login", "n10", "n11", "n12"]):
                return True
        except Exception:
            pass
            
        # Method 3: Check for SGE environment variables (specific to Hoffman2's job scheduler)
        if os.environ.get("SGE_ROOT") or os.environ.get("SGE_CELL"):
            return True
            
        # Method 4: Try SGE commands
        try:
            result = subprocess.run("qstat -help", shell=True, stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, timeout=2)
            if result.returncode == 0:
                return True
        except Exception:
            pass
            
        # Method 5: Look for specific Hoffman2 directories
        if os.path.exists("/u/systems/hoffman2") or os.path.exists("/u/local/hoffman2"):
            return True
                
        return False
                
    def run_command(self, command):
        """Run a shell command and return its output."""
        try:
            result = subprocess.run(command, shell=True, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout if hasattr(e, 'stdout') else "",
                "stderr": e.stderr if hasattr(e, 'stderr') else "",
                "command": command,
                "error": str(e)
            }
            
    def get_environment_info(self):
        """Collect information about the user's environment."""
        if not self.is_on_hoffman2:
            return {"error": "Not running on Hoffman2 cluster"}
            
        info = {
            "username": self.username,
            "current_directory": os.getcwd(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Get loaded modules
        modules_result = self.run_command("module list 2>&1")
        if modules_result["success"]:
            info["loaded_modules"] = modules_result["stdout"].strip()
        
        # Get disk usage
        disk_result = self.run_command(f"du -sh {self.user_home} 2>/dev/null")
        if disk_result["success"]:
            info["home_disk_usage"] = disk_result["stdout"].strip()
            
        # Get quota information
        quota_result = self.run_command("lfs quota -h -u $USER /u")
        if quota_result["success"]:
            info["quota"] = quota_result["stdout"].strip()
            
        # Get job status
        jobs_result = self.run_command("qstat -u $USER")
        if jobs_result["success"]:
            info["current_jobs"] = jobs_result["stdout"].strip()
            
        return info
        
    def analyze_error_file(self, file_path):
        """Analyze a job error file for common issues."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                
            # Extract key error patterns
            analysis = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "content": content[:2000] if len(content) > 2000 else content,  # First 2000 chars only
            }
            
            # Look for common error patterns
            error_patterns = {
                "out_of_memory": r"Out of memory|MemoryError|Killed|memory allocation failed",
                "permission_denied": r"Permission denied",
                "file_not_found": r"No such file or directory",
                "job_exceeded_time": r"exceeded resource usage|DUE TO TIME LIMIT",
                "module_error": r"module: command not found|failed to load module"
            }
            
            analysis["detected_errors"] = {}
            for error_type, pattern in error_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["detected_errors"][error_type] = len(matches)
            
            return analysis
            
        except Exception as e:
            return {"error": str(e), "file_path": file_path}
            
    def find_recent_jobs(self, days=1, status="all"):
        """Find recent job files based on status and time period."""
        if not self.is_on_hoffman2:
            return {"error": "Not running on Hoffman2 cluster"}
            
        # Get recent job IDs
        cmd = f"qacct -j -o {self.username} -d {days}"
        if status.lower() != "all":
            cmd += f" -s {status}"
            
        result = self.run_command(cmd)
        if not result["success"]:
            return {"error": "Failed to retrieve job history", "details": result}
            
        # Parse job IDs
        job_ids = []
        for line in result["stdout"].splitlines():
            if line.startswith("jobnumber"):
                parts = line.split()
                if len(parts) > 1:
                    job_ids.append(parts[1])
                    
        # Find job output files
        job_files = []
        for job_id in job_ids:
            # Look for output and error files with this job ID
            output_files = glob.glob(f"{self.user_home}/*o{job_id}*")
            error_files = glob.glob(f"{self.user_home}/*e{job_id}*")
            
            job_files.append({
                "job_id": job_id,
                "output_files": output_files,
                "error_files": error_files
            })
            
        return {
            "days": days,
            "status": status,
            "job_count": len(job_ids),
            "jobs": job_files
        }
        
    def suggest_job_improvements(self, job_script_path):
        """Analyze a job script and suggest improvements."""
        try:
            with open(job_script_path, 'r') as file:
                script_content = file.read()
                
            suggestions = []
            
            # Check for memory settings
            if not re.search(r'-l\s+h_data=', script_content):
                suggestions.append("Consider specifying memory requirements with -l h_data=SIZE")
                
            # Check for runtime limits
            if not re.search(r'-l\s+h_rt=', script_content):
                suggestions.append("Consider setting a runtime limit with -l h_rt=HH:MM:SS")
                
            # Check for output path configurations
            if not re.search(r'-o\s+', script_content) and not re.search(r'-e\s+', script_content):
                suggestions.append("Consider setting output and error file paths with -o and -e")
                
            # Check for email notifications
            if not re.search(r'-m\s+', script_content):
                suggestions.append("Consider enabling email notifications with -m bea")
                
            # Check for proper shebang
            if not script_content.startswith("#!/bin/bash"):
                suggestions.append("Make sure your script starts with #!/bin/bash")
                
            # Use LlamaIndex knowledge base for deeper analysis
            prompt = f"""
            Please analyze this job script for potential issues or improvements:
            
            ```bash
            {script_content[:1500]}  # Limit to first 1500 chars
            ```
            
            Focus on:
            1. Resource specifications (CPU, memory, time)
            2. Error handling
            3. Efficiency considerations
            4. Best practices for Hoffman2
            """
            
            kb_response = self.query_engine.query(prompt)
            
            return {
                "script_path": job_script_path,
                "automatic_suggestions": suggestions,
                "knowledge_base_analysis": kb_response.response
            }
            
        except Exception as e:
            return {"error": str(e), "script_path": job_script_path}

    def process_module_info(self):
        """Gather and process module information"""
        if not self.is_on_hoffman2:
            return {"error": "Not running on Hoffman2 cluster"}
            
        # Get available modules
        result = self.run_command("module avail 2>&1")
        if not result["success"]:
            return {"error": "Failed to retrieve module information"}
            
        # Process module categories
        modules_by_category = {}
        current_category = "Other"
        
        for line in result["stdout"].splitlines():
            # Check if this is a category line
            if line.strip().endswith(':'):
                current_category = line.strip().rstrip(':').strip()
                modules_by_category[current_category] = []
            elif line.strip() and not line.startswith('-'):
                # Add module to current category
                modules = line.strip().split()
                modules_by_category[current_category].extend(modules)
                
        return modules_by_category

    def find_similar_modules(self, module_name):
        """Find similar modules to the requested one"""
        modules = self.process_module_info()
        
        if "error" in modules:
            return modules
            
        # Flatten all modules into one list
        all_modules = []
        for category, module_list in modules.items():
            all_modules.extend(module_list)
            
        # Find similar modules using string matching
        similar_modules = []
        for module in all_modules:
            # Skip modules/versions
            if '/' not in module:
                continue
                
            base_name = module.split('/')[0]
            if module_name.lower() in base_name.lower():
                similar_modules.append(module)
                
        return {
            "query": module_name,
            "found_count": len(similar_modules),
            "similar_modules": similar_modules[:10]  # Limit to top 10
        }
        
    def find_files(self, pattern, directory=None, max_depth=3, max_files=50):
        """Find files matching a pattern."""
        if not self.is_on_hoffman2:
            return {"error": "Not running on Hoffman2 cluster"}
            
        # Use the current directory if none provided
        if directory is None:
            directory = os.getcwd()
            
        try:
            # Use find command with depth limit for better performance
            cmd = f"find {directory} -maxdepth {max_depth} -name '{pattern}' -type f | head -n {max_files}"
            result = self.run_command(cmd)
            
            if not result["success"]:
                return {"error": "Failed to search for files", "details": result}
                
            files = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
            
            return {
                "pattern": pattern,
                "directory": directory,
                "file_count": len(files),
                "files": files,
                "limited": len(files) >= max_files  # Flag if results were limited
            }
        except Exception as e:
            return {"error": str(e)}
            
    def view_file(self, file_path, line_count=None, head=None, tail=None):
        """View contents of a file with options for head/tail/specific lines."""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
                
            # Get file info
            file_info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Handle different view options
            if head is not None:
                cmd = f"head -n {head} {file_path}"
                result = self.run_command(cmd)
                if result["success"]:
                    file_info["content"] = result["stdout"]
                    file_info["view_type"] = f"first {head} lines"
                else:
                    return {"error": f"Failed to read file: {result['stderr']}"}
                    
            elif tail is not None:
                cmd = f"tail -n {tail} {file_path}"
                result = self.run_command(cmd)
                if result["success"]:
                    file_info["content"] = result["stdout"]
                    file_info["view_type"] = f"last {tail} lines"
                else:
                    return {"error": f"Failed to read file: {result['stderr']}"}
                    
            elif line_count is not None:
                # Parse line range like "5-10" or "15-20"
                try:
                    start, end = map(int, line_count.split('-'))
                    cmd = f"sed -n '{start},{end}p' {file_path}"
                    result = self.run_command(cmd)
                    if result["success"]:
                        file_info["content"] = result["stdout"]
                        file_info["view_type"] = f"lines {start}-{end}"
                    else:
                        return {"error": f"Failed to read file: {result['stderr']}"}
                except ValueError:
                    return {"error": "Invalid line range format. Use 'start-end' (e.g., '5-10')"}
                    
            else:
                # Default: check file size first to avoid loading huge files
                if file_info["size"] > 50000:  # 50KB limit for full view
                    file_info["content"] = f"File is too large ({file_info['size']} bytes) to display in full. Use !view file --head 20 or !view file --tail 20 to see portions."
                    file_info["view_type"] = "size warning"
                else:
                    with open(file_path, 'r') as f:
                        file_info["content"] = f.read()
                    file_info["view_type"] = "full content"
            
            return file_info
            
        except Exception as e:
            return {"error": str(e)}
            
    def grep_file(self, pattern, file_path=None, directory=None, recursive=False):
        """Search for pattern in files."""
        if not self.is_on_hoffman2:
            return {"error": "Not running on Hoffman2 cluster"}
            
        try:
            grep_cmd = "grep --color=never -n"  # Line numbers, no color codes
            
            if recursive:
                grep_cmd += " -r"
                
            if directory is None and file_path is None:
                directory = os.getcwd()  # Default to current directory
                
            # Build the grep command
            if file_path:
                if os.path.exists(file_path):
                    cmd = f"{grep_cmd} '{pattern}' {file_path}"
                else:
                    return {"error": f"File not found: {file_path}"}
            else:
                if os.path.exists(directory):
                    cmd = f"{grep_cmd} '{pattern}' {directory}" + ("/*" if not recursive else "")
                else:
                    return {"error": f"Directory not found: {directory}"}
                    
            result = self.run_command(cmd)
            matches = []
            
            if result["success"]:
                for line in result["stdout"].splitlines():
                    if line.strip():
                        matches.append(line)
                        
                return {
                    "pattern": pattern,
                    "location": file_path or directory,
                    "matches": matches,
                    "match_count": len(matches)
                }
            else:
                # grep returns non-zero exit code if no matches found (not an error)
                if "No such file or directory" in result["stderr"]:
                    return {"error": "File or directory not found"}
                return {
                    "pattern": pattern,
                    "location": file_path or directory,
                    "matches": [],
                    "match_count": 0
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    def smart_file_analyze(self, file_path):
        """Intelligently analyze an unknown file type to extract useful information."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        try:
            # Get basic file info
            file_info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "basename": os.path.basename(file_path)
            }
            
            # Try to determine file type
            file_type_cmd = f"file {file_path}"
            file_type_result = self.run_command(file_type_cmd)
            if file_type_result["success"]:
                file_info["detected_type"] = file_type_result["stdout"].split(":", 1)[1].strip()
                
            # Read a sample of file content
            try:
                with open(file_path, 'r', errors='replace') as f:
                    sample = f.read(4000)  # First 4KB for analysis
                    file_info["content_sample"] = sample
            except Exception:
                file_info["content_sample"] = "(Unable to read file as text)"
                
            # For files that look like error logs or output files, look for common patterns
            if "error" in file_path.lower() or "output" in file_path.lower() or ".err" in file_path.lower() or ".out" in file_path.lower():
                # Use the existing error pattern detection
                analysis = self.analyze_error_file(file_path)
                if "detected_errors" in analysis:
                    file_info["detected_errors"] = analysis["detected_errors"]
                    
            # Use the query engine to extract useful insights
            prompt = f"""
            Please analyze this file content and provide useful insights:
            
            Filename: {file_info['basename']}
            Detected type: {file_info.get('detected_type', 'Unknown')}
            File size: {file_info['size']} bytes
            
            Sample content:
            ```
            {file_info.get('content_sample', '(No content available)')}
            ```
            
            If this appears to be an error log, what errors do you see and what might have caused them?
            If this is a configuration file, what is it configuring and are there any issues?
            If this is a script or code, what is it doing and are there any potential issues?
            If this is output from a program or job, what was the program trying to do and did it succeed?
            
            Provide a brief, practical analysis focusing on problems and solutions.
            """
            
            try:
                analysis = self.query_engine.query(prompt)
                file_info["ai_analysis"] = analysis.response
            except Exception as e:
                file_info["ai_analysis_error"] = str(e)
                
            return file_info
            
        except Exception as e:
            return {"error": str(e)}

def agent_interface(query_engine):
    """Run an interactive session with the Hoffman2 agent."""
    agent = Hoffman2Agent(query_engine)
    
    print("\n=== Hoffman2 Agent ===")
    if agent.is_on_hoffman2:
        print(f"Running on Hoffman2 as user: {agent.username}")
        info = agent.get_environment_info()
        print(f"Current directory: {info['current_directory']}")
        if 'loaded_modules' in info:
            print(f"Loaded modules: {info['loaded_modules']}")
    else:
        print("Not running on Hoffman2 cluster. Some features will be limited.")
    
    print("\nType your questions or use these special commands:")
    print("  !env           - Show my Hoffman2 environment details")
    print("  !jobs          - Show my current jobs")
    print("  !analyze FILE  - Analyze an error or output file")
    print("  !recent [N]    - Show my N most recent jobs (default: 5)")
    print("  !improve FILE  - Suggest improvements for a job script")
    print("  !module NAME   - Find modules similar to NAME")
    print("  !find PATTERN  - Find files matching a pattern")
    print("  !view FILE     - View contents of a file")
    print("  !grep PATTERN  - Search for pattern in files")
    print("  !smart FILE    - Perform smart analysis on a file")
    print("  !help          - Show this help message")
    print("  !exit          - Exit the agent")
    
    # Keep conversation context
    conversation_context = []
    
    while True:
        query = input("\nWhat can I help with? ").strip()
        
        if not query:
            continue
            
        if query.lower() == "!exit":
            print("Thank you for using the Hoffman2 Agent. Goodbye!")
            break
            
        if query.lower() == "!help":
            print("\nAvailable commands:")
            print("  !env           - Show my Hoffman2 environment details")
            print("  !jobs          - Show my current jobs")
            print("  !analyze FILE  - Analyze an error or output file")
            print("  !recent [N]    - Show my N most recent jobs (default: 5)")
            print("  !improve FILE  - Suggest improvements for a job script")
            print("  !module NAME   - Find modules similar to NAME")
            print("  !find PATTERN  - Find files matching a pattern")
            print("  !view FILE     - View contents of a file (options: --head N, --tail N, --lines N-M)")
            print("  !grep PATTERN [FILE|DIR] - Search for pattern in files (option: --recursive)")
            print("  !smart FILE    - Perform smart analysis on a file")
            print("  !help          - Show this help message")
            print("  !exit          - Exit the agent")
            continue
            
        if query.lower() == "!env":
            info = agent.get_environment_info()
            print("\n=== Your Hoffman2 Environment ===")
            for key, value in info.items():
                if key != "current_jobs":  # Print jobs separately
                    print(f"{key}: {value}")
            continue
            
        if query.lower() == "!jobs":
            info = agent.get_environment_info()
            if "current_jobs" in info and info["current_jobs"]:
                print("\n=== Your Current Jobs ===")
                print(info["current_jobs"])
            else:
                print("No active jobs found.")
            continue
            
        if query.lower().startswith("!analyze "):
            file_path = query[9:].strip()
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            print(f"\nAnalyzing file: {file_path}...")
            analysis = agent.analyze_error_file(file_path)
            
            print("\n=== File Analysis ===")
            print(f"File: {analysis['file_path']}")
            print(f"Size: {analysis['file_size']} bytes")
            print(f"Last modified: {analysis['last_modified']}")
            
            if "detected_errors" in analysis and analysis["detected_errors"]:
                print("\nDetected error types:")
                for error_type, count in analysis["detected_errors"].items():
                    print(f"  - {error_type}: {count} occurrences")
                    
                # Use query engine to explain how to fix these errors
                error_types = list(analysis["detected_errors"].keys())
                explanation = query_engine.query(
                    f"How do I fix these errors on Hoffman2: {', '.join(error_types)}? "
                    f"Please provide concrete steps for each error type."
                )
                print("\nSuggested solutions:")
                print(explanation.response)
            else:
                print("\nNo common errors detected in this file.")
                
            continue
            
        if query.lower().startswith("!recent"):
            parts = query.split()
            num_jobs = 5  # Default
            if len(parts) > 1 and parts[1].isdigit():
                num_jobs = int(parts[1])
                
            print(f"\nFinding your {num_jobs} most recent jobs...")
            recent_jobs = agent.find_recent_jobs(days=7)  # Last week
            
            if "error" in recent_jobs:
                print(f"Error: {recent_jobs['error']}")
            elif recent_jobs["job_count"] == 0:
                print("No recent jobs found.")
            else:
                print(f"\nFound {recent_jobs['job_count']} jobs in the last 7 days")
                for i, job in enumerate(recent_jobs["jobs"][:num_jobs]):
                    print(f"\nJob {i+1}: {job['job_id']}")
                    if job["output_files"]:
                        print(f"  Output files: {', '.join(job['output_files'])}")
                    if job["error_files"]:
                        print(f"  Error files: {', '.join(job['error_files'])}")
            continue
            
        if query.lower().startswith("!improve "):
            script_path = query[9:].strip()
            if not os.path.exists(script_path):
                print(f"Script not found: {script_path}")
                continue
                
            print(f"\nAnalyzing job script: {script_path}...")
            suggestions = agent.suggest_job_improvements(script_path)
            
            if "error" in suggestions:
                print(f"Error: {suggestions['error']}")
            else:
                print("\n=== Job Script Analysis ===")
                if suggestions["automatic_suggestions"]:
                    print("\nAutomatic suggestions:")
                    for i, suggestion in enumerate(suggestions["automatic_suggestions"]):
                        print(f"  {i+1}. {suggestion}")
                        
                print("\nDetailed analysis:")
                print(suggestions["knowledge_base_analysis"])
            continue
            
        if query.lower().startswith("!module "):
            module_name = query[8:].strip()
            if not module_name:
                print("Please specify a module name to search for")
                continue
                
            print(f"\nSearching for modules similar to '{module_name}'...")
            modules = agent.find_similar_modules(module_name)
            
            if "error" in modules:
                print(f"Error: {modules['error']}")
            elif modules["found_count"] == 0:
                print(f"No modules found matching '{module_name}'")
            else:
                print(f"\nFound {modules['found_count']} modules matching '{module_name}':")
                for module in modules["similar_modules"]:
                    print(f"  - {module}")
                    
                if modules["found_count"] > 10:
                    print(f"  (showing 10 of {modules['found_count']} matches)")
            continue
            
        # Add the new command handlers
        if query.lower().startswith("!find "):
            pattern = query[6:].strip()
            if not pattern:
                print("Please provide a search pattern")
                continue
                
            print(f"\nFinding files matching '{pattern}'...")
            result = agent.find_files(pattern)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            elif result["file_count"] == 0:
                print("No matching files found.")
            else:
                print(f"\nFound {result['file_count']} matching files in {result['directory']}:")
                for file_path in result["files"]:
                    print(f"  - {file_path}")
                if result.get("limited", False):
                    print("  (results limited to first 50 matches)")
            continue
            
        if query.lower().startswith("!view "):
            args = query[6:].strip().split()
            if not args:
                print("Please provide a file path to view")
                continue
                
            file_path = args[0]
            head = None
            tail = None
            lines = None
            
            # Parse additional arguments (--head, --tail, --lines)
            i = 1
            while i < len(args):
                if args[i] == "--head" and i+1 < len(args):
                    try:
                        head = int(args[i+1])
                        i += 2
                    except ValueError:
                        print(f"Invalid value for --head: {args[i+1]}")
                        break
                elif args[i] == "--tail" and i+1 < len(args):
                    try:
                        tail = int(args[i+1])
                        i += 2
                    except ValueError:
                        print(f"Invalid value for --tail: {args[i+1]}")
                        break
                elif args[i] == "--lines" and i+1 < len(args):
                    lines = args[i+1]
                    i += 2
                else:
                    i += 1
                    
            print(f"\nViewing file: {file_path}")
            result = agent.view_file(file_path, line_count=lines, head=head, tail=tail)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\n=== File: {result['path']} ===")
                print(f"Size: {result['size']} bytes")
                print(f"Modified: {result['modified']}")
                print(f"Showing: {result['view_type']}")
                print("\n--- Content ---\n")
                print(result["content"])
                print("\n--- End of Content ---")
            continue
            
        if query.lower().startswith("!grep "):
            parts = query[6:].strip().split()
            if len(parts) < 1:
                print("Please provide a search pattern")
                continue
                
            pattern = parts[0]
            file_or_dir = None
            recursive = False
            
            # Parse additional arguments
            for i in range(1, len(parts)):
                if parts[i] == "--recursive":
                    recursive = True
                elif file_or_dir is None and not parts[i].startswith("--"):
                    file_or_dir = parts[i]
                    
            print(f"\nSearching for '{pattern}'...")
            if recursive:
                print("(recursive search)")
                
            result = agent.grep_file(pattern, file_path=file_or_dir if os.path.isfile(file_or_dir) else None, 
                                  directory=file_or_dir if os.path.isdir(file_or_dir) else None,
                                  recursive=recursive)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            elif result["match_count"] == 0:
                print(f"No matches found for '{pattern}'")
            else:
                print(f"\nFound {result['match_count']} matches for '{pattern}' in {result['location']}:")
                for match in result["matches"]:
                    print(f"  {match}")
            continue
            
        if query.lower().startswith("!smart "):
            file_path = query[7:].strip()
            if not file_path:
                print("Please provide a file path to analyze")
                continue
                
            print(f"\nPerforming smart analysis on: {file_path}")
            result = agent.smart_file_analyze(file_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\n=== Smart Analysis: {result['path']} ===")
                print(f"Size: {result['size']} bytes")
                print(f"Modified: {result['modified']}")
                if "detected_type" in result:
                    print(f"Detected type: {result['detected_type']}")
                
                if "detected_errors" in result and result["detected_errors"]:
                    print("\n--- Detected Errors ---")
                    for error_type, count in result["detected_errors"].items():
                        print(f"  - {error_type}: {count} occurrences")
                    
                print("\n--- AI Analysis ---")
                if "ai_analysis" in result:
                    print(result["ai_analysis"])
                else:
                    print("(Analysis not available)")
            continue
            
        # If we get here, treat the query as a regular question
        print("\nSearching knowledge base...")
        start_time = datetime.now()
        
        # Enhance the query with environment context if on Hoffman2
        if agent.is_on_hoffman2:
            info = agent.get_environment_info()
            
            # Add simple conversation context for follow-up questions
            if len(conversation_context) > 0 and not query.endswith("?"):
                # If this looks like a follow-up question, include context
                enhanced_query = f"""
                Previous question: {conversation_context[-1]}
                
                User question: {query}
                
                User's context:
                - Username: {info['username']}
                - Current directory: {info['current_directory']}
                - Loaded modules: {info.get('loaded_modules', 'None')}
                
                Please answer the follow-up question with this context in mind.
                """
            else:
                enhanced_query = f"""
                User question: {query}
                
                User's context:
                - Username: {info['username']}
                - Current directory: {info['current_directory']}
                - Loaded modules: {info.get('loaded_modules', 'None')}
                
                Please answer the question with this context in mind.
                """
        else:
            if len(conversation_context) > 0 and not query.endswith("?"):
                # If this looks like a follow-up question, include context
                enhanced_query = f"""
                Previous question: {conversation_context[-1]}
                
                Follow-up question: {query}
                
                Please answer the follow-up question taking into account the previous context.
                """
            else:
                enhanced_query = query
                
        # Remember this question for context
        conversation_context.append(query)
        if len(conversation_context) > 3:
            # Keep only the most recent questions
            conversation_context = conversation_context[-3:]
            
        response = query_engine.query(enhanced_query)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nAnswer (found in {duration:.2f} seconds):\n")
        print(response.response)

def main():
    """Main function to run the Hoffman2 agent."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hoffman2 Interactive Agent")
    parser.add_argument("--index-path", default="./hoffman2_index", help="Path to the index directory")
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Load the Hoffman2 documentation index
    query_engine = load_hoffman2_index(args.index_path)
    
    # Start the agent interface
    agent_interface(query_engine)

if __name__ == "__main__":
    main()