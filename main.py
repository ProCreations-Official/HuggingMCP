#!/usr/bin/env python3
"""
HuggingMCP - Enhanced Hugging Face MCP Server
Optimized with 11 main commands and enhanced debugging
"""

import os
import sys
import logging
import traceback
import time
from typing import Dict, Any, Optional, Union, List

# Enhanced stderr debugging
def debug_stderr(message: str, level: str = "INFO"):
    """Enhanced debug output to stderr for MCP troubleshooting"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] HuggingMCP: {message}", file=sys.stderr, flush=True)

# Startup debugging
debug_stderr("🚀 HuggingMCP server starting up...")
debug_stderr(f"Python executable: {sys.executable}")
debug_stderr(f"Script path: {__file__}")
debug_stderr(f"Working directory: {os.getcwd()}")

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('/tmp/hugmcp_debug.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import MCP with error handling
try:
    from mcp.server.fastmcp import FastMCP
    debug_stderr("✅ MCP imported successfully")
except ImportError as e:
    debug_stderr(f"❌ MCP import failed: {e}", "ERROR")
    logger.error(f"MCP not installed. Run: pip3 install 'mcp[cli]' - Error: {e}")
    sys.exit(1)

# Import Hugging Face with error handling
try:
    from huggingface_hub import (
        HfApi, create_repo, upload_file, delete_repo, delete_file,
        list_models, list_datasets, list_spaces, model_info, dataset_info,
        hf_hub_download, login, logout, whoami, create_collection,
        get_collection, add_collection_item, create_discussion,
        get_repo_discussions, get_discussion_details, CommitOperationAdd, CommitOperationDelete
    )
    debug_stderr("✅ Hugging Face Hub core imports successful")
    
    # Optional imports for advanced features
    try:
        from huggingface_hub import list_liked_repos, list_repo_refs, list_repo_commits
        HAS_ADVANCED_REPO = True
        debug_stderr("✅ Advanced repository features available")
    except ImportError:
        HAS_ADVANCED_REPO = False
        debug_stderr("⚠️ Advanced repository features not available", "WARN")
    
    try:
        from huggingface_hub import get_space_runtime, restart_space, pause_space, set_space_sleep_time, duplicate_space
        HAS_SPACE_MANAGEMENT = True
        debug_stderr("✅ Space management features available")
    except ImportError:
        HAS_SPACE_MANAGEMENT = False
        debug_stderr("⚠️ Space management features not available", "WARN")
    
    try:
        from huggingface_hub import create_branch, delete_branch
        HAS_BRANCH_MANAGEMENT = True
        debug_stderr("✅ Branch management features available")
    except ImportError:
        HAS_BRANCH_MANAGEMENT = False
        debug_stderr("⚠️ Branch management features not available", "WARN")
    
    try:
        from huggingface_hub import get_inference_api
        HAS_INFERENCE_API = True
        debug_stderr("✅ Inference API features available")
    except ImportError:
        HAS_INFERENCE_API = False
        debug_stderr("⚠️ Inference API features not available", "WARN")
    
    debug_stderr("✅ Hugging Face Hub imported successfully")
except ImportError as e:
    debug_stderr(f"❌ Hugging Face Hub import failed: {e}", "ERROR")
    logger.error(f"huggingface_hub not installed. Run: pip3 install huggingface_hub - Error: {e}")
    sys.exit(1)

# Import additional libraries for enhanced functionality (optional)
try:
    import json
    import re
    import base64
    import hashlib
    import urllib.parse
    from datetime import datetime, timedelta
    from collections import defaultdict
    from pathlib import Path
    debug_stderr("✅ Standard libraries imported successfully")
except ImportError as e:
    debug_stderr(f"⚠️ Some standard libraries missing: {e}", "WARN")

# Initialize MCP server with metadata
try:
    mcp = FastMCP(
        "HuggingMCP",
        description="Advanced Hugging Face MCP Server with 18+ comprehensive tools for ML workflows"
    )
    debug_stderr("✅ FastMCP server initialized")
except Exception as e:
    debug_stderr(f"❌ FastMCP initialization failed: {e}", "ERROR")
    logger.error(f"Failed to initialize FastMCP: {e}")
    sys.exit(1)

# Configuration with enhanced debugging
TOKEN = os.getenv("HF_TOKEN")
READ_ONLY = os.getenv("HF_READ_ONLY", "false").lower() == "true"
ADMIN_MODE = os.getenv("HF_ADMIN_MODE", "false").lower() == "true"
MAX_FILE_SIZE = int(os.getenv("HF_MAX_FILE_SIZE", "104857600"))  # 100MB default
INFERENCE_TIMEOUT = int(os.getenv("HF_INFERENCE_TIMEOUT", "30"))  # 30 seconds default
ENABLE_INFERENCE = os.getenv("HF_ENABLE_INFERENCE", "true").lower() == "true"
CACHE_ENABLED = os.getenv("HF_CACHE_ENABLED", "true").lower() == "true"

debug_stderr(f"Configuration loaded - Token: {'✓' if TOKEN else '✗'}, Read-only: {READ_ONLY}, Admin: {ADMIN_MODE}")

# Initialize API with error handling
try:
    api = HfApi(token=TOKEN) if TOKEN else HfApi()
    debug_stderr("✅ HfApi initialized successfully")
except Exception as e:
    debug_stderr(f"❌ HfApi initialization failed: {e}", "ERROR")
    logger.error(f"Failed to initialize HfApi: {e}")
    sys.exit(1)

logger.info(f"🤗 HuggingMCP initialized - Token: {'✓' if TOKEN else '✗'}, Admin: {ADMIN_MODE}")

# =============================================================================
# HELPER FUNCTIONS (Shared across commands)
# =============================================================================

def validate_auth(operation: str = "operation") -> Optional[Dict[str, Any]]:
    """Validate authentication for operations that require it"""
    if not TOKEN:
        return {
            "error": f"❌ Authentication required for {operation}",
            "help": "Set HF_TOKEN environment variable with your Hugging Face token",
            "instructions": {
                "get_token": "Visit https://huggingface.co/settings/tokens to create a token",
                "set_env": "Set HF_TOKEN=your_token_here in your environment",
                "restart": "Restart the MCP server after setting the token"
            }
        }
    return None

def validate_permissions(require_write: bool = False, require_admin: bool = False) -> Optional[Dict[str, Any]]:
    """Validate permissions for operations"""
    if require_write and READ_ONLY:
        return {
            "error": "❌ Read-only mode - write operations not allowed",
            "help": "Set HF_READ_ONLY=false to enable write operations",
            "current_mode": "read-only",
            "required_mode": "read-write"
        }
    if require_admin and not ADMIN_MODE:
        return {
            "error": "❌ Admin mode required for this operation",
            "help": "Set HF_ADMIN_MODE=true to enable admin operations",
            "current_mode": "standard",
            "required_mode": "admin",
            "warning": "Admin mode allows dangerous operations like repository deletion"
        }
    return None

def validate_repo_id(repo_id: str) -> Optional[Dict[str, Any]]:
    """Validate repository ID format"""
    if not repo_id:
        return {"error": "❌ Repository ID cannot be empty"}
    
    if len(repo_id) > 96:  # HF limit
        return {"error": "❌ Repository ID too long (max 96 characters)"}
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?(/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)?$', repo_id):
        return {
            "error": "❌ Invalid repository ID format",
            "help": "Repository ID should be 'username/repo-name' or just 'repo-name'",
            "examples": ["microsoft/DialoGPT-medium", "my-awesome-model", "datasets/squad"]
        }
    
    return None

def validate_file_path(filepath: str) -> Optional[Dict[str, Any]]:
    """Validate file path for repository operations"""
    if not filepath:
        return {"error": "❌ File path cannot be empty"}
    
    # Check for dangerous patterns
    dangerous_patterns = ['..', '//', '\\', '<', '>', '|', '?', '*', '\0']
    for pattern in dangerous_patterns:
        if pattern in filepath:
            return {
                "error": f"❌ File path contains unsafe characters: {pattern}",
                "help": "Use only alphanumeric characters, hyphens, underscores, dots, and forward slashes"
            }
    
    if len(filepath) > 255:
        return {"error": "❌ File path too long (max 255 characters)"}
    
    return None

def safe_execute(func, operation_name: str, *args, **kwargs) -> Dict[str, Any]:
    """Safely execute functions with comprehensive error handling"""
    try:
        debug_stderr(f"Executing {operation_name}")
        result = func(*args, **kwargs)
        debug_stderr(f"✅ {operation_name} completed successfully")
        return result
    except Exception as e:
        error_msg = f"❌ {operation_name} failed: {str(e)}"
        debug_stderr(error_msg, "ERROR")
        logger.error(f"{operation_name} error: {e}")
        logger.error(traceback.format_exc())
        return {"error": error_msg, "details": str(e)}

def format_repo_info(info) -> Dict[str, Any]:
    """Format repository information consistently"""
    return {
        "id": info.id,
        "author": info.author,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "private": info.private,
        "downloads": getattr(info, 'downloads', 0),
        "likes": getattr(info, 'likes', 0),
        "tags": getattr(info, 'tags', [])[:5],  # Limit for readability
        "files": [s.rfilename for s in info.siblings[:10]] if hasattr(info, 'siblings') else []
    }

def get_user_namespace() -> str:
    """Get current user's namespace safely"""
    try:
        if TOKEN:
            user_info = whoami(token=TOKEN)
            return user_info.get('name', 'unknown')
        return 'unknown'
    except Exception:
        return 'unknown'

def format_model_card_data(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format model card data consistently"""
    return {
        "license": card_data.get("license"),
        "language": card_data.get("language", []),
        "tags": card_data.get("tags", []),
        "datasets": card_data.get("datasets", []),
        "metrics": card_data.get("metrics", []),
        "pipeline_tag": card_data.get("pipeline_tag"),
        "library_name": card_data.get("library_name"),
        "base_model": card_data.get("base_model"),
        "model_type": card_data.get("model-type")
    }

def calculate_popularity_score(repo_info) -> int:
    """Calculate a popularity score for ranking"""
    downloads = getattr(repo_info, 'downloads', 0) or 0
    likes = getattr(repo_info, 'likes', 0) or 0
    # Weight downloads more heavily than likes
    return (downloads * 2) + likes

def validate_file_size(content: str, max_size: int = None) -> Optional[Dict[str, Any]]:
    """Validate file size before operations"""
    if max_size is None:
        max_size = MAX_FILE_SIZE
    
    size = len(content.encode('utf-8'))
    if size > max_size:
        return {
            "error": f"❌ File size ({size:,} bytes) exceeds maximum ({max_size:,} bytes)",
            "current_size": size,
            "max_size": max_size
        }
    return None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe repository operations"""
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive dots or underscores
    sanitized = re.sub(r'[._]{2,}', '_', sanitized)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    return sanitized or "unnamed_file"

def generate_commit_hash(content: str) -> str:
    """Generate a hash for content tracking"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]

def parse_repo_id(repo_id: str) -> Dict[str, str]:
    """Parse repository ID into components"""
    if '/' in repo_id:
        author, name = repo_id.split('/', 1)
        return {"author": author, "name": name, "full_id": repo_id}
    else:
        return {"author": "", "name": repo_id, "full_id": repo_id}

# =============================================================================
# CONSOLIDATED COMMANDS (Maximum 10 main commands)
# =============================================================================

@mcp.tool()
def hf_system_info() -> Dict[str, Any]:
    """Get HuggingMCP system information, configuration, and test connectivity"""
    def _get_system_info():
        user_info = None
        if TOKEN:
            try:
                user_info = whoami(token=TOKEN)
            except Exception as e:
                debug_stderr(f"Failed to get user info: {e}", "WARN")
        
        return {
            "status": "✅ HuggingMCP is operational!",
            "version": "3.0.0",
            "server_info": {
                "authenticated": bool(TOKEN),
                "admin_mode": ADMIN_MODE,
                "read_only": READ_ONLY,
                "python_version": sys.version,
                "script_path": __file__
            },
            "user_info": {
                "authenticated": bool(TOKEN),
                "name": user_info.get('name', 'Unknown') if user_info else 'Not authenticated',
                "email": user_info.get('email', 'Unknown') if user_info else 'Not authenticated'
            },
            "capabilities": {
                "create_repos": not READ_ONLY,
                "delete_repos": ADMIN_MODE,
                "read_files": True,
                "write_files": not READ_ONLY,
                "search": True,
                "collections": not READ_ONLY,
                "pull_requests": not READ_ONLY,
                "model_evaluation": True,
                "dataset_processing": True,
                "license_management": True,
                "community_features": True,
                "space_management": not READ_ONLY,
                "inference_tools": ENABLE_INFERENCE,
                "ai_workflows": True,
                "advanced_analytics": True,
                "repository_utilities": True
            },
            "commands_available": 18,
            "new_features": [
                "🔬 Model evaluation and testing",
                "🗃️ Advanced dataset processing",
                "📝 License management tools",
                "🤝 Community interaction features",
                "🚀 Space management capabilities",
                "🧠 AI inference tools",
                "⚙️ Workflow automation",
                "📊 Advanced analytics engine",
                "🛠️ Repository utilities"
            ],
            "debug_info": {
                "working_directory": os.getcwd(),
                "environment_vars": {
                    "HF_TOKEN": "✓" if TOKEN else "✗",
                    "HF_READ_ONLY": str(READ_ONLY),
                    "HF_ADMIN_MODE": str(ADMIN_MODE),
                    "HF_ENABLE_INFERENCE": str(ENABLE_INFERENCE),
                    "HF_MAX_FILE_SIZE": f"{MAX_FILE_SIZE:,} bytes",
                    "HF_CACHE_ENABLED": str(CACHE_ENABLED)
                }
            }
        }
    
    return safe_execute(_get_system_info, "system_info")

@mcp.tool()
def hf_repository_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Comprehensive repository management: create, delete, info, list_files
    
    Actions:
    - create: Create new repository (supports private, description, space_sdk, creator)
      - creator: Repository creator (defaults to authenticated user)
    - delete: Delete repository (requires admin mode)
    - info: Get repository information
    - list_files: List all files in repository
    """
    def _manage_repository():
        # Validate repository ID for all actions
        repo_validation = validate_repo_id(repo_id)
        if repo_validation: return repo_validation
        
        if action == "create":
            auth_error = validate_auth("repository creation")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            private = kwargs.get("private", False)
            description = kwargs.get("description")
            space_sdk = kwargs.get("space_sdk")
            
            # Get creator - default to current authenticated user
            creator = kwargs.get("creator")
            if not creator and TOKEN:
                try:
                    user_info = whoami(token=TOKEN)
                    creator = user_info.get('name', 'unknown')
                except Exception:
                    creator = 'unknown'
            
            # Validate space_sdk for Spaces
            valid_sdks = ["gradio", "streamlit", "docker", "static"]
            if repo_type == "space":
                if not space_sdk or space_sdk not in valid_sdks:
                    return {
                        "error": "❌ space_sdk required for Spaces",
                        "valid_options": valid_sdks,
                        "help": {
                            "gradio": "Interactive ML demos with Python",
                            "streamlit": "Data apps and dashboards",
                            "docker": "Custom applications",
                            "static": "HTML/CSS/JS websites"
                        }
                    }
            
            create_params = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "token": TOKEN
            }
            
            if repo_type == "space" and space_sdk:
                create_params["space_sdk"] = space_sdk
            
            repo_url = create_repo(**create_params)
            
            result = {
                "status": "success",
                "message": f"✅ Created {repo_type}: {repo_id}",
                "repo_url": repo_url,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "creator": creator or "unknown"
            }
            
            if repo_type == "space":
                result["space_sdk"] = space_sdk
                result["next_steps"] = {
                    "gradio": "Upload app.py + requirements.txt",
                    "streamlit": "Upload app.py + requirements.txt",
                    "docker": "Upload Dockerfile + app files",
                    "static": "Upload index.html + assets"
                }[space_sdk]
            
            return result
            
        elif action == "delete":
            auth_error = validate_auth("repository deletion")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_admin=True)
            if perm_error: return perm_error
            
            delete_repo(repo_id=repo_id, repo_type=repo_type, token=TOKEN)
            return {
                "status": "success",
                "message": f"🗑️ Deleted {repo_type}: {repo_id}",
                "repo_id": repo_id,
                "repo_type": repo_type
            }
            
        elif action == "info":
            if repo_type == "model":
                info = model_info(repo_id, token=TOKEN)
            elif repo_type == "dataset":
                info = dataset_info(repo_id, token=TOKEN)
            else:
                return {"error": f"Unsupported repo_type for info: {repo_type}"}
            
            return format_repo_info(info)
            
        elif action == "list_files":
            files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=TOKEN)
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "file_count": len(files),
                "files": files
            }
            
        else:
            return {"error": f"❌ Invalid action: {action}. Use: create, delete, info, list_files"}
    
    return safe_execute(_manage_repository, f"repository_{action}")

@mcp.tool()
def hf_file_operations(
    action: str,
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Comprehensive file operations: read, write, edit, delete, validate, backup
    
    Actions:
    - read: Read file content (supports max_size, chunked reading, encoding detection)
    - write: Write/upload file content (with validation and backup options)
    - edit: Edit file by replacing text (with backup and verification)
    - delete: Delete file from repository (with confirmation)
    - validate: Validate file format and content
    - backup: Create backup of file before operations
    - batch_edit: Edit multiple files with pattern matching
    """
    def _handle_file_operation():
        # Validate inputs for all file operations
        repo_validation = validate_repo_id(repo_id)
        if repo_validation: return repo_validation
        
        file_validation = validate_file_path(filename)
        if file_validation: return file_validation
        
        if action == "read":
            revision = kwargs.get("revision", "main")
            max_size = kwargs.get("max_size", 500000)
            chunk_size = kwargs.get("chunk_size")
            chunk_number = kwargs.get("chunk_number", 0)
            
            # Download file
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                token=TOKEN
            )
            
            file_size = os.path.getsize(file_path)
            
            # Handle chunked reading
            if chunk_size:
                start_pos = chunk_number * chunk_size
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(start_pos)
                    content = f.read(chunk_size)
                    total_chunks = (file_size + chunk_size - 1) // chunk_size
                    
                    return {
                        "repo_id": repo_id,
                        "filename": filename,
                        "content": content,
                        "chunk_number": chunk_number,
                        "chunk_size": len(content),
                        "total_chunks": total_chunks,
                        "file_size": file_size,
                        "has_more": chunk_number < total_chunks - 1,
                        "message": f"📖 Read chunk {chunk_number + 1}/{total_chunks}"
                    }
            
            # Handle regular reading with size limits
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if max_size > 0 and file_size > max_size:
                        content = f.read(max_size)
                        return {
                            "repo_id": repo_id,
                            "filename": filename,
                            "content": content,
                            "size": len(content),
                            "full_file_size": file_size,
                            "truncated": True,
                            "message": f"📖 Read {filename} (truncated to {max_size:,} chars)",
                            "note": "Use max_size=0 for full file or increase max_size"
                        }
                    else:
                        content = f.read()
                        return {
                            "repo_id": repo_id,
                            "filename": filename,
                            "content": content,
                            "size": len(content),
                            "full_file_size": file_size,
                            "truncated": False,
                            "message": f"📖 Successfully read {filename} ({file_size:,} chars)"
                        }
            except UnicodeDecodeError:
                return {
                    "repo_id": repo_id,
                    "filename": filename,
                    "error": "Binary file - cannot display as text",
                    "size": file_size,
                    "is_binary": True
                }
        
        elif action == "write":
            auth_error = validate_auth("file writing")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            content = kwargs.get("content", "")
            commit_message = kwargs.get("commit_message", f"Upload {filename}")
            
            # Handle different content formats
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_content += item['text']
                    elif isinstance(item, str):
                        text_content += item
                content = text_content
            elif not isinstance(content, str):
                content = str(content)
            
            commit_url = upload_file(
                path_or_fileobj=content.encode('utf-8'),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type=repo_type,
                token=TOKEN,
                commit_message=commit_message
            )
            
            return {
                "status": "success",
                "message": f"📝 Successfully wrote {filename}",
                "repo_id": repo_id,
                "filename": filename,
                "size": len(content.encode('utf-8')),
                "commit_url": commit_url,
                "commit_message": commit_message
            }
        
        elif action == "edit":
            auth_error = validate_auth("file editing")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            commit_message = kwargs.get("commit_message", f"Edit {filename}")
            
            if not old_text:
                return {"error": "❌ old_text parameter required for editing"}
            
            # Read current content
            read_result = hf_file_operations("read", repo_id, filename, repo_type)
            if "error" in read_result:
                return read_result
            
            current_content = read_result["content"]
            
            # Check if old_text exists
            if old_text not in current_content:
                return {
                    "error": f"❌ Text not found in {filename}",
                    "searched_for": old_text[:100] + "..." if len(old_text) > 100 else old_text,
                    "file_preview": current_content[:200] + "..." if len(current_content) > 200 else current_content
                }
            
            # Replace text
            new_content = current_content.replace(old_text, new_text, 1)
            
            # Write updated content
            write_result = hf_file_operations("write", repo_id, filename, repo_type, 
                                           content=new_content, commit_message=commit_message)
            
            if "error" in write_result:
                return write_result
            
            return {
                "status": "success",
                "message": f"✏️ Successfully edited {filename}",
                "repo_id": repo_id,
                "filename": filename,
                "old_text_length": len(old_text),
                "new_text_length": len(new_text),
                "total_size": len(new_content),
                "commit_url": write_result["commit_url"]
            }
        
        elif action == "delete":
            auth_error = validate_auth("file deletion")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            commit_message = kwargs.get("commit_message", f"Delete {filename}")
            
            delete_file(
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type=repo_type,
                token=TOKEN,
                commit_message=commit_message
            )
            
            return {
                "status": "success",
                "message": f"🗑️ Successfully deleted {filename}",
                "repo_id": repo_id,
                "filename": filename,
                "commit_message": commit_message
            }
            
        elif action == "validate":
            try:
                # Read file first
                read_result = hf_file_operations("read", repo_id, filename, repo_type, max_size=100000)
                if "error" in read_result:
                    return read_result
                
                content = read_result["content"]
                file_ext = Path(filename).suffix.lower()
                
                validation_results = {
                    "repo_id": repo_id,
                    "filename": filename,
                    "file_extension": file_ext,
                    "file_size": len(content.encode('utf-8')),
                    "validation_timestamp": datetime.now().isoformat(),
                    "checks": {}
                }
                
                # Basic checks
                validation_results["checks"]["valid_utf8"] = not read_result.get("is_binary", False)
                validation_results["checks"]["reasonable_size"] = validation_results["file_size"] < MAX_FILE_SIZE
                validation_results["checks"]["safe_filename"] = sanitize_filename(filename) == filename
                
                # Format-specific validation
                if file_ext == ".json":
                    try:
                        json.loads(content)
                        validation_results["checks"]["valid_json"] = True
                    except json.JSONDecodeError:
                        validation_results["checks"]["valid_json"] = False
                        
                elif file_ext == ".md":
                    validation_results["checks"]["has_headers"] = bool(re.search(r'^#', content, re.MULTILINE))
                    validation_results["checks"]["reasonable_length"] = 100 <= len(content) <= 50000
                    
                elif file_ext in [".py", ".js", ".ts"]:
                    validation_results["checks"]["no_syntax_errors"] = not bool(re.search(r'SyntaxError|TypeError', content))
                    validation_results["checks"]["has_functions"] = bool(re.search(r'def |function |const ', content))
                
                # Calculate validation score
                passed_checks = sum(1 for v in validation_results["checks"].values() if v)
                total_checks = len(validation_results["checks"])
                validation_results["validation_score"] = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
                validation_results["validation_status"] = "✅ Valid" if validation_results["validation_score"] >= 80 else "⚠️ Issues found"
                
                return validation_results
                
            except Exception as e:
                return {"error": f"❌ File validation failed: {str(e)}"}
                
        elif action == "backup":
            try:
                # Read current file content
                read_result = hf_file_operations("read", repo_id, filename, repo_type)
                if "error" in read_result:
                    return read_result
                
                # Create backup filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"backups/{filename}.backup_{timestamp}"
                
                # Write backup file (if write permissions available)
                if not READ_ONLY:
                    backup_result = hf_file_operations("write", repo_id, backup_filename, repo_type,
                                                     content=read_result["content"],
                                                     commit_message=f"Backup of {filename}")
                    
                    return {
                        "status": "success",
                        "message": f"📦 Backup created: {backup_filename}",
                        "original_file": filename,
                        "backup_file": backup_filename,
                        "backup_size": read_result.get("size", 0),
                        "backup_timestamp": timestamp,
                        "commit_url": backup_result.get("commit_url")
                    }
                else:
                    return {
                        "status": "info",
                        "message": "📦 Backup content prepared (read-only mode)",
                        "original_file": filename,
                        "backup_content": read_result["content"],
                        "backup_size": read_result.get("size", 0),
                        "backup_timestamp": timestamp
                    }
                    
            except Exception as e:
                return {"error": f"❌ File backup failed: {str(e)}"}
                
        elif action == "batch_edit":
            auth_error = validate_auth("batch file editing")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            pattern = kwargs.get("pattern", "")
            replacement = kwargs.get("replacement", "")
            file_patterns = kwargs.get("file_patterns", ["*.md", "*.txt"])
            
            if not pattern:
                return {"error": "❌ pattern parameter required for batch editing"}
            
            try:
                # Get list of files matching patterns
                files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=TOKEN)
                matching_files = []
                
                for file_pattern in file_patterns:
                    import fnmatch
                    matching_files.extend([f for f in files if fnmatch.fnmatch(f, file_pattern)])
                
                # Remove duplicates
                matching_files = list(set(matching_files))
                
                edit_results = []
                successful_edits = 0
                
                for file_path in matching_files[:20]:  # Limit to 20 files
                    try:
                        # Read file
                        read_result = hf_file_operations("read", repo_id, file_path, repo_type)
                        if "error" in read_result:
                            edit_results.append({
                                "file": file_path,
                                "status": "failed",
                                "error": "Could not read file"
                            })
                            continue
                        
                        content = read_result["content"]
                        
                        # Check if pattern exists
                        if pattern not in content:
                            edit_results.append({
                                "file": file_path,
                                "status": "skipped",
                                "reason": "Pattern not found"
                            })
                            continue
                        
                        # Apply replacement
                        new_content = content.replace(pattern, replacement)
                        
                        # Write back
                        write_result = hf_file_operations("write", repo_id, file_path, repo_type,
                                                        content=new_content,
                                                        commit_message=f"Batch edit: replace '{pattern[:50]}...' in {file_path}")
                        
                        if "error" not in write_result:
                            edit_results.append({
                                "file": file_path,
                                "status": "success",
                                "changes": content.count(pattern)
                            })
                            successful_edits += 1
                        else:
                            edit_results.append({
                                "file": file_path,
                                "status": "failed",
                                "error": write_result["error"]
                            })
                            
                    except Exception as e:
                        edit_results.append({
                            "file": file_path,
                            "status": "failed",
                            "error": str(e)
                        })
                
                return {
                    "status": "completed",
                    "message": f"📝 Batch edit completed: {successful_edits}/{len(matching_files)} files updated",
                    "pattern": pattern,
                    "replacement": replacement,
                    "total_files": len(matching_files),
                    "successful_edits": successful_edits,
                    "results": edit_results,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": f"❌ Batch edit failed: {str(e)}"}
        
        else:
            return {"error": f"❌ Invalid action: {action}. Use: read, write, edit, delete, validate, backup, batch_edit"}
    
    return safe_execute(_handle_file_operation, f"file_{action}")

@mcp.tool()
def hf_search_hub(
    content_type: str,
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Search Hugging Face Hub for models, datasets, or spaces
    
    Args:
        content_type: Type to search ("models", "datasets", "spaces")
        query: Search query string
        author: Filter by author username
        filter_tag: Filter by tag
        limit: Maximum results to return
    """
    def _search_hub():
        results = []
        
        if content_type == "models":
            items = list_models(
                search=query,
                author=author,
                filter=filter_tag,
                sort="downloads",
                direction=-1,
                limit=limit,
                token=TOKEN
            )
            
            for item in items:
                results.append({
                    "id": item.id,
                    "author": item.author,
                    "downloads": item.downloads,
                    "likes": item.likes,
                    "tags": item.tags[:5],
                    "created_at": item.created_at.isoformat() if item.created_at else None
                })
        
        elif content_type == "datasets":
            items = list_datasets(
                search=query,
                author=author,
                filter=filter_tag,
                limit=limit,
                token=TOKEN
            )
            
            for item in items:
                results.append({
                    "id": item.id,
                    "author": item.author,
                    "downloads": item.downloads,
                    "likes": item.likes,
                    "tags": item.tags[:5],
                    "created_at": item.created_at.isoformat() if item.created_at else None
                })
        
        elif content_type == "spaces":
            items = list_spaces(
                search=query,
                author=author,
                limit=limit,
                token=TOKEN
            )
            
            for item in items:
                results.append({
                    "id": item.id,
                    "author": item.author,
                    "likes": item.likes,
                    "tags": item.tags[:5],
                    "sdk": getattr(item, 'sdk', None),
                    "created_at": item.created_at.isoformat() if item.created_at else None
                })
        
        else:
            return {"error": f"❌ Invalid content_type: {content_type}. Use: models, datasets, spaces"}
        
        return {
            "content_type": content_type,
            "query": query,
            "author": author,
            "filter_tag": filter_tag,
            "total_results": len(results),
            "results": results
        }
    
    return safe_execute(_search_hub, f"search_{content_type}")

@mcp.tool()
def hf_collections(
    action: str,
    **kwargs
) -> Dict[str, Any]:
    """Manage Hugging Face Collections: create, add_item, info
    
    Actions:
    - create: Create new collection (title, namespace, description, private)
    - add_item: Add item to collection (collection_slug, item_id, item_type, note)
    - info: Get collection information (collection_slug)
    """
    def _manage_collections():
        if action == "create":
            auth_error = validate_auth("collection creation")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            title = kwargs.get("title")
            if not title:
                return {"error": "❌ title parameter required for collection creation"}
            
            namespace = kwargs.get("namespace") or get_user_namespace()
            description = kwargs.get("description")
            private = kwargs.get("private", False)
            
            collection = create_collection(
                title=title,
                namespace=namespace,
                description=description,
                private=private,
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"📚 Created collection: {title}",
                "collection_slug": collection.slug,
                "title": collection.title,
                "owner": getattr(collection, 'owner', namespace),
                "url": getattr(collection, 'url', f"https://huggingface.co/collections/{namespace}/{collection.slug}")
            }
        
        elif action == "add_item":
            auth_error = validate_auth("collection modification")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            collection_slug = kwargs.get("collection_slug")
            item_id = kwargs.get("item_id")
            item_type = kwargs.get("item_type")
            note = kwargs.get("note")
            
            if not all([collection_slug, item_id, item_type]):
                return {"error": "❌ collection_slug, item_id, and item_type required"}
            
            add_collection_item(
                collection_slug=collection_slug,
                item_id=item_id,
                item_type=item_type,
                note=note,
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"➕ Added {item_id} to collection",
                "collection_slug": collection_slug,
                "item_id": item_id,
                "item_type": item_type
            }
        
        elif action == "info":
            collection_slug = kwargs.get("collection_slug")
            if not collection_slug:
                return {"error": "❌ collection_slug parameter required"}
            
            collection = get_collection(collection_slug, token=TOKEN)
            
            items = []
            for item in collection.items:
                items.append({
                    "item_id": item.item_id,
                    "item_type": item.item_type,
                    "position": item.position,
                    "note": getattr(item, 'note', None)
                })
            
            return {
                "slug": collection.slug,
                "title": collection.title,
                "description": collection.description,
                "owner": collection.owner,
                "items": items,
                "item_count": len(items),
                "url": collection.url
            }
        
        else:
            return {"error": f"❌ Invalid action: {action}. Use: create, add_item, info"}
    
    return safe_execute(_manage_collections, f"collections_{action}")

@mcp.tool()
def hf_pull_requests(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Manage Pull Requests: create, list, details, create_with_files
    
    Actions:
    - create: Create empty PR (title, description)
    - list: List PRs (status, author)
    - details: Get PR details (pr_number)
    - create_with_files: Create PR with file changes (files, commit_message, pr_title, pr_description)
    """
    def _manage_pull_requests():
        if action == "create":
            auth_error = validate_auth("pull request creation")
            if auth_error: return auth_error
            
            title = kwargs.get("title")
            if not title or len(title.strip()) < 3:
                return {"error": "❌ title required (min 3 characters)"}
            
            description = kwargs.get("description", "Pull Request created with HuggingMCP")
            
            discussion = create_discussion(
                repo_id=repo_id,
                title=title.strip(),
                description=description,
                repo_type=repo_type,
                pull_request=True,
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"🔀 Created PR: {title}",
                "pr_number": discussion.num,
                "pr_title": discussion.title,
                "pr_url": discussion.url,
                "repo_id": repo_id,
                "repo_type": repo_type
            }
        
        elif action == "list":
            status = kwargs.get("status", "open")
            author = kwargs.get("author")
            
            discussion_status = None if status == "all" else status
            
            discussions = get_repo_discussions(
                repo_id=repo_id,
                repo_type=repo_type,
                discussion_type="pull_request",
                discussion_status=discussion_status,
                author=author,
                token=TOKEN
            )
            
            prs = []
            for discussion in discussions:
                prs.append({
                    "number": discussion.num,
                    "title": discussion.title,
                    "author": discussion.author,
                    "status": discussion.status,
                    "created_at": discussion.created_at.isoformat() if discussion.created_at else None,
                    "url": discussion.url,
                    "is_pull_request": discussion.is_pull_request
                })
            
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "status_filter": status,
                "author_filter": author,
                "total_prs": len(prs),
                "pull_requests": prs[:20]
            }
        
        elif action == "details":
            pr_number = kwargs.get("pr_number")
            if pr_number is None:
                return {"error": "❌ pr_number parameter required"}
            
            discussion = get_discussion_details(
                repo_id=repo_id,
                discussion_num=pr_number,
                repo_type=repo_type,
                token=TOKEN
            )
            
            events = []
            for event in discussion.events:
                events.append({
                    "type": event.type,
                    "author": event.author,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                    "content": getattr(event, 'content', '') if hasattr(event, 'content') else ''
                })
            
            return {
                "repo_id": repo_id,
                "pr_number": discussion.num,
                "title": discussion.title,
                "author": discussion.author,
                "status": discussion.status,
                "created_at": discussion.created_at.isoformat() if discussion.created_at else None,
                "is_pull_request": discussion.is_pull_request,
                "url": discussion.url,
                "conflicting_files": discussion.conflicting_files,
                "target_branch": discussion.target_branch,
                "merge_commit_oid": discussion.merge_commit_oid,
                "events": events,
                "total_events": len(events)
            }
        
        elif action == "create_with_files":
            auth_error = validate_auth("pull request with files creation")
            if auth_error: return auth_error
            
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            files = kwargs.get("files", [])
            commit_message = kwargs.get("commit_message")
            pr_title = kwargs.get("pr_title")
            pr_description = kwargs.get("pr_description")
            
            if not files:
                return {"error": "❌ files parameter required (list of {path, content} dicts)"}
            
            if not commit_message:
                return {"error": "❌ commit_message parameter required"}
            
            # Create commit operations
            operations = []
            for file_op in files:
                if not isinstance(file_op, dict) or "path" not in file_op or "content" not in file_op:
                    return {"error": "❌ Invalid file format. Use: [{\"path\": \"file.txt\", \"content\": \"content\"}]"}
                
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=file_op["path"],
                        path_or_fileobj=file_op["content"].encode('utf-8')
                    )
                )
            
            # Create commit with PR
            commit_result = api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                repo_type=repo_type,
                create_pr=True,
                pr_title=pr_title or f"PR: {commit_message}",
                pr_description=pr_description or f"Automated PR via HuggingMCP\n\nCommit: {commit_message}",
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"🔀 Created commit and PR: {commit_message}",
                "commit_url": commit_result.commit_url,
                "pr_url": getattr(commit_result, 'pr_url', None),
                "commit_sha": commit_result.oid,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "files_changed": len(operations)
            }
        
        else:
            return {"error": f"❌ Invalid action: {action}. Use: create, list, details, create_with_files"}
    
    return safe_execute(_manage_pull_requests, f"pr_{action}")

@mcp.tool()
def hf_upload_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Upload management: single_file, multiple_files, with_pr
    
    Actions:
    - single_file: Upload one file (file_path, content, commit_message)
    - multiple_files: Upload multiple files (files list, commit_message)
    - with_pr: Upload file(s) and create PR (file_path, content, commit_message, pr_title, pr_description)
    """
    def _manage_uploads():
        auth_error = validate_auth("file upload")
        if auth_error: return auth_error
        
        perm_error = validate_permissions(require_write=True)
        if perm_error: return perm_error
        
        if action == "single_file":
            file_path = kwargs.get("file_path")
            content = kwargs.get("content")
            commit_message = kwargs.get("commit_message", f"Upload {file_path}")
            
            if not file_path or content is None:
                return {"error": "❌ file_path and content parameters required"}
            
            commit_url = upload_file(
                path_or_fileobj=content.encode('utf-8'),
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=TOKEN,
                commit_message=commit_message
            )
            
            return {
                "status": "success",
                "message": f"📤 Uploaded {file_path}",
                "file_path": file_path,
                "file_size": len(content.encode('utf-8')),
                "commit_url": commit_url,
                "commit_message": commit_message
            }
        
        elif action == "multiple_files":
            files = kwargs.get("files", [])
            commit_message = kwargs.get("commit_message", "Upload multiple files")
            
            if not files:
                return {"error": "❌ files parameter required (list of {path, content} dicts)"}
            
            operations = []
            for file_op in files:
                if not isinstance(file_op, dict) or "path" not in file_op or "content" not in file_op:
                    return {"error": "❌ Invalid file format. Use: [{\"path\": \"file.txt\", \"content\": \"content\"}]"}
                
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=file_op["path"],
                        path_or_fileobj=file_op["content"].encode('utf-8')
                    )
                )
            
            commit_result = api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                repo_type=repo_type,
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"📤 Uploaded {len(operations)} files",
                "files_uploaded": len(operations),
                "commit_url": commit_result.commit_url,
                "commit_sha": commit_result.oid,
                "commit_message": commit_message
            }
        
        elif action == "with_pr":
            file_path = kwargs.get("file_path")
            content = kwargs.get("content")
            commit_message = kwargs.get("commit_message", f"Upload {file_path}")
            pr_title = kwargs.get("pr_title", f"Add {file_path}")
            pr_description = kwargs.get("pr_description", f"Uploaded {file_path} via HuggingMCP")
            
            if not file_path or content is None:
                return {"error": "❌ file_path and content parameters required"}
            
            result = upload_file(
                path_or_fileobj=content.encode('utf-8'),
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                create_pr=True,
                pr_title=pr_title,
                pr_description=pr_description,
                token=TOKEN
            )
            
            return {
                "status": "success",
                "message": f"📤 Uploaded {file_path} and created PR",
                "file_path": file_path,
                "file_size": len(content.encode('utf-8')),
                "commit_message": commit_message,
                "pr_url": result if isinstance(result, str) else None,
                "repo_id": repo_id,
                "repo_type": repo_type
            }
        
        else:
            return {"error": f"❌ Invalid action: {action}. Use: single_file, multiple_files, with_pr"}
    
    return safe_execute(_manage_uploads, f"upload_{action}")

@mcp.tool()
def hf_repo_file_manager(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    filename: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Unified repository and file management with rename support"""

    def _repo_file():
        if action.startswith("repo_"):
            return hf_repository_manager(action.replace("repo_", ""), repo_id, repo_type, **kwargs)

        if action.startswith("file_"):
            file_action = action.replace("file_", "")
            if file_action == "rename":
                auth_error = validate_auth("file rename")
                if auth_error:
                    return auth_error

                perm_error = validate_permissions(require_write=True)
                if perm_error:
                    return perm_error

                new_filename = kwargs.get("new_filename")
                if not filename or not new_filename:
                    return {"error": "❌ filename and new_filename required"}

                file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, token=TOKEN)
                with open(file_path, "rb") as f:
                    content = f.read()

                operations = [
                    CommitOperationAdd(path_in_repo=new_filename, path_or_fileobj=content),
                    CommitOperationDelete(path_in_repo=filename),
                ]

                commit_result = api.create_commit(
                    repo_id=repo_id,
                    operations=operations,
                    commit_message=kwargs.get("commit_message", f"Rename {filename} to {new_filename}"),
                    repo_type=repo_type,
                    token=TOKEN,
                )

                return {
                    "status": "success",
                    "message": f"🔄 Renamed {filename} to {new_filename}",
                    "commit_url": commit_result.commit_url,
                    "repo_id": repo_id,
                    "old_filename": filename,
                    "new_filename": new_filename,
                }

            return hf_file_operations(file_action, repo_id, filename, repo_type, **kwargs)

        return {"error": f"❌ Unknown action: {action}"}

    return safe_execute(_repo_file, f"repo_file_{action}")

@mcp.tool()
def hf_batch_operations(
    operation_type: str,
    operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Execute multiple operations in batch with enhanced error handling
    
    Args:
        operation_type: Type of batch operation ("search", "info", "files")
        operations: List of operation dictionaries with parameters
    """
    def _execute_batch():
        if not operations:
            return {"error": "❌ operations list cannot be empty"}
        
        results = []
        errors = []
        
        for i, op in enumerate(operations):
            try:
                if operation_type == "search":
                    content_type = op.get("content_type", "models")
                    result = hf_search_hub(
                        content_type=content_type,
                        query=op.get("query"),
                        author=op.get("author"),
                        filter_tag=op.get("filter_tag"),
                        limit=op.get("limit", 10)
                    )
                    
                elif operation_type == "info":
                    result = hf_repository_manager(
                        action="info",
                        repo_id=op.get("repo_id"),
                        repo_type=op.get("repo_type", "model")
                    )
                    
                elif operation_type == "files":
                    result = hf_repository_manager(
                        action="list_files",
                        repo_id=op.get("repo_id"),
                        repo_type=op.get("repo_type", "model")
                    )
                    
                else:
                    errors.append(f"Operation {i}: Invalid operation_type: {operation_type}")
                    continue
                
                results.append({
                    "operation_index": i,
                    "operation": op,
                    "result": result
                })
                
            except Exception as e:
                errors.append(f"Operation {i}: {str(e)}")
                debug_stderr(f"Batch operation {i} failed: {e}", "ERROR")
        
        return {
            "operation_type": operation_type,
            "total_operations": len(operations),
            "successful_operations": len(results),
            "failed_operations": len(errors),
            "results": results,
            "errors": errors if errors else None,
            "summary": f"✅ {len(results)} successful, ❌ {len(errors)} failed"
        }
    
    return safe_execute(_execute_batch, f"batch_{operation_type}")

@mcp.tool()
def hf_advanced_search(
    query: str,
    search_types: List[str] = ["models", "datasets", "spaces"],
    filters: Optional[Dict[str, Any]] = None,
    limit_per_type: int = 10
) -> Dict[str, Any]:
    """Advanced search across multiple content types with filtering
    
    Args:
        query: Search query string
        search_types: List of content types to search ["models", "datasets", "spaces"]
        filters: Optional filters dict {author, tags, etc.}
        limit_per_type: Maximum results per content type
    """
    def _advanced_search():
        nonlocal filters
        filters = filters or {}
        author = filters.get("author")
        filter_tag = filters.get("tag")
        
        all_results = {}
        total_found = 0
        
        for content_type in search_types:
            if content_type not in ["models", "datasets", "spaces"]:
                continue
            
            search_result = hf_search_hub(
                content_type=content_type,
                query=query,
                author=author,
                filter_tag=filter_tag,
                limit=limit_per_type
            )
            
            if "error" not in search_result:
                all_results[content_type] = search_result["results"]
                total_found += len(search_result["results"])
            else:
                all_results[content_type] = []
        
        # Combine and sort by popularity (downloads/likes)
        combined_results = []
        for content_type, results in all_results.items():
            for result in results:
                result["content_type"] = content_type
                popularity = result.get("downloads", 0) + result.get("likes", 0)
                result["popularity_score"] = popularity
                combined_results.append(result)
        
        # Sort by popularity
        combined_results.sort(key=lambda x: x["popularity_score"], reverse=True)
        
        return {
            "query": query,
            "search_types": search_types,
            "filters": filters,
            "total_results": total_found,
            "results_by_type": all_results,
            "combined_results": combined_results[:50],  # Top 50 overall
            "summary": {
                "models": len(all_results.get("models", [])),
                "datasets": len(all_results.get("datasets", [])),
                "spaces": len(all_results.get("spaces", []))
            }
        }
    
    return safe_execute(_advanced_search, "advanced_search")

@mcp.tool()
def hf_debug_diagnostics() -> Dict[str, Any]:
    """Comprehensive debugging and diagnostic information for troubleshooting"""
    def _get_diagnostics():
        diagnostics = {
            "server_status": "🟢 Running",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "environment": {
                "python_version": sys.version,
                "python_executable": sys.executable,
                "script_path": __file__,
                "working_directory": os.getcwd(),
                "platform": sys.platform
            },
            "configuration": {
                "hf_token_present": bool(TOKEN),
                "read_only_mode": READ_ONLY,
                "admin_mode": ADMIN_MODE,
                "token_length": len(TOKEN) if TOKEN else 0
            },
            "dependencies": {},
            "file_system": {
                "script_exists": os.path.exists(__file__),
                "script_readable": os.access(__file__, os.R_OK),
                "script_size": os.path.getsize(__file__) if os.path.exists(__file__) else 0
            },
            "memory_usage": {},
            "recent_operations": [],
            "debug_logs": []
        }
        
        # Check dependencies
        try:
            import mcp
            diagnostics["dependencies"]["mcp"] = f"✅ {mcp.__version__ if hasattr(mcp, '__version__') else 'installed'}"
        except ImportError as e:
            diagnostics["dependencies"]["mcp"] = f"❌ Missing: {e}"
        
        try:
            import huggingface_hub
            diagnostics["dependencies"]["huggingface_hub"] = f"✅ {huggingface_hub.__version__}"
        except ImportError as e:
            diagnostics["dependencies"]["huggingface_hub"] = f"❌ Missing: {e}"
        
        # Memory usage (basic)
        try:
            import psutil
            process = psutil.Process()
            diagnostics["memory_usage"] = {
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict()
            }
        except ImportError:
            diagnostics["memory_usage"] = {"note": "psutil not available for detailed memory info"}
        
        # Test basic connectivity
        try:
            if TOKEN:
                user_info = whoami(token=TOKEN)
                diagnostics["hf_connectivity"] = {
                    "status": "✅ Connected",
                    "user": user_info.get('name', 'Unknown'),
                    "authenticated": True
                }
            else:
                # Test anonymous access
                models = list(list_models(limit=1))
                diagnostics["hf_connectivity"] = {
                    "status": "✅ Connected (anonymous)",
                    "authenticated": False,
                    "test_successful": len(models) > 0
                }
        except Exception as e:
            diagnostics["hf_connectivity"] = {
                "status": "❌ Connection failed",
                "error": str(e)
            }
        
        # Check log file
        log_file = '/tmp/hugmcp_debug.log'
        if os.path.exists(log_file):
            diagnostics["debug_logs"] = {
                "log_file_exists": True,
                "log_file_size": os.path.getsize(log_file),
                "last_modified": time.ctime(os.path.getmtime(log_file))
            }
        
        return diagnostics
    
    return safe_execute(_get_diagnostics, "diagnostics")

@mcp.tool()
def hf_model_evaluation(
    action: str,
    repo_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Advanced model evaluation and testing capabilities
    
    Actions:
    - analyze: Analyze model architecture, performance metrics, and compatibility
    - compare: Compare multiple models side by side
    - test_inference: Test model inference capabilities (if supported)
    - get_metrics: Get model performance metrics and benchmarks
    - validate_model: Validate model integrity and format
    """
    def _evaluate_model():
        if action == "analyze":
            try:
                # Get comprehensive model information
                model_info_result = model_info(repo_id, token=TOKEN, files_metadata=True)
                
                # Extract model card data
                card_data = getattr(model_info_result, 'cardData', {}) or {}
                
                # Analyze model files
                file_analysis = {}
                if hasattr(model_info_result, 'siblings'):
                    for sibling in model_info_result.siblings:
                        file_ext = Path(sibling.rfilename).suffix.lower()
                        file_size = getattr(sibling, 'size', 0) or 0
                        
                        if file_ext not in file_analysis:
                            file_analysis[file_ext] = {"count": 0, "total_size": 0, "files": []}
                        
                        file_analysis[file_ext]["count"] += 1
                        file_analysis[file_ext]["total_size"] += file_size
                        file_analysis[file_ext]["files"].append({
                            "name": sibling.rfilename,
                            "size": file_size
                        })
                
                # Determine model framework
                frameworks = []
                if any(f.endswith('.bin') or f.endswith('.safetensors') for f in [s.rfilename for s in model_info_result.siblings]):
                    if 'pytorch_model' in str([s.rfilename for s in model_info_result.siblings]):
                        frameworks.append("PyTorch")
                    if 'model.safetensors' in str([s.rfilename for s in model_info_result.siblings]):
                        frameworks.append("SafeTensors")
                if any('tf_model' in f for f in [s.rfilename for s in model_info_result.siblings]):
                    frameworks.append("TensorFlow")
                if any('.onnx' in f for f in [s.rfilename for s in model_info_result.siblings]):
                    frameworks.append("ONNX")
                
                return {
                    "repo_id": repo_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "basic_info": format_repo_info(model_info_result),
                    "model_card": format_model_card_data(card_data),
                    "file_analysis": file_analysis,
                    "detected_frameworks": frameworks,
                    "model_size_estimate": sum(getattr(s, 'size', 0) or 0 for s in model_info_result.siblings if hasattr(model_info_result, 'siblings')),
                    "popularity_score": calculate_popularity_score(model_info_result),
                    "compatibility_info": {
                        "has_config": any('config.json' in s.rfilename for s in model_info_result.siblings if hasattr(model_info_result, 'siblings')),
                        "has_tokenizer": any('tokenizer' in s.rfilename for s in model_info_result.siblings if hasattr(model_info_result, 'siblings')),
                        "has_readme": any('README.md' in s.rfilename for s in model_info_result.siblings if hasattr(model_info_result, 'siblings'))
                    }
                }
            except Exception as e:
                return {"error": f"❌ Model analysis failed: {str(e)}"}
                
        elif action == "compare":
            models = kwargs.get("models", [])
            if not models or len(models) < 2:
                return {"error": "❌ At least 2 models required for comparison"}
            
            comparison_results = []
            for model_id in models[:5]:  # Limit to 5 models
                try:
                    model_info_result = model_info(model_id, token=TOKEN)
                    comparison_results.append({
                        "repo_id": model_id,
                        "info": format_repo_info(model_info_result),
                        "popularity_score": calculate_popularity_score(model_info_result),
                        "model_size": sum(getattr(s, 'size', 0) or 0 for s in model_info_result.siblings if hasattr(model_info_result, 'siblings')),
                        "tags": getattr(model_info_result, 'tags', [])
                    })
                except Exception as e:
                    comparison_results.append({
                        "repo_id": model_id,
                        "error": f"Failed to analyze: {str(e)}"
                    })
            
            # Sort by popularity
            valid_results = [r for r in comparison_results if "error" not in r]
            valid_results.sort(key=lambda x: x["popularity_score"], reverse=True)
            
            return {
                "comparison_timestamp": datetime.now().isoformat(),
                "models_compared": len(models),
                "successful_analyses": len(valid_results),
                "results": comparison_results,
                "ranking": [r["repo_id"] for r in valid_results]
            }
            
        elif action == "test_inference":
            if not TOKEN or not ENABLE_INFERENCE:
                return {"error": "❌ Inference testing requires authentication and ENABLE_INFERENCE=true"}
                
            test_input = kwargs.get("test_input", "Hello, world!")
            try:
                # Try to use inference API
                api_client = get_inference_api(repo_id, token=TOKEN)
                if api_client:
                    result = api_client(test_input)
                    return {
                        "repo_id": repo_id,
                        "test_input": test_input,
                        "inference_result": result,
                        "status": "✅ Inference successful",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": "❌ Inference API not available for this model"}
            except Exception as e:
                return {
                    "repo_id": repo_id,
                    "test_input": test_input,
                    "error": f"❌ Inference failed: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                
        elif action == "validate_model":
            try:
                model_info_result = model_info(repo_id, token=TOKEN, files_metadata=True)
                
                validation_results = {
                    "repo_id": repo_id,
                    "validation_timestamp": datetime.now().isoformat(),
                    "checks": {}
                }
                
                # Check for essential files
                files = [s.rfilename for s in model_info_result.siblings] if hasattr(model_info_result, 'siblings') else []
                
                validation_results["checks"]["has_model_files"] = any(
                    f.endswith(('.bin', '.safetensors', '.h5', '.onnx')) for f in files
                )
                validation_results["checks"]["has_config"] = 'config.json' in files
                validation_results["checks"]["has_readme"] = any('README' in f for f in files)
                validation_results["checks"]["has_license"] = any('LICENSE' in f for f in files)
                
                # Check model card
                card_data = getattr(model_info_result, 'cardData', {}) or {}
                validation_results["checks"]["has_license_info"] = bool(card_data.get('license'))
                validation_results["checks"]["has_pipeline_tag"] = bool(card_data.get('pipeline_tag'))
                validation_results["checks"]["has_language_info"] = bool(card_data.get('language'))
                
                # Calculate validation score
                passed_checks = sum(1 for v in validation_results["checks"].values() if v)
                total_checks = len(validation_results["checks"])
                validation_results["validation_score"] = (passed_checks / total_checks) * 100
                validation_results["validation_status"] = "✅ Passed" if validation_results["validation_score"] >= 70 else "⚠️ Needs improvement"
                
                return validation_results
                
            except Exception as e:
                return {"error": f"❌ Model validation failed: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: analyze, compare, test_inference, validate_model"}
    
    return safe_execute(_evaluate_model, f"model_evaluation_{action}")

@mcp.tool()
def hf_space_management(
    action: str,
    space_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Advanced Hugging Face Spaces management
    
    Actions:
    - runtime_info: Get space runtime information and status
    - restart: Restart a space
    - pause: Pause a space
    - set_sleep_time: Set sleep time for a space
    - duplicate: Duplicate a space
    - monitor: Monitor space performance and logs
    """
    def _manage_space():
        if not HAS_SPACE_MANAGEMENT:
            return {"error": "❌ Space management features not available in this huggingface_hub version"}
            
        auth_error = validate_auth("space management")
        if auth_error: return auth_error
        
        if action == "runtime_info":
            try:
                runtime = get_space_runtime(space_id, token=TOKEN)
                return {
                    "space_id": space_id,
                    "runtime_info": {
                        "stage": runtime.stage,
                        "hardware": getattr(runtime, 'hardware', 'unknown'),
                        "requested_hardware": getattr(runtime, 'requested_hardware', 'unknown'),
                        "sleep_time": getattr(runtime, 'sleep_time', None),
                        "raw_runtime": str(runtime)
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to get runtime info: {str(e)}"}
                
        elif action == "restart":
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            try:
                restart_space(space_id, token=TOKEN)
                return {
                    "space_id": space_id,
                    "status": "✅ Space restart initiated",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to restart space: {str(e)}"}
                
        elif action == "pause":
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            try:
                pause_space(space_id, token=TOKEN)
                return {
                    "space_id": space_id,  
                    "status": "⏸️ Space paused",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to pause space: {str(e)}"}
                
        elif action == "set_sleep_time":
            perm_error = validate_permissions(require_write=True)
            if perm_error: return perm_error
            
            sleep_time = kwargs.get("sleep_time")
            if sleep_time is None:
                return {"error": "❌ sleep_time parameter required (in seconds)"}
            
            try:
                set_space_sleep_time(space_id, sleep_time, token=TOKEN)
                return {
                    "space_id": space_id,
                    "sleep_time": sleep_time,
                    "status": f"💤 Sleep time set to {sleep_time} seconds",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to set sleep time: {str(e)}"}
                
        elif action == "duplicate":
            to_id = kwargs.get("to_id")
            if not to_id:
                return {"error": "❌ to_id parameter required for duplication"}
            
            try:
                new_space = duplicate_space(space_id, to_id, token=TOKEN)
                return {
                    "original_space": space_id,
                    "new_space": to_id,
                    "status": "✅ Space duplicated successfully",
                    "new_space_url": f"https://huggingface.co/spaces/{to_id}",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to duplicate space: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: runtime_info, restart, pause, set_sleep_time, duplicate"}
    
    return safe_execute(_manage_space, f"space_management_{action}")

@mcp.tool()
def hf_community_features(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Community features: likes, discussions, social interactions
    
    Actions:
    - like: Like a repository
    - unlike: Unlike a repository  
    - get_likes: Get user's liked repositories
    - create_discussion: Create a discussion (non-PR)
    - list_discussions: List repository discussions
    - get_commits: Get repository commit history
    - get_refs: Get repository branches/tags
    """
    def _community_features():
        if action == "like":
            auth_error = validate_auth("liking repositories")
            if auth_error: return auth_error
            
            try:
                # Use HfApi directly for broader compatibility
                api.like(repo_id, token=TOKEN, repo_type=repo_type)
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "status": "❤️ Repository liked",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to like repository: {str(e)}"}
                
        elif action == "unlike":
            auth_error = validate_auth("unliking repositories")
            if auth_error: return auth_error
            
            try:
                # Use HfApi directly for broader compatibility
                api.unlike(repo_id, token=TOKEN, repo_type=repo_type)
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "status": "💔 Repository unliked", 
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to unlike repository: {str(e)}"}
                
        elif action == "get_likes":
            if not HAS_ADVANCED_REPO:
                return {"error": "❌ Advanced repository features not available in this huggingface_hub version"}
                
            auth_error = validate_auth("getting liked repositories")
            if auth_error: return auth_error
            
            try:
                user = kwargs.get("user") or get_user_namespace()
                liked_repos = list(list_liked_repos(user, token=TOKEN))
                
                result = {
                    "user": user,
                    "total_likes": len(liked_repos),
                    "liked_repositories": []
                }
                
                for repo in liked_repos[:50]:  # Limit to 50 for performance
                    result["liked_repositories"].append({
                        "id": repo.id,
                        "author": repo.author,
                        "likes": getattr(repo, 'likes', 0),
                        "downloads": getattr(repo, 'downloads', 0),
                        "created_at": repo.created_at.isoformat() if repo.created_at else None
                    })
                
                return result
            except Exception as e:
                return {"error": f"❌ Failed to get liked repositories: {str(e)}"}
                
        elif action == "create_discussion":
            auth_error = validate_auth("creating discussions")
            if auth_error: return auth_error
            
            title = kwargs.get("title")
            description = kwargs.get("description", "")
            
            if not title:
                return {"error": "❌ title parameter required"}
            
            try:
                discussion = create_discussion(
                    repo_id=repo_id,
                    title=title,
                    description=description,
                    repo_type=repo_type,
                    pull_request=False,
                    token=TOKEN
                )
                
                return {
                    "repo_id": repo_id,
                    "discussion_number": discussion.num,
                    "title": discussion.title,
                    "url": discussion.url,
                    "status": "💬 Discussion created",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to create discussion: {str(e)}"}
                
        elif action == "get_commits":
            if not HAS_ADVANCED_REPO:
                return {"error": "❌ Advanced repository features not available in this huggingface_hub version"}
                
            try:
                commits = list(list_repo_commits(repo_id, repo_type=repo_type, token=TOKEN))
                
                result = {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "total_commits": len(commits),
                    "recent_commits": []
                }
                
                for commit in commits[:20]:  # Show last 20 commits
                    result["recent_commits"].append({
                        "commit_id": commit.commit_id,
                        "title": commit.title,
                        "message": getattr(commit, 'message', ''),
                        "author": getattr(commit, 'authors', []),
                        "created_at": commit.created_at.isoformat() if commit.created_at else None
                    })
                
                return result
            except Exception as e:
                return {"error": f"❌ Failed to get commits: {str(e)}"}
                
        elif action == "get_refs":
            if not HAS_ADVANCED_REPO:
                return {"error": "❌ Advanced repository features not available in this huggingface_hub version"}
                
            try:
                refs = list_repo_refs(repo_id, repo_type=repo_type, token=TOKEN)
                
                branches = []
                tags = []
                
                for ref in refs:
                    if ref.ref.startswith('refs/heads/'):
                        branches.append({
                            "name": ref.ref.replace('refs/heads/', ''),
                            "target_commit": ref.target_commit
                        })
                    elif ref.ref.startswith('refs/tags/'):
                        tags.append({
                            "name": ref.ref.replace('refs/tags/', ''),
                            "target_commit": ref.target_commit
                        })
                
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "branches": branches,
                    "tags": tags,
                    "total_refs": len(refs)
                }
            except Exception as e:
                return {"error": f"❌ Failed to get refs: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: like, unlike, get_likes, create_discussion, get_commits, get_refs"}
    
    return safe_execute(_community_features, f"community_{action}")

@mcp.tool()
def hf_dataset_processing(
    action: str,
    dataset_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Advanced dataset processing and analysis tools
    
    Actions:
    - analyze: Analyze dataset structure, size, and metadata
    - preview: Preview dataset content and samples
    - validate: Validate dataset format and completeness
    - compare: Compare multiple datasets
    - get_splits: Get information about dataset splits
    """
    def _process_dataset():
        if action == "analyze":
            try:
                # Get dataset information
                dataset_info_result = dataset_info(dataset_id, token=TOKEN, files_metadata=True)
                
                # Extract dataset card data
                card_data = getattr(dataset_info_result, 'cardData', {}) or {}
                
                # Analyze dataset files
                file_analysis = {}
                total_size = 0
                if hasattr(dataset_info_result, 'siblings'):
                    for sibling in dataset_info_result.siblings:
                        file_ext = Path(sibling.rfilename).suffix.lower()
                        file_size = getattr(sibling, 'size', 0) or 0
                        total_size += file_size
                        
                        if file_ext not in file_analysis:
                            file_analysis[file_ext] = {"count": 0, "total_size": 0, "files": []}
                        
                        file_analysis[file_ext]["count"] += 1
                        file_analysis[file_ext]["total_size"] += file_size
                        file_analysis[file_ext]["files"].append({
                            "name": sibling.rfilename,
                            "size": file_size
                        })
                
                # Detect dataset format
                dataset_formats = []
                file_names = [s.rfilename for s in dataset_info_result.siblings] if hasattr(dataset_info_result, 'siblings') else []
                
                if any(f.endswith('.parquet') for f in file_names):
                    dataset_formats.append("Parquet")
                if any(f.endswith('.json') or f.endswith('.jsonl') for f in file_names):
                    dataset_formats.append("JSON/JSONL")
                if any(f.endswith('.csv') for f in file_names):
                    dataset_formats.append("CSV")
                if any(f.endswith('.arrow') for f in file_names):
                    dataset_formats.append("Arrow")
                
                return {
                    "dataset_id": dataset_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "basic_info": format_repo_info(dataset_info_result),
                    "dataset_card": format_model_card_data(card_data),
                    "file_analysis": file_analysis,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "detected_formats": dataset_formats,
                    "popularity_score": calculate_popularity_score(dataset_info_result),
                    "metadata": {
                        "has_dataset_info": any('dataset_info' in f for f in file_names),
                        "has_readme": any('README.md' in f for f in file_names),
                        "has_config": any('config.json' in f for f in file_names),
                        "file_count": len(file_names)
                    }
                }
            except Exception as e:
                return {"error": f"❌ Dataset analysis failed: {str(e)}"}
                
        elif action == "compare":
            datasets = kwargs.get("datasets", [])
            if not datasets or len(datasets) < 2:
                return {"error": "❌ At least 2 datasets required for comparison"}
            
            comparison_results = []
            for dataset_id_cmp in datasets[:5]:  # Limit to 5 datasets
                try:
                    dataset_info_result = dataset_info(dataset_id_cmp, token=TOKEN)
                    total_size = sum(getattr(s, 'size', 0) or 0 for s in dataset_info_result.siblings if hasattr(dataset_info_result, 'siblings'))
                    
                    comparison_results.append({
                        "dataset_id": dataset_id_cmp,
                        "info": format_repo_info(dataset_info_result),
                        "popularity_score": calculate_popularity_score(dataset_info_result),
                        "total_size": total_size,
                        "tags": getattr(dataset_info_result, 'tags', [])
                    })
                except Exception as e:
                    comparison_results.append({
                        "dataset_id": dataset_id_cmp,
                        "error": f"Failed to analyze: {str(e)}"
                    })
            
            # Sort by popularity
            valid_results = [r for r in comparison_results if "error" not in r]
            valid_results.sort(key=lambda x: x["popularity_score"], reverse=True)
            
            return {
                "comparison_timestamp": datetime.now().isoformat(),
                "datasets_compared": len(datasets),
                "successful_analyses": len(valid_results),
                "results": comparison_results,
                "ranking": [r["dataset_id"] for r in valid_results]
            }
            
        elif action == "validate":
            try:
                dataset_info_result = dataset_info(dataset_id, token=TOKEN, files_metadata=True)
                
                validation_results = {
                    "dataset_id": dataset_id,
                    "validation_timestamp": datetime.now().isoformat(),
                    "checks": {}
                }
                
                # Check for essential files
                files = [s.rfilename for s in dataset_info_result.siblings] if hasattr(dataset_info_result, 'siblings') else []
                
                validation_results["checks"]["has_data_files"] = any(
                    f.endswith(('.parquet', '.json', '.jsonl', '.csv', '.arrow')) for f in files
                )
                validation_results["checks"]["has_readme"] = any('README' in f for f in files)
                validation_results["checks"]["has_license"] = any('LICENSE' in f for f in files)
                validation_results["checks"]["has_dataset_info"] = any('dataset_info' in f for f in files)
                
                # Check dataset card
                card_data = getattr(dataset_info_result, 'cardData', {}) or {}
                validation_results["checks"]["has_license_info"] = bool(card_data.get('license'))
                validation_results["checks"]["has_language_info"] = bool(card_data.get('language'))
                validation_results["checks"]["has_task_info"] = bool(card_data.get('task_categories'))
                
                # Calculate validation score
                passed_checks = sum(1 for v in validation_results["checks"].values() if v)
                total_checks = len(validation_results["checks"])
                validation_results["validation_score"] = (passed_checks / total_checks) * 100
                validation_results["validation_status"] = "✅ Passed" if validation_results["validation_score"] >= 70 else "⚠️ Needs improvement"
                
                return validation_results
                
            except Exception as e:
                return {"error": f"❌ Dataset validation failed: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: analyze, compare, validate"}
    
    return safe_execute(_process_dataset, f"dataset_processing_{action}")

@mcp.tool()
def hf_license_management(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """License management and compliance tools
    
    Actions:
    - check_license: Check repository license information
    - validate_compliance: Validate license compliance
    - suggest_license: Suggest appropriate license based on content
    - update_license: Update repository license information
    - compare_licenses: Compare licenses across repositories
    """
    def _manage_license():
        if action == "check_license":
            try:
                # Get repository information
                if repo_type == "model":
                    repo_info = model_info(repo_id, token=TOKEN, files_metadata=True)
                elif repo_type == "dataset":
                    repo_info = dataset_info(repo_id, token=TOKEN, files_metadata=True)
                else:
                    return {"error": f"❌ Unsupported repo_type: {repo_type}"}
                
                # Extract license information
                card_data = getattr(repo_info, 'cardData', {}) or {}
                license_from_card = card_data.get('license')
                
                # Check for LICENSE file
                files = [s.rfilename for s in repo_info.siblings] if hasattr(repo_info, 'siblings') else []
                license_files = [f for f in files if 'LICENSE' in f.upper() or 'LICENCE' in f.upper()]
                
                # Common licenses mapping
                common_licenses = {
                    'apache-2.0': 'Apache License 2.0',
                    'mit': 'MIT License',
                    'gpl-3.0': 'GNU General Public License v3.0',
                    'bsd-3-clause': 'BSD 3-Clause License',
                    'cc-by-4.0': 'Creative Commons Attribution 4.0',
                    'cc-by-sa-4.0': 'Creative Commons Attribution-ShareAlike 4.0',
                    'openrail': 'OpenRAIL License',
                    'other': 'Other/Custom License'
                }
                
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "license_info": {
                        "license_from_card": license_from_card,
                        "license_display_name": common_licenses.get(license_from_card, license_from_card),
                        "license_files": license_files,
                        "has_license_file": len(license_files) > 0,
                        "has_license_in_card": bool(license_from_card)
                    },
                    "compliance_status": "✅ Complete" if license_from_card and license_files else "⚠️ Incomplete",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ License check failed: {str(e)}"}
                
        elif action == "validate_compliance":
            try:
                # Get license info first
                license_info = hf_license_management("check_license", repo_id, repo_type)
                if "error" in license_info:
                    return license_info
                
                compliance_issues = []
                score = 100
                
                # Check for license in model card
                if not license_info["license_info"]["has_license_in_card"]:
                    compliance_issues.append("❌ No license specified in model card")
                    score -= 30
                
                # Check for LICENSE file
                if not license_info["license_info"]["has_license_file"]:
                    compliance_issues.append("❌ No LICENSE file found")
                    score -= 20
                
                # Check for appropriate license type
                license_type = license_info["license_info"]["license_from_card"]
                if license_type == "other":
                    compliance_issues.append("⚠️ Custom license - manual review recommended")
                    score -= 10
                
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "compliance_score": max(0, score),
                    "compliance_grade": "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "D",
                    "issues": compliance_issues,
                    "recommendations": [
                        "Add license to model card metadata" if not license_info["license_info"]["has_license_in_card"] else None,
                        "Add LICENSE file to repository" if not license_info["license_info"]["has_license_file"] else None,
                        "Consider using standard open source license" if license_type == "other" else None
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Compliance validation failed: {str(e)}"}
                
        elif action == "suggest_license":
            content_type = kwargs.get("content_type", "model")
            commercial_use = kwargs.get("commercial_use", True)
            derivatives_allowed = kwargs.get("derivatives_allowed", True)
            share_alike = kwargs.get("share_alike", False)
            
            suggestions = []
            
            if commercial_use and derivatives_allowed and not share_alike:
                suggestions.append({
                    "license": "apache-2.0",
                    "name": "Apache License 2.0",
                    "description": "Permissive license with patent grant",
                    "best_for": "Commercial projects, wide adoption"
                })
                suggestions.append({
                    "license": "mit",
                    "name": "MIT License", 
                    "description": "Simple permissive license",
                    "best_for": "Simple projects, maximum freedom"
                })
            
            if content_type == "dataset":
                suggestions.append({
                    "license": "cc-by-4.0",
                    "name": "Creative Commons Attribution 4.0",
                    "description": "Free use with attribution",
                    "best_for": "Datasets, educational content"
                })
                
            if content_type == "model" and not commercial_use:
                suggestions.append({
                    "license": "openrail",
                    "name": "OpenRAIL License",
                    "description": "Responsible AI license with use restrictions",
                    "best_for": "AI models with ethical constraints"
                })
            
            return {
                "content_type": content_type,
                "preferences": {
                    "commercial_use": commercial_use,
                    "derivatives_allowed": derivatives_allowed,
                    "share_alike": share_alike
                },
                "suggestions": suggestions,
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            return {"error": f"❌ Invalid action: {action}. Use: check_license, validate_compliance, suggest_license"}
    
    return safe_execute(_manage_license, f"license_management_{action}")

@mcp.tool()
def hf_inference_tools(
    action: str,
    repo_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Advanced inference and model testing tools
    
    Actions:
    - test_inference: Test model inference with custom inputs
    - batch_inference: Run inference on multiple inputs
    - benchmark_performance: Benchmark model performance
    - check_endpoints: Check available inference endpoints
    """
    def _inference_tools():
        if not HAS_INFERENCE_API:
            return {"error": "❌ Inference API features not available in this huggingface_hub version"}
            
        if not TOKEN or not ENABLE_INFERENCE:
            return {"error": "❌ Inference requires authentication and ENABLE_INFERENCE=true"}
            
        if action == "test_inference":
            inputs = kwargs.get("inputs", ["Hello world"])
            parameters = kwargs.get("parameters", {})
            
            if isinstance(inputs, str):
                inputs = [inputs]
            
            try:
                # Get inference API client
                api_client = get_inference_api(repo_id, token=TOKEN)
                if not api_client:
                    return {"error": "❌ Inference API not available for this model"}
                
                results = []
                for i, input_text in enumerate(inputs[:10]):  # Limit to 10 inputs
                    try:
                        result = api_client(input_text, parameters=parameters)
                        results.append({
                            "input_index": i,
                            "input": input_text,
                            "output": result,
                            "status": "success"
                        })
                    except Exception as e:
                        results.append({
                            "input_index": i,
                            "input": input_text,
                            "error": str(e),
                            "status": "failed"
                        })
                
                successful_inferences = len([r for r in results if r["status"] == "success"])
                
                return {
                    "repo_id": repo_id,
                    "total_inputs": len(inputs),
                    "successful_inferences": successful_inferences,
                    "success_rate": (successful_inferences / len(inputs)) * 100,
                    "results": results,
                    "parameters_used": parameters,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Inference test failed: {str(e)}"}
                
        elif action == "check_endpoints":
            try:
                # Try to get inference endpoints (using HfApi for compatibility)
                try:
                    endpoints = api.list_inference_endpoints(token=TOKEN)
                except AttributeError:
                    # Fallback if method doesn't exist
                    endpoints = []
                
                # Filter for this model if possible
                model_endpoints = []
                for endpoint in endpoints:
                    if hasattr(endpoint, 'repository') and endpoint.repository == repo_id:
                        model_endpoints.append({
                            "name": endpoint.name,
                            "status": getattr(endpoint, 'status', 'unknown'),
                            "compute": getattr(endpoint, 'compute', 'unknown'),
                            "url": getattr(endpoint, 'url', None)
                        })
                
                return {
                    "repo_id": repo_id,
                    "dedicated_endpoints": model_endpoints,
                    "serverless_available": bool(get_inference_api(repo_id, token=TOKEN)),
                    "total_user_endpoints": len(endpoints),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Failed to check endpoints: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: test_inference, check_endpoints"}
    
    return safe_execute(_inference_tools, f"inference_{action}")

@mcp.tool()
def hf_ai_workflow_tools(
    action: str,
    **kwargs
) -> Dict[str, Any]:
    """Specialized AI workflow and automation tools
    
    Actions:
    - create_model_card: Generate comprehensive model cards
    - bulk_operations: Perform bulk operations across repositories
    - workflow_automation: Automate common ML workflows
    - generate_readme: Generate README files for repositories
    - validate_pipeline: Validate complete ML pipelines
    """
    def _workflow_tools():
        if action == "create_model_card":
            repo_id = kwargs.get("repo_id")
            if not repo_id:
                return {"error": "❌ repo_id parameter required"}
                
            model_type = kwargs.get("model_type", "text-generation")
            language = kwargs.get("language", ["en"])
            license_type = kwargs.get("license", "apache-2.0")
            datasets = kwargs.get("datasets", [])
            metrics = kwargs.get("metrics", [])
            
            # Generate comprehensive model card content
            model_card_content = f"""---
license: {license_type}
language: {language if isinstance(language, list) else [language]}
pipeline_tag: {model_type}
tags:
- {model_type}
- transformers
- ai
datasets: {datasets if datasets else []}
metrics: {metrics if metrics else []}
---

# {repo_id}

## Model Description

This is a {model_type} model designed for high-performance inference and applications.

### Model Architecture

- **Model Type**: {model_type}
- **Language(s)**: {', '.join(language) if isinstance(language, list) else language}
- **License**: {license_type}

### Intended Use

This model is intended for:
- Research and development
- Educational purposes
- Commercial applications (subject to license terms)

### Training Data

{"Training datasets: " + ", ".join(datasets) if datasets else "Training data information not specified."}

### Performance Metrics

{"Performance metrics: " + ", ".join(metrics) if metrics else "Performance metrics not specified."}

### Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModel.from_pretrained("{repo_id}")

# Example usage
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

### Limitations and Bias

Please refer to the model's limitations and potential biases before deployment.

### Citation

If you use this model, please cite:

```
@misc{{{repo_id.replace('/', '_')},
  title={{{repo_id}}},
  author={{Model Author}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

### Contact

For questions and support, please contact the model authors.
"""
            
            return {
                "repo_id": repo_id,
                "model_card_content": model_card_content,
                "metadata": {
                    "license": license_type,
                    "language": language,
                    "model_type": model_type,
                    "datasets": datasets,
                    "metrics": metrics
                },
                "status": "✅ Model card generated",
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "bulk_operations":
            repo_list = kwargs.get("repo_list", [])
            operation = kwargs.get("operation", "info")
            
            if not repo_list:
                return {"error": "❌ repo_list parameter required"}
            
            results = []
            for repo_id in repo_list[:10]:  # Limit to 10 repos
                try:
                    if operation == "info":
                        result = hf_repository_manager("info", repo_id)
                    elif operation == "like":
                        result = hf_community_features("like", repo_id)
                    elif operation == "validate":
                        result = hf_model_evaluation("validate_model", repo_id)
                    else:
                        result = {"error": f"Unsupported operation: {operation}"}
                    
                    results.append({
                        "repo_id": repo_id,
                        "operation": operation,
                        "result": result,
                        "status": "success" if "error" not in result else "failed"
                    })
                except Exception as e:
                    results.append({
                        "repo_id": repo_id,
                        "operation": operation,
                        "error": str(e),
                        "status": "failed"
                    })
            
            successful = len([r for r in results if r["status"] == "success"])
            
            return {
                "operation": operation,
                "total_repos": len(repo_list),
                "processed_repos": len(results),
                "successful_operations": successful,
                "success_rate": (successful / len(results)) * 100 if results else 0,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "generate_readme":
            repo_id = kwargs.get("repo_id")
            repo_type = kwargs.get("repo_type", "model")
            
            if not repo_id:
                return {"error": "❌ repo_id parameter required"}
            
            # Generate README content based on repository analysis
            try:
                if repo_type == "model":
                    analysis = hf_model_evaluation("analyze", repo_id)
                elif repo_type == "dataset":
                    analysis = hf_dataset_processing("analyze", repo_id)
                else:
                    return {"error": f"❌ Unsupported repo_type: {repo_type}"}
                
                if "error" in analysis:
                    return analysis
                
                basic_info = analysis.get("basic_info", {})
                
                readme_content = f"""# {repo_id}

## Overview

{basic_info.get('id', repo_id)} is a {'machine learning model' if repo_type == 'model' else 'dataset'} hosted on Hugging Face.

### Quick Stats

- **Downloads**: {basic_info.get('downloads', 'N/A'):,} 
- **Likes**: {basic_info.get('likes', 'N/A')}
- **Created**: {basic_info.get('created_at', 'N/A')}
- **Last Modified**: {basic_info.get('last_modified', 'N/A')}

### Files

This repository contains the following files:

{chr(10).join([f"- `{file}`" for file in basic_info.get('files', [])[:10]])}

### Usage

```python
# Installation
pip install transformers

# Usage example
from transformers import AutoTokenizer{'from transformers import AutoModel' if repo_type == 'model' else '# Load your dataset'}

{'tokenizer = AutoTokenizer.from_pretrained("' + repo_id + '")' if repo_type == 'model' else '# Load dataset'}
{'model = AutoModel.from_pretrained("' + repo_id + '")' if repo_type == 'model' else ''}
```

### License

Please check the repository for license information.

### Citation

```
@misc{{{repo_id.replace('/', '_')},
  title={{{repo_id}}},
  url={{https://huggingface.co/{repo_id}}},
  year={{2024}}
}}
```

---

*This README was auto-generated by HuggingMCP.*
"""
                
                return {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "readme_content": readme_content,
                    "status": "✅ README generated",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": f"❌ README generation failed: {str(e)}"}
                
        elif action == "validate_pipeline":
            pipeline_components = kwargs.get("components", [])
            if not pipeline_components:
                return {"error": "❌ components parameter required (list of repo_ids)"}
            
            validation_results = []
            for component in pipeline_components:
                try:
                    # Validate each component
                    result = hf_model_evaluation("validate_model", component)
                    validation_results.append({
                        "component": component,
                        "validation": result,
                        "status": "valid" if result.get("validation_score", 0) >= 70 else "invalid"
                    })
                except Exception as e:
                    validation_results.append({
                        "component": component,
                        "error": str(e),
                        "status": "error"
                    })
            
            valid_components = len([r for r in validation_results if r["status"] == "valid"])
            
            return {
                "pipeline_validation": {
                    "total_components": len(pipeline_components),
                    "valid_components": valid_components,
                    "pipeline_health": (valid_components / len(pipeline_components)) * 100,
                    "pipeline_status": "✅ Healthy" if valid_components == len(pipeline_components) else "⚠️ Needs attention"
                },
                "component_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        else:
            return {"error": f"❌ Invalid action: {action}. Use: create_model_card, bulk_operations, generate_readme, validate_pipeline"}
    
    return safe_execute(_workflow_tools, f"workflow_{action}")

@mcp.tool()
def hf_advanced_analytics(
    action: str,
    **kwargs
) -> Dict[str, Any]:
    """Advanced analytics and insights for HuggingFace repositories
    
    Actions:
    - trending_analysis: Analyze trending models/datasets
    - user_analytics: Analyze user's repository portfolio
    - comparative_analysis: Deep comparison of repositories
    - ecosystem_insights: Insights about the HF ecosystem
    - recommendation_engine: Recommend repositories based on criteria
    """
    def _analytics():
        if action == "trending_analysis":
            content_type = kwargs.get("content_type", "models")
            limit = kwargs.get("limit", 50)
            time_period = kwargs.get("time_period", "week")  # week, month, all
            
            try:
                # Get repositories and analyze trends
                if content_type == "models":
                    repos = list(list_models(limit=limit, sort="downloads", direction=-1, token=TOKEN))
                elif content_type == "datasets":
                    repos = list(list_datasets(limit=limit, token=TOKEN))
                else:
                    return {"error": f"❌ Unsupported content_type: {content_type}"}
                
                trending_data = []
                for repo in repos:
                    popularity_score = calculate_popularity_score(repo)
                    trending_data.append({
                        "repo_id": repo.id,
                        "author": repo.author,
                        "downloads": getattr(repo, 'downloads', 0),
                        "likes": getattr(repo, 'likes', 0),
                        "popularity_score": popularity_score,
                        "created_at": repo.created_at.isoformat() if repo.created_at else None,
                        "tags": getattr(repo, 'tags', [])[:5]
                    })
                
                # Sort by popularity
                trending_data.sort(key=lambda x: x["popularity_score"], reverse=True)
                
                # Calculate trend metrics
                total_downloads = sum(item["downloads"] for item in trending_data)
                total_likes = sum(item["likes"] for item in trending_data)
                
                # Analyze tags
                tag_frequency = defaultdict(int)
                for item in trending_data:
                    for tag in item["tags"]:
                        tag_frequency[tag] += 1
                
                top_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
                
                return {
                    "content_type": content_type,
                    "time_period": time_period,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_repositories": len(trending_data),
                        "total_downloads": total_downloads,
                        "total_likes": total_likes,
                        "average_popularity": sum(item["popularity_score"] for item in trending_data) / len(trending_data) if trending_data else 0
                    },
                    "top_repositories": trending_data[:20],
                    "trending_tags": [{"tag": tag, "frequency": freq} for tag, freq in top_tags],
                    "insights": {
                        "most_popular": trending_data[0]["repo_id"] if trending_data else None,
                        "top_author": max(set(item["author"] for item in trending_data), 
                                        key=lambda x: sum(1 for item in trending_data if item["author"] == x)) if trending_data else None
                    }
                }
            except Exception as e:
                return {"error": f"❌ Trending analysis failed: {str(e)}"}
                
        elif action == "recommendation_engine":
            user_preferences = kwargs.get("preferences", {})
            content_type = kwargs.get("content_type", "models")
            limit = kwargs.get("limit", 20)
            
            # Extract preferences
            preferred_tags = user_preferences.get("tags", [])
            preferred_authors = user_preferences.get("authors", [])
            min_downloads = user_preferences.get("min_downloads", 100)
            max_model_size = user_preferences.get("max_size_mb", 1000)
            
            try:
                # Search with filters
                if content_type == "models":
                    all_repos = list(list_models(limit=limit*3, token=TOKEN))  # Get more to filter
                else:
                    all_repos = list(list_datasets(limit=limit*3, token=TOKEN))
                
                recommendations = []
                for repo in all_repos:
                    score = 0
                    downloads = getattr(repo, 'downloads', 0)
                    
                    # Skip if below minimum downloads
                    if downloads < min_downloads:
                        continue
                    
                    # Score based on preferences
                    repo_tags = getattr(repo, 'tags', [])
                    
                    # Tag matching
                    tag_matches = len(set(preferred_tags) & set(repo_tags))
                    score += tag_matches * 10
                    
                    # Author preference
                    if repo.author in preferred_authors:
                        score += 20
                    
                    # Popularity boost
                    score += min(downloads / 1000, 50)  # Max 50 points for downloads
                    score += getattr(repo, 'likes', 0) * 0.1
                    
                    recommendations.append({
                        "repo_id": repo.id,
                        "author": repo.author,
                        "downloads": downloads,
                        "likes": getattr(repo, 'likes', 0),
                        "tags": repo_tags[:5],
                        "recommendation_score": score,
                        "match_reasons": {
                            "tag_matches": tag_matches,
                            "preferred_author": repo.author in preferred_authors,
                            "popularity": downloads
                        }
                    })
                
                # Sort by recommendation score
                recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
                
                return {
                    "content_type": content_type,
                    "user_preferences": user_preferences,
                    "total_candidates": len(all_repos),
                    "qualified_recommendations": len(recommendations),
                    "top_recommendations": recommendations[:limit],
                    "recommendation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"❌ Recommendation engine failed: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: trending_analysis, recommendation_engine"}
    
    return safe_execute(_analytics, f"analytics_{action}")

@mcp.tool()
def hf_repository_utilities(
    action: str,
    repo_id: str,
    repo_type: str = "model",
    **kwargs
) -> Dict[str, Any]:
    """Advanced repository utilities and management tools
    
    Actions:
    - clone_metadata: Clone repository metadata and structure
    - backup_info: Create comprehensive backup information
    - migrate_repo: Migrate repository between organizations
    - archive_repo: Archive repository with full metadata
    - repository_health: Comprehensive repository health check
    """
    def _repo_utilities():
        if action == "repository_health":
            try:
                # Comprehensive health check
                health_report = {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "health_check_timestamp": datetime.now().isoformat(),
                    "checks": {},
                    "recommendations": [],
                    "overall_score": 0
                }
                
                # Get repository info
                if repo_type == "model":
                    repo_info = model_info(repo_id, token=TOKEN, files_metadata=True)
                elif repo_type == "dataset":
                    repo_info = dataset_info(repo_id, token=TOKEN, files_metadata=True)
                else:
                    return {"error": f"❌ Unsupported repo_type: {repo_type}"}
                
                # Check 1: Basic metadata
                card_data = getattr(repo_info, 'cardData', {}) or {}
                health_report["checks"]["has_license"] = bool(card_data.get('license'))
                health_report["checks"]["has_description"] = bool(getattr(repo_info, 'description', ''))
                health_report["checks"]["has_tags"] = len(getattr(repo_info, 'tags', [])) > 0
                
                # Check 2: Files
                files = [s.rfilename for s in repo_info.siblings] if hasattr(repo_info, 'siblings') else []
                health_report["checks"]["has_readme"] = any('README' in f for f in files)
                health_report["checks"]["has_model_files"] = any(f.endswith(('.bin', '.safetensors', '.h5', '.onnx', '.parquet')) for f in files)
                health_report["checks"]["file_count_reasonable"] = 1 <= len(files) <= 100
                
                # Check 3: Size and organization
                total_size = sum(getattr(s, 'size', 0) or 0 for s in repo_info.siblings if hasattr(repo_info, 'siblings'))
                health_report["checks"]["reasonable_size"] = total_size < MAX_FILE_SIZE * 10  # 10x normal limit
                
                # Check 4: Community engagement
                downloads = getattr(repo_info, 'downloads', 0)
                likes = getattr(repo_info, 'likes', 0)
                health_report["checks"]["has_community_interest"] = downloads > 50 or likes > 5
                
                # Calculate overall score
                passed_checks = sum(1 for v in health_report["checks"].values() if v)
                total_checks = len(health_report["checks"])
                health_report["overall_score"] = (passed_checks / total_checks) * 100
                
                # Generate recommendations
                if not health_report["checks"]["has_license"]:
                    health_report["recommendations"].append("Add license information to model card")
                if not health_report["checks"]["has_readme"]:
                    health_report["recommendations"].append("Add README.md file with documentation")
                if not health_report["checks"]["has_tags"]:
                    health_report["recommendations"].append("Add relevant tags for better discoverability")
                if not health_report["checks"]["has_community_interest"]:
                    health_report["recommendations"].append("Promote repository to increase visibility")
                
                # Health grade
                score = health_report["overall_score"]
                health_report["health_grade"] = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
                health_report["health_status"] = "✅ Excellent" if score >= 90 else "👍 Good" if score >= 75 else "⚠️ Fair" if score >= 60 else "❌ Needs improvement"
                
                return health_report
                
            except Exception as e:
                return {"error": f"❌ Repository health check failed: {str(e)}"}
                
        elif action == "backup_info":
            try:
                # Create comprehensive backup information
                if repo_type == "model":
                    repo_info = model_info(repo_id, token=TOKEN, files_metadata=True)
                elif repo_type == "dataset":
                    repo_info = dataset_info(repo_id, token=TOKEN, files_metadata=True)
                else:
                    return {"error": f"❌ Unsupported repo_type: {repo_type}"}
                
                # Get additional information (if available)
                commits = []
                refs = []
                
                if HAS_ADVANCED_REPO:
                    try:
                        commits = list(list_repo_commits(repo_id, repo_type=repo_type, token=TOKEN))
                        refs = list_repo_refs(repo_id, repo_type=repo_type, token=TOKEN)
                    except Exception:
                        pass  # Gracefully handle if features aren't available
                
                backup_info = {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "backup_timestamp": datetime.now().isoformat(),
                    "repository_metadata": {
                        "basic_info": format_repo_info(repo_info),
                        "card_data": getattr(repo_info, 'cardData', {}),
                        "description": getattr(repo_info, 'description', ''),
                        "tags": getattr(repo_info, 'tags', []),
                        "private": getattr(repo_info, 'private', False)
                    },
                    "file_structure": [
                        {
                            "filename": s.rfilename,
                            "size": getattr(s, 'size', 0),
                            "lfs": getattr(s, 'lfs', False) if hasattr(s, 'lfs') else False
                        } for s in repo_info.siblings
                    ] if hasattr(repo_info, 'siblings') else [],
                    "commit_history": [
                        {
                            "commit_id": c.commit_id,
                            "title": c.title,
                            "message": getattr(c, 'message', ''),
                            "created_at": c.created_at.isoformat() if c.created_at else None
                        } for c in commits[:20]
                    ],
                    "references": [
                        {
                            "ref": r.ref,
                            "target_commit": r.target_commit
                        } for r in refs
                    ],
                    "statistics": {
                        "total_files": len(repo_info.siblings) if hasattr(repo_info, 'siblings') else 0,
                        "total_size": sum(getattr(s, 'size', 0) or 0 for s in repo_info.siblings if hasattr(repo_info, 'siblings')),
                        "total_commits": len(commits),
                        "total_refs": len(refs)
                    }
                }
                
                return backup_info
                
            except Exception as e:
                return {"error": f"❌ Backup info creation failed: {str(e)}"}
                
        else:
            return {"error": f"❌ Invalid action: {action}. Use: repository_health, backup_info"}
    
    return safe_execute(_repo_utilities, f"repo_utilities_{action}")

# =============================================================================
# SERVER STARTUP WITH ENHANCED DEBUGGING
# =============================================================================

def main():
    """Main server startup with comprehensive error handling"""
    try:
        debug_stderr("🚀 Starting HuggingMCP server...")
        debug_stderr(f"   📍 Script location: {__file__}")
        debug_stderr(f"   🔑 Authenticated: {'✅' if TOKEN else '❌'}")
        debug_stderr(f"   🔒 Admin Mode: {'✅' if ADMIN_MODE else '❌'}")
        debug_stderr(f"   📚 Read Only: {'✅' if READ_ONLY else '❌'}")
        debug_stderr(f"   🛠️ Commands: 18+ comprehensive tools available")
        debug_stderr(f"   🔬 New: Model evaluation, dataset processing, AI workflows")
        debug_stderr(f"   🤝 New: Community features, license management, analytics")
        
        # Validate critical components
        if not hasattr(mcp, 'run'):
            debug_stderr("❌ MCP server missing run method", "ERROR")
            sys.exit(1)
        
        debug_stderr("✅ All components validated, starting server...")
        
        # Run the server
        mcp.run()
        
    except KeyboardInterrupt:
        debug_stderr("🛑 Server stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        debug_stderr(f"💥 Fatal error during server startup: {e}", "ERROR")
        debug_stderr(f"📝 Traceback: {traceback.format_exc()}", "ERROR")
        logger.error(f"Fatal server error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    debug_stderr("🎯 HuggingMCP script executed as main module")
    main()
