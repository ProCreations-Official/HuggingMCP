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
    debug_stderr("✅ Hugging Face Hub imported successfully")
except ImportError as e:
    debug_stderr(f"❌ Hugging Face Hub import failed: {e}", "ERROR")
    logger.error(f"huggingface_hub not installed. Run: pip3 install huggingface_hub - Error: {e}")
    sys.exit(1)

# Initialize MCP server with metadata
try:
    mcp = FastMCP(
        "HuggingMCP",
        description="Enhanced Hugging Face MCP Server with 11 optimized commands"
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
        return {"error": f"❌ Authentication required for {operation} - set HF_TOKEN environment variable"}
    return None

def validate_permissions(require_write: bool = False, require_admin: bool = False) -> Optional[Dict[str, Any]]:
    """Validate permissions for operations"""
    if require_write and READ_ONLY:
        return {"error": "❌ Read-only mode - write operations not allowed"}
    if require_admin and not ADMIN_MODE:
        return {"error": "❌ Admin mode required for this operation"}
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
            "version": "2.0.0",
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
                "pull_requests": not READ_ONLY
            },
            "commands_available": 10,
            "debug_info": {
                "working_directory": os.getcwd(),
                "environment_vars": {
                    "HF_TOKEN": "✓" if TOKEN else "✗",
                    "HF_READ_ONLY": str(READ_ONLY),
                    "HF_ADMIN_MODE": str(ADMIN_MODE)
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
    """Comprehensive file operations: read, write, edit, delete
    
    Actions:
    - read: Read file content (supports max_size, chunked reading)
    - write: Write/upload file content
    - edit: Edit file by replacing text
    - delete: Delete file from repository
    """
    def _handle_file_operation():
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
        
        else:
            return {"error": f"❌ Invalid action: {action}. Use: read, write, edit, delete"}
    
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
        debug_stderr(f"   🛠️ Commands: 11 optimized tools available")
        
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
