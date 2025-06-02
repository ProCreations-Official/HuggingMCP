#!/usr/bin/env python3
"""
HuggingMCP - Model Context Protocol Server for Hugging Face Hub

This MCP server allows Claude to interact with Hugging Face Hub:
- Create and manage repositories (models, datasets, spaces)
- Read and write files in repositories
- Manage collections
- Search and explore HF content
- Advanced file editing with precise text replacement

Author: Built for Claude and AI developers
License: MIT
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# MCP Server imports
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent

# Hugging Face imports
from huggingface_hub import (
    HfApi, 
    create_repo, 
    upload_file, 
    upload_folder,
    delete_repo,
    delete_file,
    list_models,
    list_datasets,
    list_spaces,
    model_info,
    dataset_info,
    space_info,
    hf_hub_download,
    snapshot_download,
    login,
    logout,
    whoami,
    create_collection,
    get_collection,
    add_collection_item,
    update_collection_item,
    delete_collection_item
)
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("HuggingMCP")

# Global configuration
class HuggingMCPConfig:
    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        self.read_only = os.getenv("HF_READ_ONLY", "false").lower() == "true"
        self.write_only = os.getenv("HF_WRITE_ONLY", "false").lower() == "true"
        self.admin_mode = os.getenv("HF_ADMIN_MODE", "false").lower() == "true"
        self.max_file_size = int(os.getenv("HF_MAX_FILE_SIZE", "50000000"))  # 50MB default
        
        # Initialize HF API
        self.api = HfApi(token=self.token) if self.token else HfApi()
        
        # Validate permissions
        if self.read_only and self.write_only:
            raise ValueError("Cannot set both HF_READ_ONLY and HF_WRITE_ONLY to true")

config = HuggingMCPConfig()

def permission_check(operation: str) -> bool:
    """Check if operation is allowed based on current permissions"""
    if operation == "read":
        return not config.write_only
    elif operation == "write":
        return not config.read_only
    elif operation == "admin":
        return config.admin_mode
    return True

def handle_hf_error(func):
    """Decorator to handle Hugging Face API errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as e:
            logger.error(f"HF API Error: {e}")
            return {"error": f"Hugging Face API Error: {str(e)}"}
        except RepositoryNotFoundError as e:
            logger.error(f"Repository not found: {e}")
            return {"error": f"Repository not found: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    return wrapper

# =============================================================================
# AUTHENTICATION & USER MANAGEMENT
# =============================================================================

@mcp.tool()
@handle_hf_error
def hf_login(token: str) -> Dict[str, Any]:
    """
    Login to Hugging Face with a token
    
    Args:
        token: Your Hugging Face access token
        
    Returns:
        Login status and user information
    """
    try:
        login(token=token)
        user_info = whoami(token=token)
        config.token = token
        config.api = HfApi(token=token)
        return {
            "status": "success",
            "message": "Successfully logged in to Hugging Face",
            "user": user_info
        }
    except Exception as e:
        return {"error": f"Login failed: {str(e)}"}

@mcp.tool()
@handle_hf_error
def hf_whoami() -> Dict[str, Any]:
    """
    Get current user information
    
    Returns:
        Current user details
    """
    if not config.token:
        return {"error": "Not logged in. Please use hf_login first."}
    
    user_info = whoami(token=config.token)
    return {"user": user_info}

@mcp.tool()
@handle_hf_error
def hf_logout() -> Dict[str, str]:
    """
    Logout from Hugging Face
    
    Returns:
        Logout status
    """
    logout()
    config.token = None
    config.api = HfApi()
    return {"status": "success", "message": "Successfully logged out"}

# =============================================================================
# REPOSITORY MANAGEMENT
# =============================================================================

@mcp.tool()
@handle_hf_error
def hf_create_repository(
    repo_id: str,
    repo_type: str = "model",
    private: bool = False,
    description: Optional[str] = None,
    license: Optional[str] = None,
    readme_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new repository on Hugging Face Hub
    
    Args:
        repo_id: Repository ID (username/repo-name)
        repo_type: Type of repository ("model", "dataset", "space")
        private: Whether the repository should be private
        description: Repository description
        license: Repository license
        readme_content: Custom README content
        
    Returns:
        Repository creation details
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required. Please login first."}
    
    repo_url = create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        token=config.token
    )
    
    result = {
        "status": "success",
        "repo_url": repo_url,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "private": private
    }
    
    # Add README if provided
    if readme_content:
        upload_result = upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            token=config.token,
            commit_message="Initial README"
        )
        result["readme_uploaded"] = True
    
    return result

@mcp.tool()
@handle_hf_error
def hf_delete_repository(repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
    """
    Delete a repository (DANGEROUS - USE WITH CAUTION!)
    
    Args:
        repo_id: Repository ID to delete
        repo_type: Type of repository ("model", "dataset", "space")
        
    Returns:
        Deletion status
    """
    if not permission_check("admin"):
        return {"error": "Admin permissions required for repository deletion"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    delete_repo(repo_id=repo_id, repo_type=repo_type, token=config.token)
    return {
        "status": "success",
        "message": f"Repository {repo_id} deleted successfully",
        "repo_id": repo_id,
        "repo_type": repo_type
    }

@mcp.tool()
@handle_hf_error
def hf_get_repository_info(repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
    """
    Get detailed information about a repository
    
    Args:
        repo_id: Repository ID
        repo_type: Type of repository ("model", "dataset", "space")
        
    Returns:
        Repository information and metadata
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    if repo_type == "model":
        info = model_info(repo_id, token=config.token)
    elif repo_type == "dataset":
        info = dataset_info(repo_id, token=config.token)
    elif repo_type == "space":
        info = space_info(repo_id, token=config.token)
    else:
        return {"error": f"Unknown repo_type: {repo_type}"}
    
    return {
        "repo_id": info.id,
        "author": info.author,
        "sha": info.sha,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "private": info.private,
        "downloads": getattr(info, 'downloads', 0),
        "likes": getattr(info, 'likes', 0),
        "tags": getattr(info, 'tags', []),
        "siblings": [{"filename": s.rfilename, "size": s.size} for s in info.siblings] if hasattr(info, 'siblings') else []
    }

@mcp.tool()
@handle_hf_error
def list_repository_files(repo_id: str, repo_type: str = "model", path: str = "") -> Dict[str, Any]:
    """
    List files in a repository
    
    Args:
        repo_id: Repository ID
        repo_type: Type of repository
        path: Path within the repository
        
    Returns:
        List of files and directories
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    files = config.api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        token=config.token
    )
    
    if path:
        files = [f for f in files if f.startswith(path)]
    
    return {
        "repo_id": repo_id,
        "path": path,
        "files": files
    }

# =============================================================================
# FILE OPERATIONS
# =============================================================================

@mcp.tool()
@handle_hf_error
def read_file(
    repo_id: str, 
    filename: str, 
    repo_type: str = "model",
    revision: str = "main"
) -> Dict[str, Any]:
    """
    Read a file from a Hugging Face repository
    
    Args:
        repo_id: Repository ID
        filename: Path to file in repository
        repo_type: Type of repository
        revision: Git revision/branch
        
    Returns:
        File content and metadata
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=config.token
        )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_size = os.path.getsize(file_path)
        
        return {
            "repo_id": repo_id,
            "filename": filename,
            "content": content,
            "size": file_size,
            "revision": revision
        }
    except UnicodeDecodeError:
        # Try reading as binary for non-text files
        with open(file_path, 'rb') as f:
            content = f.read()
        
        return {
            "repo_id": repo_id,
            "filename": filename,
            "content": f"<Binary file: {len(content)} bytes>",
            "size": len(content),
            "revision": revision,
            "is_binary": True
        }

@mcp.tool()
@handle_hf_error
def write_file(
    repo_id: str,
    filename: str,
    content: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Write/upload a file to a Hugging Face repository
    
    Args:
        repo_id: Repository ID
        filename: Path where to save the file
        content: File content to write
        repo_type: Type of repository
        commit_message: Commit message
        
    Returns:
        Upload status and details
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    if len(content.encode()) > config.max_file_size:
        return {"error": f"File too large. Maximum size: {config.max_file_size} bytes"}
    
    if not commit_message:
        commit_message = f"Update {filename}"
    
    commit_url = upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        token=config.token,
        commit_message=commit_message
    )
    
    return {
        "status": "success",
        "repo_id": repo_id,
        "filename": filename,
        "commit_url": commit_url,
        "commit_message": commit_message,
        "size": len(content.encode())
    }

@mcp.tool()
@handle_hf_error
def edit_file(
    repo_id: str,
    filename: str,
    old_text: str,
    new_text: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edit a file by replacing specific text content (PRECISE EDITING)
    
    Args:
        repo_id: Repository ID
        filename: Path to file to edit
        old_text: Exact text to find and replace (must match exactly)
        new_text: Text to replace it with
        repo_type: Type of repository
        commit_message: Commit message
        
    Returns:
        Edit status and details
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    # First, read the current file content
    read_result = read_file(repo_id, filename, repo_type)
    if "error" in read_result:
        return read_result
    
    current_content = read_result["content"]
    
    # Check if old_text exists in the file
    if old_text not in current_content:
        return {
            "error": f"Text not found in file. Could not locate: {old_text[:100]}{'...' if len(old_text) > 100 else ''}"
        }
    
    # Perform the replacement
    new_content = current_content.replace(old_text, new_text, 1)  # Replace only first occurrence
    
    if not commit_message:
        commit_message = f"Edit {filename}: Replace text"
    
    # Write the updated content
    write_result = write_file(repo_id, filename, new_content, repo_type, commit_message)
    
    if "error" in write_result:
        return write_result
    
    return {
        "status": "success",
        "repo_id": repo_id,
        "filename": filename,
        "old_text_length": len(old_text),
        "new_text_length": len(new_text),
        "total_size": len(new_content.encode()),
        "commit_url": write_result["commit_url"],
        "commit_message": commit_message
    }

@mcp.tool()
@handle_hf_error
def delete_file_from_repo(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a file from a repository
    
    Args:
        repo_id: Repository ID
        filename: Path to file to delete
        repo_type: Type of repository
        commit_message: Commit message
        
    Returns:
        Deletion status
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    if not commit_message:
        commit_message = f"Delete {filename}"
    
    delete_file(
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        token=config.token,
        commit_message=commit_message
    )
    
    return {
        "status": "success",
        "repo_id": repo_id,
        "filename": filename,
        "commit_message": commit_message
    }

# =============================================================================
# SEARCH & DISCOVERY
# =============================================================================

@mcp.tool()
@handle_hf_error
def search_models(
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    sort: str = "downloads",
    direction: int = -1,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for models on Hugging Face Hub
    
    Args:
        query: Search query
        author: Filter by author
        filter_tag: Filter by tag (e.g., "pytorch", "text-classification")
        sort: Sort by ("downloads", "created_at", "last_modified")
        direction: Sort direction (-1 for desc, 1 for asc)
        limit: Maximum results
        
    Returns:
        List of matching models
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    models = list_models(
        search=query,
        author=author,
        filter=filter_tag,
        sort=sort,
        direction=direction,
        limit=limit,
        token=config.token
    )
    
    results = []
    for model in models:
        results.append({
            "id": model.id,
            "author": model.author,
            "downloads": model.downloads,
            "likes": model.likes,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "last_modified": model.last_modified.isoformat() if model.last_modified else None,
            "tags": model.tags,
            "private": model.private
        })
    
    return {
        "query": query,
        "total_results": len(results),
        "models": results
    }

@mcp.tool()
@handle_hf_error
def search_datasets(
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    sort: str = "downloads",
    direction: int = -1,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for datasets on Hugging Face Hub
    
    Args:
        query: Search query
        author: Filter by author
        filter_tag: Filter by tag
        sort: Sort by ("downloads", "created_at", "last_modified")
        direction: Sort direction
        limit: Maximum results
        
    Returns:
        List of matching datasets
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    datasets = list_datasets(
        search=query,
        author=author,
        filter=filter_tag,
        sort=sort,
        direction=direction,
        limit=limit,
        token=config.token
    )
    
    results = []
    for dataset in datasets:
        results.append({
            "id": dataset.id,
            "author": dataset.author,
            "downloads": dataset.downloads,
            "likes": dataset.likes,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "last_modified": dataset.last_modified.isoformat() if dataset.last_modified else None,
            "tags": dataset.tags,
            "private": dataset.private
        })
    
    return {
        "query": query,
        "total_results": len(results),
        "datasets": results
    }

@mcp.tool()
@handle_hf_error
def search_spaces(
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    sort: str = "created_at",
    direction: int = -1,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for Spaces on Hugging Face Hub
    
    Args:
        query: Search query
        author: Filter by author
        filter_tag: Filter by tag
        sort: Sort by
        direction: Sort direction
        limit: Maximum results
        
    Returns:
        List of matching Spaces
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    spaces = list_spaces(
        search=query,
        author=author,
        filter=filter_tag,
        sort=sort,
        direction=direction,
        limit=limit,
        token=config.token
    )
    
    results = []
    for space in spaces:
        results.append({
            "id": space.id,
            "author": space.author,
            "likes": space.likes,
            "created_at": space.created_at.isoformat() if space.created_at else None,
            "last_modified": space.last_modified.isoformat() if space.last_modified else None,
            "tags": space.tags,
            "private": space.private,
            "sdk": getattr(space, 'sdk', None),
            "runtime": getattr(space, 'runtime', None)
        })
    
    return {
        "query": query,
        "total_results": len(results),
        "spaces": results
    }

# =============================================================================
# COLLECTIONS MANAGEMENT
# =============================================================================

@mcp.tool()
@handle_hf_error
def create_hf_collection(
    title: str,
    namespace: str,
    description: Optional[str] = None,
    private: bool = False
) -> Dict[str, Any]:
    """
    Create a new collection
    
    Args:
        title: Collection title
        namespace: Namespace (username or org)
        description: Collection description
        private: Whether collection is private
        
    Returns:
        Collection creation details
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    collection = create_collection(
        title=title,
        namespace=namespace,
        description=description,
        private=private,
        token=config.token
    )
    
    return {
        "status": "success",
        "collection_slug": collection.slug,
        "title": collection.title,
        "namespace": collection.namespace,
        "url": f"https://huggingface.co/collections/{namespace}/{collection.slug}"
    }

@mcp.tool()
@handle_hf_error
def add_to_collection(
    collection_slug: str,
    item_id: str,
    item_type: str,
    note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add an item to a collection
    
    Args:
        collection_slug: Collection slug (namespace/slug)
        item_id: Repository ID to add
        item_type: Type of item ("model", "dataset", "space", "paper")
        note: Optional note about the item
        
    Returns:
        Addition status
    """
    if not permission_check("write"):
        return {"error": "Write operations not permitted"}
    
    if not config.token:
        return {"error": "Authentication required"}
    
    add_collection_item(
        collection_slug=collection_slug,
        item_id=item_id,
        item_type=item_type,
        note=note,
        token=config.token
    )
    
    return {
        "status": "success",
        "collection_slug": collection_slug,
        "item_id": item_id,
        "item_type": item_type,
        "note": note
    }

@mcp.tool()
@handle_hf_error
def get_collection_info(collection_slug: str) -> Dict[str, Any]:
    """
    Get information about a collection
    
    Args:
        collection_slug: Collection slug (namespace/slug)
        
    Returns:
        Collection details and items
    """
    if not permission_check("read"):
        return {"error": "Read operations not permitted"}
    
    collection = get_collection(collection_slug, token=config.token)
    
    items = []
    for item in collection.items:
        items.append({
            "item_object_id": item.item_object_id,
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

# =============================================================================
# CONFIGURATION & UTILITIES
# =============================================================================

@mcp.tool()
def get_hf_config() -> Dict[str, Any]:
    """
    Get current HuggingMCP configuration
    
    Returns:
        Current configuration settings
    """
    return {
        "authenticated": bool(config.token),
        "read_only": config.read_only,
        "write_only": config.write_only,
        "admin_mode": config.admin_mode,
        "max_file_size": config.max_file_size,
        "permissions": {
            "read": permission_check("read"),
            "write": permission_check("write"),
            "admin": permission_check("admin")
        }
    }

@mcp.tool()
def set_hf_permissions(
    read_only: Optional[bool] = None,
    write_only: Optional[bool] = None,
    admin_mode: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Update HuggingMCP permissions
    
    Args:
        read_only: Enable read-only mode
        write_only: Enable write-only mode  
        admin_mode: Enable admin mode
        
    Returns:
        Updated configuration
    """
    if read_only is not None:
        config.read_only = read_only
    if write_only is not None:
        config.write_only = write_only
    if admin_mode is not None:
        config.admin_mode = admin_mode
    
    if config.read_only and config.write_only:
        config.write_only = False  # Prioritize read_only
    
    return get_hf_config()

# =============================================================================
# MAIN SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Print startup information
    print("🤗 HuggingMCP Server Starting...")
    print(f"   Read Only: {config.read_only}")
    print(f"   Write Only: {config.write_only}")
    print(f"   Admin Mode: {config.admin_mode}")
    print(f"   Authenticated: {bool(config.token)}")
    print("   Ready for MCP connections! 🚀")
    
    # Run the MCP server
    mcp.run()
