#!/usr/bin/env python3
"""
HuggingMCP - Model Context Protocol server for HuggingFace integration

Provides Claude with comprehensive access to HuggingFace functionality including:
- Creating and managing spaces, models, datasets, and collections
- Reading and editing files in repositories
- Searching and exploring HuggingFace content
- Managing repository settings and permissions

Author: ProCreations-Official
License: MIT
"""

import json
import os
import sys
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from io import BytesIO

# MCP server imports
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# HuggingFace imports
from huggingface_hub import HfApi, list_models, list_datasets, list_spaces
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub import hf_hub_download, upload_file, create_repo, delete_repo
from huggingface_hub import login, logout, whoami, create_collection, get_collection
from huggingface_hub.utils import EntryNotFoundError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("HuggingMCP")

# Global HuggingFace API client
hf_api = None

# Permission levels
PERMISSION_LEVELS = {
    "read": ["read", "search", "list", "get", "download"],
    "write": ["read", "search", "list", "get", "download", "create", "edit", "upload"],
    "admin": ["read", "search", "list", "get", "download", "create", "edit", "upload", "delete", "manage"]
}

def check_permission(action: str, allowed_level: str = "admin") -> bool:
    """Check if an action is allowed based on permission level"""
    return action in PERMISSION_LEVELS.get(allowed_level, [])

def get_permission_level() -> str:
    """Get the current permission level from environment variable"""
    return os.getenv("HF_MCP_PERMISSIONS", "admin").lower()

def init_hf_api():
    """Initialize HuggingFace API with token"""
    global hf_api
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
    
    try:
        hf_api = HfApi(token=token)
        # Test the token
        hf_api.whoami()
        logger.info("HuggingFace API initialized successfully")
    except Exception as e:
        raise ValueError(f"Failed to initialize HuggingFace API: {e}")

@mcp.tool()
async def hf_whoami() -> str:
    """Get information about the current HuggingFace user"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        user_info = hf_api.whoami()
        return json.dumps(user_info, indent=2)
    except Exception as e:
        return f"Error getting user info: {e}"

@mcp.tool()
async def hf_search_models(
    query: str = "",
    author: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20
) -> str:
    """Search for models on HuggingFace Hub"""
    if not check_permission("search", get_permission_level()):
        return "Permission denied: search access required"
    
    try:
        models = list_models(
            search=query,
            author=author,
            tags=tags.split(",") if tags else None,
            limit=limit
        )
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "author": getattr(model, 'author', None),
                "downloads": getattr(model, 'downloads', 0),
                "likes": getattr(model, 'likes', 0),
                "tags": getattr(model, 'tags', []),
                "created_at": str(getattr(model, 'created_at', '')),
                "last_modified": str(getattr(model, 'last_modified', ''))
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching models: {e}"

@mcp.tool()
async def hf_search_datasets(
    query: str = "",
    author: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20
) -> str:
    """Search for datasets on HuggingFace Hub"""
    if not check_permission("search", get_permission_level()):
        return "Permission denied: search access required"
    
    try:
        datasets = list_datasets(
            search=query,
            author=author,
            tags=tags.split(",") if tags else None,
            limit=limit
        )
        
        results = []
        for dataset in datasets:
            results.append({
                "id": dataset.id,
                "author": getattr(dataset, 'author', None),
                "downloads": getattr(dataset, 'downloads', 0),
                "likes": getattr(dataset, 'likes', 0),
                "tags": getattr(dataset, 'tags', []),
                "created_at": str(getattr(dataset, 'created_at', '')),
                "last_modified": str(getattr(dataset, 'last_modified', ''))
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching datasets: {e}"

@mcp.tool()
async def hf_search_spaces(
    query: str = "",
    author: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20
) -> str:
    """Search for spaces on HuggingFace Hub"""
    if not check_permission("search", get_permission_level()):
        return "Permission denied: search access required"
    
    try:
        spaces = list_spaces(
            search=query,
            author=author,
            tags=tags.split(",") if tags else None,
            limit=limit
        )
        
        results = []
        for space in spaces:
            results.append({
                "id": space.id,
                "author": getattr(space, 'author', None),
                "likes": getattr(space, 'likes', 0),
                "tags": getattr(space, 'tags', []),
                "sdk": getattr(space, 'sdk', None),
                "created_at": str(getattr(space, 'created_at', '')),
                "last_modified": str(getattr(space, 'last_modified', ''))
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching spaces: {e}"

@mcp.tool()
async def hf_get_repo_info(repo_id: str, repo_type: str = "model") -> str:
    """Get detailed information about a repository (model, dataset, or space)"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        info = hf_api.repo_info(repo_id=repo_id, repo_type=repo_type)
        
        result = {
            "id": info.id,
            "author": getattr(info, 'author', None),
            "sha": info.sha,
            "created_at": str(info.created_at),
            "last_modified": str(info.last_modified),
            "private": getattr(info, 'private', False),
            "downloads": getattr(info, 'downloads', 0),
            "likes": getattr(info, 'likes', 0),
            "tags": getattr(info, 'tags', []),
            "siblings": [{"rfilename": s.rfilename, "size": s.size} for s in info.siblings[:50]]  # Limit for readability
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting repo info: {e}"

@mcp.tool()
async def hf_list_repo_files(repo_id: str, repo_type: str = "model", path: str = "") -> str:
    """List files in a repository"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        files = hf_api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision="main")
        
        # Filter by path if specified
        if path:
            files = [f for f in files if f.startswith(path)]
        
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error listing repo files: {e}"

@mcp.tool()
async def hf_read_file(repo_id: str, file_path: str, repo_type: str = "model") -> str:
    """Read a file from a HuggingFace repository"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        # Download file content
        file_content = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type=repo_type
        )
        
        # Read the content
        with open(file_content, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
async def hf_create_repo(
    repo_id: str,
    repo_type: str = "model",
    private: bool = False,
    license: Optional[str] = None,
    space_sdk: Optional[str] = None
) -> str:
    """Create a new repository (model, dataset, or space)"""
    if not check_permission("create", get_permission_level()):
        return "Permission denied: create access required"
    
    try:
        kwargs = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private
        }
        
        if license:
            kwargs["license"] = license
        
        if repo_type == "space" and space_sdk:
            kwargs["space_sdk"] = space_sdk
        
        url = create_repo(**kwargs)
        return f"Repository created successfully: {url}"
    except Exception as e:
        return f"Error creating repository: {e}"

@mcp.tool()
async def hf_upload_file(
    repo_id: str,
    file_path: str,
    file_content: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> str:
    """Upload a file to a HuggingFace repository"""
    if not check_permission("upload", get_permission_level()):
        return "Permission denied: upload access required"
    
    try:
        # Create a temporary file with the content
        temp_file = BytesIO(file_content.encode('utf-8'))
        
        result = upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {file_path}"
        )
        
        return f"File uploaded successfully: {result}"
    except Exception as e:
        return f"Error uploading file: {e}"

@mcp.tool()
async def hf_edit_file(
    repo_id: str,
    file_path: str,
    old_text: str,
    new_text: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> str:
    """Edit a file in a HuggingFace repository by replacing old_text with new_text"""
    if not check_permission("edit", get_permission_level()):
        return "Permission denied: edit access required"
    
    try:
        # First, read the current file content
        current_content = await hf_read_file(repo_id, file_path, repo_type)
        if current_content.startswith("Error"):
            return current_content
        
        # Replace the old text with new text
        if old_text not in current_content:
            return f"Error: old_text not found in file. Current content length: {len(current_content)} chars"
        
        new_content = current_content.replace(old_text, new_text)
        
        # Upload the modified content
        result = await hf_upload_file(
            repo_id=repo_id,
            file_path=file_path,
            file_content=new_content,
            repo_type=repo_type,
            commit_message=commit_message or f"Edit {file_path}"
        )
        
        return result
    except Exception as e:
        return f"Error editing file: {e}"

@mcp.tool()
async def hf_delete_file(
    repo_id: str,
    file_path: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> str:
    """Delete a file from a HuggingFace repository"""
    if not check_permission("delete", get_permission_level()):
        return "Permission denied: delete access required"
    
    try:
        hf_api.delete_file(
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Delete {file_path}"
        )
        return f"File {file_path} deleted successfully"
    except Exception as e:
        return f"Error deleting file: {e}"

@mcp.tool()
async def hf_create_collection(
    title: str,
    namespace: str,
    description: Optional[str] = None,
    private: bool = False
) -> str:
    """Create a new collection"""
    if not check_permission("create", get_permission_level()):
        return "Permission denied: create access required"
    
    try:
        collection = create_collection(
            title=title,
            namespace=namespace,
            description=description,
            private=private
        )
        return f"Collection created successfully: {collection.slug}"
    except Exception as e:
        return f"Error creating collection: {e}"

@mcp.tool()
async def hf_get_collection(collection_slug: str) -> str:
    """Get information about a collection"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        collection = get_collection(collection_slug)
        
        result = {
            "slug": collection.slug,
            "title": collection.title,
            "description": collection.description,
            "private": collection.private,
            "author": collection.author,
            "items": [
                {
                    "item_object_id": item.item_object_id,
                    "item_id": item.item_id,
                    "item_type": item.item_type,
                    "position": item.position,
                    "note": item.note
                }
                for item in collection.items[:20]  # Limit for readability
            ]
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting collection: {e}"

@mcp.tool()
async def hf_add_to_collection(
    collection_slug: str,
    item_id: str,
    item_type: str,
    note: Optional[str] = None
) -> str:
    """Add an item to a collection"""
    if not check_permission("edit", get_permission_level()):
        return "Permission denied: edit access required"
    
    try:
        hf_api.add_collection_item(
            collection_slug=collection_slug,
            item_id=item_id,
            item_type=item_type,
            note=note
        )
        return f"Item {item_id} added to collection {collection_slug}"
    except Exception as e:
        return f"Error adding to collection: {e}"

@mcp.tool()
async def hf_update_repo_settings(
    repo_id: str,
    repo_type: str = "model",
    private: Optional[bool] = None,
    description: Optional[str] = None
) -> str:
    """Update repository settings"""
    if not check_permission("manage", get_permission_level()):
        return "Permission denied: manage access required"
    
    try:
        kwargs = {"repo_id": repo_id, "repo_type": repo_type}
        if private is not None:
            kwargs["private"] = private
        if description is not None:
            kwargs["description"] = description
        
        hf_api.update_repo_settings(**kwargs)
        return f"Repository settings updated successfully"
    except Exception as e:
        return f"Error updating repo settings: {e}"

@mcp.tool()
async def hf_delete_repo(repo_id: str, repo_type: str = "model") -> str:
    """Delete a repository (DANGEROUS!)"""
    if not check_permission("delete", get_permission_level()):
        return "Permission denied: delete access required"
    
    try:
        delete_repo(repo_id=repo_id, repo_type=repo_type)
        return f"Repository {repo_id} deleted successfully"
    except Exception as e:
        return f"Error deleting repository: {e}"

@mcp.tool()
async def hf_create_space(
    repo_id: str,
    sdk: str = "gradio",
    private: bool = False,
    hardware: Optional[str] = None
) -> str:
    """Create a new HuggingFace Space"""
    if not check_permission("create", get_permission_level()):
        return "Permission denied: create access required"
    
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            private=private,
            space_sdk=sdk
        )
        
        # Set hardware if specified
        if hardware:
            try:
                hf_api.request_space_hardware(repo_id=repo_id, hardware=hardware)
            except Exception as hw_error:
                logger.warning(f"Could not set hardware: {hw_error}")
        
        return f"Space created successfully: {url}"
    except Exception as e:
        return f"Error creating space: {e}"

@mcp.tool()
async def hf_list_user_repos(repo_type: str = "model", limit: int = 50) -> str:
    """List repositories owned by the current user"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        user_info = hf_api.whoami()
        username = user_info["name"]
        
        if repo_type == "model":
            repos = list_models(author=username, limit=limit)
        elif repo_type == "dataset":
            repos = list_datasets(author=username, limit=limit)
        elif repo_type == "space":
            repos = list_spaces(author=username, limit=limit)
        else:
            return "Error: repo_type must be 'model', 'dataset', or 'space'"
        
        results = []
        for repo in repos:
            results.append({
                "id": repo.id,
                "downloads": getattr(repo, 'downloads', 0),
                "likes": getattr(repo, 'likes', 0),
                "tags": getattr(repo, 'tags', []),
                "last_modified": str(getattr(repo, 'last_modified', ''))
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error listing user repos: {e}"

@mcp.tool()
async def hf_get_download_stats(repo_id: str, repo_type: str = "model") -> str:
    """Get download statistics for a repository"""
    if not check_permission("read", get_permission_level()):
        return "Permission denied: read access required"
    
    try:
        info = hf_api.repo_info(repo_id=repo_id, repo_type=repo_type)
        
        stats = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "downloads": getattr(info, 'downloads', 0),
            "likes": getattr(info, 'likes', 0),
            "created_at": str(info.created_at),
            "last_modified": str(info.last_modified)
        }
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error getting download stats: {e}"

async def main():
    """Main function to run the MCP server"""
    # Initialize HuggingFace API
    try:
        init_hf_api()
    except ValueError as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Get permission level
    perm_level = get_permission_level()
    logger.info(f"Starting HuggingMCP server with permission level: {perm_level}")
    
    # Run the MCP server
    async with mcp.run_server() as (read_stream, write_stream):
        await Server(mcp).run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="HuggingMCP",
                server_version="1.0.0",
                capabilities=mcp.get_capabilities()
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
