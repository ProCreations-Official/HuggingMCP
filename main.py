#!/usr/bin/env python3
"""
HuggingMCP - Clean Hugging Face MCP Server
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCP
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.error("MCP not installed. Run: pip3 install 'mcp[cli]'")
    sys.exit(1)

# Import Hugging Face
try:
    from huggingface_hub import (
        HfApi, create_repo, upload_file, delete_repo, delete_file,
        list_models, list_datasets, list_spaces, model_info, dataset_info,
        hf_hub_download, login, logout, whoami, create_collection,
        get_collection, add_collection_item, create_discussion,
        get_repo_discussions, get_discussion_details
    )
except ImportError:
    logger.error("huggingface_hub not installed. Run: pip3 install huggingface_hub")
    sys.exit(1)

# Initialize MCP server
mcp = FastMCP("HuggingMCP")

# Configuration
TOKEN = os.getenv("HF_TOKEN")
READ_ONLY = os.getenv("HF_READ_ONLY", "false").lower() == "true"
ADMIN_MODE = os.getenv("HF_ADMIN_MODE", "false").lower() == "true"

# Initialize API
api = HfApi(token=TOKEN) if TOKEN else HfApi()

logger.info(f"🤗 HuggingMCP initialized - Token: {'✓' if TOKEN else '✗'}, Admin: {ADMIN_MODE}")

# =============================================================================
# CONFIGURATION TOOLS
# =============================================================================

@mcp.tool()
def hf_test() -> Dict[str, Any]:
    """Test HuggingMCP server functionality"""
    return {
        "status": "✅ HuggingMCP is working!",
        "authenticated": bool(TOKEN),
        "admin_mode": ADMIN_MODE,
        "read_only": READ_ONLY,
        "tools_registered": "All tools loaded successfully! 🎉"
    }

@mcp.tool()
def get_hf_config() -> Dict[str, Any]:
    """Get current HuggingMCP configuration"""
    return {
        "authenticated": bool(TOKEN),
        "admin_mode": ADMIN_MODE,
        "read_only": READ_ONLY,
        "server_version": "1.0.0",
        "capabilities": {
            "create_repos": not READ_ONLY,
            "delete_repos": ADMIN_MODE,
            "read_files": True,
            "write_files": not READ_ONLY,
            "search": True,
            "collections": not READ_ONLY
        }
    }

# =============================================================================
# AUTHENTICATION
# =============================================================================

@mcp.tool()
def hf_whoami() -> Dict[str, Any]:
    """Get current authenticated user info"""
    if not TOKEN:
        return {"error": "Not authenticated. Set HF_TOKEN environment variable."}
    try:
        user_info = whoami(token=TOKEN)
        return {
            "authenticated": True,
            "user": user_info,
            "name": user_info.get('name', 'Unknown'),
            "email": user_info.get('email', 'Unknown')
        }
    except Exception as e:
        return {"error": f"Failed to get user info: {str(e)}"}

# =============================================================================
# REPOSITORY MANAGEMENT
# =============================================================================

@mcp.tool()
def hf_create_repository(
    repo_id: str,
    repo_type: str = "model",
    private: bool = False,
    description: Optional[str] = None,
    space_sdk: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new repository on Hugging Face Hub
    
    Args:
        repo_id: Repository ID (username/repo-name)
        repo_type: Type of repository ("model", "dataset", "space")
        private: Whether the repository should be private
        description: Repository description
        space_sdk: Required for Spaces - must be one of: "gradio", "streamlit", "docker", "static"
    """
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot create repositories"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    # Validate space_sdk for Spaces
    valid_sdks = ["gradio", "streamlit", "docker", "static"]
    if repo_type == "space":
        if not space_sdk:
            return {
                "error": "❌ space_sdk is required for Spaces",
                "valid_options": valid_sdks,
                "help": {
                    "gradio": "Interactive ML demos with Python",
                    "streamlit": "Data apps and dashboards with Python",
                    "docker": "Custom applications with Docker",
                    "static": "HTML/CSS/JavaScript websites"
                }
            }
        if space_sdk not in valid_sdks:
            return {
                "error": f"❌ Invalid space_sdk: {space_sdk}",
                "valid_options": valid_sdks
            }
    
    try:
        # Create repository with appropriate parameters
        create_params = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private,
            "token": TOKEN
        }
        
        # Add space_sdk for Spaces
        if repo_type == "space" and space_sdk:
            create_params["space_sdk"] = space_sdk
        
        repo_url = create_repo(**create_params)
        
        result = {
            "status": "success",
            "message": f"✅ Created {repo_type} repository: {repo_id}",
            "repo_url": repo_url,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "private": private
        }
        
        # Add SDK info for Spaces
        if repo_type == "space":
            result["space_sdk"] = space_sdk
            result["next_steps"] = {
                "gradio": "Upload app.py with Gradio interface + requirements.txt",
                "streamlit": "Upload app.py with Streamlit app + requirements.txt",
                "docker": "Upload Dockerfile + your application files",
                "static": "Upload index.html + CSS/JS files"
            }[space_sdk]
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to create repository: {str(e)}"}

@mcp.tool()
def hf_delete_repository(repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
    """Delete a repository (requires admin mode)"""
    if not ADMIN_MODE:
        return {"error": "❌ Admin mode required for repository deletion"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    try:
        delete_repo(repo_id=repo_id, repo_type=repo_type, token=TOKEN)
        return {
            "status": "success",
            "message": f"🗑️ Deleted repository: {repo_id}",
            "repo_id": repo_id,
            "repo_type": repo_type
        }
    except Exception as e:
        return {"error": f"Failed to delete repository: {str(e)}"}

@mcp.tool()
def hf_get_repository_info(repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
    """Get detailed information about a repository"""
    try:
        if repo_type == "model":
            info = model_info(repo_id, token=TOKEN)
        elif repo_type == "dataset":
            info = dataset_info(repo_id, token=TOKEN)
        else:
            return {"error": f"Unsupported repo_type: {repo_type}"}
        
        return {
            "repo_id": info.id,
            "author": info.author,
            "created_at": info.created_at.isoformat() if info.created_at else None,
            "last_modified": info.last_modified.isoformat() if info.last_modified else None,
            "private": info.private,
            "downloads": getattr(info, 'downloads', 0),
            "likes": getattr(info, 'likes', 0),
            "tags": getattr(info, 'tags', []),
            "file_count": len(info.siblings) if hasattr(info, 'siblings') else 0,
            "files": [s.rfilename for s in info.siblings[:10]] if hasattr(info, 'siblings') else []
        }
    except Exception as e:
        return {"error": f"Failed to get repository info: {str(e)}"}

@mcp.tool()
def hf_list_repository_files(repo_id: str, repo_type: str = "model") -> Dict[str, Any]:
    """List all files in a repository"""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=TOKEN)
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "file_count": len(files),
            "files": files
        }
    except Exception as e:
        return {"error": f"Failed to list files: {str(e)}"}

# =============================================================================
# FILE OPERATIONS
# =============================================================================

@mcp.tool()
def hf_read_file(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    revision: str = "main",
    max_size: int = 500000  # 500KB default limit, 0 = no limit
) -> Dict[str, Any]:
    """Read a file from a repository"""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=TOKEN
        )
        
        # Check file size first
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_size > 0 and file_size > max_size:
                # Read only the specified amount
                content = f.read(max_size)
                return {
                    "repo_id": repo_id,
                    "filename": filename,
                    "content": content,
                    "size": len(content),
                    "full_file_size": file_size,
                    "truncated": True,
                    "message": f"📖 Read {filename} (truncated to {max_size:,} chars of {file_size:,} total)",
                    "note": f"File was truncated. Use max_size=0 to read full file or increase max_size parameter."
                }
            else:
                # Read full file
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
        file_size = os.path.getsize(file_path) if 'file_path' in locals() else 0
        return {
            "repo_id": repo_id,
            "filename": filename,
            "error": "Binary file - cannot display as text",
            "size": file_size,
            "is_binary": True
        }
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

@mcp.tool()
def hf_read_file_chunked(
    repo_id: str,
    filename: str,
    chunk_size: int = 50000,
    chunk_number: int = 0,
    repo_type: str = "model",
    revision: str = "main"
) -> Dict[str, Any]:
    """Read a file from a repository in chunks (for very large files)
    
    Args:
        repo_id: Repository ID
        filename: File to read
        chunk_size: Size of each chunk in characters
        chunk_number: Which chunk to read (0-based)
        repo_type: Repository type
        revision: Repository revision
    """
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=TOKEN
        )
        
        file_size = os.path.getsize(file_path)
        start_pos = chunk_number * chunk_size
        
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(start_pos)
            content = f.read(chunk_size)
            
            total_chunks = (file_size + chunk_size - 1) // chunk_size  # Ceiling division
            
            return {
                "repo_id": repo_id,
                "filename": filename,
                "content": content,
                "chunk_number": chunk_number,
                "chunk_size": len(content),
                "total_chunks": total_chunks,
                "file_size": file_size,
                "has_more": chunk_number < total_chunks - 1,
                "message": f"📖 Read chunk {chunk_number + 1}/{total_chunks} of {filename}"
            }
        
    except Exception as e:
        return {"error": f"Failed to read file chunk: {str(e)}"}

@mcp.tool()
def hf_write_file(
    repo_id: str,
    filename: str,
    content: Union[str, list],
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """Write/upload a file to a repository"""
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot write files"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    try:
        # Handle different content formats
        if isinstance(content, list):
            # If content is a list of text objects, extract the text
            text_content = ""
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                elif isinstance(item, str):
                    text_content += item
            content = text_content
        elif not isinstance(content, str):
            # Convert other types to string
            content = str(content)
        
        if not commit_message:
            commit_message = f"Upload {filename}"
        
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
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}

@mcp.tool()
def hf_edit_file(
    repo_id: str,
    filename: str,
    old_text: str,
    new_text: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """Edit a file by replacing specific text (PRECISE EDITING)"""
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot edit files"}
    
    try:
        # Read current content
        read_result = hf_read_file(repo_id, filename, repo_type)
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
        
        # Replace text (only first occurrence)
        new_content = current_content.replace(old_text, new_text, 1)
        
        if not commit_message:
            commit_message = f"Edit {filename}: Replace text"
        
        # Write updated content
        write_result = hf_write_file(repo_id, filename, new_content, repo_type, commit_message)
        
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
    except Exception as e:
        return {"error": f"Failed to edit file: {str(e)}"}

@mcp.tool()
def hf_delete_file(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> Dict[str, Any]:
    """Delete a file from a repository"""
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot delete files"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    try:
        if not commit_message:
            commit_message = f"Delete {filename}"
        
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
    except Exception as e:
        return {"error": f"Failed to delete file: {str(e)}"}

# =============================================================================
# SEARCH OPERATIONS
# =============================================================================

@mcp.tool()
def hf_search_models(
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Search for models on Hugging Face Hub"""
    try:
        models = list_models(
            search=query,
            author=author,
            filter=filter_tag,
            sort="downloads",
            direction=-1,
            limit=limit,
            token=TOKEN
        )
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags[:5],  # Limit tags for readability
                "created_at": model.created_at.isoformat() if model.created_at else None
            })
        
        return {
            "query": query,
            "author": author,
            "filter_tag": filter_tag,
            "total_results": len(results),
            "models": results
        }
    except Exception as e:
        return {"error": f"Failed to search models: {str(e)}"}

@mcp.tool()
def hf_search_datasets(
    query: Optional[str] = None,
    author: Optional[str] = None,
    filter_tag: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Search for datasets on Hugging Face Hub"""
    try:
        datasets = list_datasets(
            search=query,
            author=author,
            filter=filter_tag,
            limit=limit,
            token=TOKEN
        )
        
        results = []
        for dataset in datasets:
            results.append({
                "id": dataset.id,
                "author": dataset.author,
                "downloads": dataset.downloads,
                "likes": dataset.likes,
                "tags": dataset.tags[:5],
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "datasets": results
        }
    except Exception as e:
        return {"error": f"Failed to search datasets: {str(e)}"}

@mcp.tool()
def hf_search_spaces(
    query: Optional[str] = None,
    author: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Search for Spaces on Hugging Face Hub"""
    try:
        spaces = list_spaces(
            search=query,
            author=author,
            limit=limit,
            token=TOKEN
        )
        
        results = []
        for space in spaces:
            results.append({
                "id": space.id,
                "author": space.author,
                "likes": space.likes,
                "tags": space.tags[:5],
                "sdk": getattr(space, 'sdk', None),
                "created_at": space.created_at.isoformat() if space.created_at else None
            })
        
        return {
            "query": query,
            "total_results": len(results),
            "spaces": results
        }
    except Exception as e:
        return {"error": f"Failed to search spaces: {str(e)}"}

# =============================================================================
# COLLECTIONS
# =============================================================================

@mcp.tool()
def hf_collection_create(
    title: str,
    namespace: Optional[str] = None,
    description: Optional[str] = None,
    private: bool = False
) -> Dict[str, Any]:
    """Create a new collection"""
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot create collections"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required"}
    
    try:
        # If no namespace provided, get the current user's username
        if namespace is None:
            user_info = whoami(token=TOKEN)
            namespace = user_info.get('name', 'unknown')
        
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
            "url": collection.url if hasattr(collection, 'url') else f"https://huggingface.co/collections/{namespace}/{collection.slug}"
        }
    except Exception as e:
        return {"error": f"Failed to create collection: {str(e)}"}

@mcp.tool()
def hf_collection_add(
    collection_slug: str,
    item_id: str,
    item_type: str,
    note: Optional[str] = None
) -> Dict[str, Any]:
    """Add an item to a collection"""
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot modify collections"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required"}
    
    try:
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
    except Exception as e:
        return {"error": f"Failed to add to collection: {str(e)}"}

@mcp.tool()
def hf_collection_info(collection_slug: str) -> Dict[str, Any]:
    """Get information about a collection"""
    try:
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
    except Exception as e:
        return {"error": f"Failed to get collection info: {str(e)}"}

# =============================================================================
# PULL REQUESTS & DISCUSSIONS
# =============================================================================

@mcp.tool()
def hf_create_pull_request(
    repo_id: str,
    title: str,
    description: Optional[str] = None,
    repo_type: str = "model"
) -> Dict[str, Any]:
    """Create a pull request on a Hugging Face repository
    
    Args:
        repo_id: Repository ID (username/repo-name)
        title: Title of the pull request (3-200 characters)
        description: Optional description for the pull request
        repo_type: Type of repository ("model", "dataset", "space")
    """
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    # Validate title length
    if len(title.strip()) < 3:
        return {"error": "❌ Title must be at least 3 characters long"}
    if len(title.strip()) > 200:
        return {"error": "❌ Title must be 200 characters or less"}
    
    try:
        # Create the pull request using create_discussion with pull_request=True
        discussion = create_discussion(
            repo_id=repo_id,
            title=title.strip(),
            description=description or "Pull Request created with HuggingMCP",
            repo_type=repo_type,
            pull_request=True,
            token=TOKEN
        )
        
        return {
            "status": "success",
            "message": f"🔀 Created pull request: {title}",
            "pr_number": discussion.num,
            "pr_title": discussion.title,
            "pr_url": discussion.url,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status_note": "Pull request created in draft mode - ready for changes!"
        }
    except Exception as e:
        return {"error": f"Failed to create pull request: {str(e)}"}

@mcp.tool()
def hf_create_commit_pr(
    repo_id: str,
    commit_message: str,
    files: list,
    pr_title: Optional[str] = None,
    pr_description: Optional[str] = None,
    repo_type: str = "model",
    parent_commit: Optional[str] = None
) -> Dict[str, Any]:
    """Create a commit with changes and open it as a pull request
    
    Args:
        repo_id: Repository ID (username/repo-name)
        commit_message: Message for the commit
        files: List of file operations [{"path": "file.txt", "content": "content"}]
        pr_title: Title for the pull request (auto-generated if not provided)
        pr_description: Description for the pull request
        repo_type: Type of repository ("model", "dataset", "space")
        parent_commit: Parent commit hash for the PR base
    """
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot create commits or pull requests"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    if not files:
        return {"error": "❌ At least one file operation is required"}
    
    try:
        from huggingface_hub import CommitOperationAdd
        
        # Create commit operations from files list
        operations = []
        for file_op in files:
            if not isinstance(file_op, dict) or "path" not in file_op or "content" not in file_op:
                return {"error": f"❌ Invalid file operation format. Use: {{\"path\": \"file.txt\", \"content\": \"content\"}}"}
            
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
            pr_description=pr_description or f"Automated pull request created with HuggingMCP\n\nCommit: {commit_message}",
            parent_commit=parent_commit,
            token=TOKEN
        )
        
        return {
            "status": "success",
            "message": f"🔀 Created commit and pull request: {commit_message}",
            "commit_url": commit_result.commit_url,
            "pr_url": commit_result.pr_url if hasattr(commit_result, 'pr_url') else None,
            "commit_sha": commit_result.oid,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "files_changed": len(operations),
            "note": "Pull request created with your changes!"
        }
    except Exception as e:
        return {"error": f"Failed to create commit with pull request: {str(e)}"}

@mcp.tool()
def hf_list_pull_requests(
    repo_id: str,
    repo_type: str = "model",
    status: str = "open",
    author: Optional[str] = None
) -> Dict[str, Any]:
    """List pull requests in a repository
    
    Args:
        repo_id: Repository ID (username/repo-name)
        repo_type: Type of repository ("model", "dataset", "space")
        status: Status filter ("open", "closed", "all")
        author: Filter by author username
    """
    try:
        # Map status parameter
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
            "pull_requests": prs[:20]  # Limit to first 20 for readability
        }
    except Exception as e:
        return {"error": f"Failed to list pull requests: {str(e)}"}

@mcp.tool()
def hf_get_pull_request_details(
    repo_id: str,
    pr_number: int,
    repo_type: str = "model"
) -> Dict[str, Any]:
    """Get detailed information about a specific pull request
    
    Args:
        repo_id: Repository ID (username/repo-name)
        pr_number: Pull request number
        repo_type: Type of repository ("model", "dataset", "space")
    """
    try:
        discussion = get_discussion_details(
            repo_id=repo_id,
            discussion_num=pr_number,
            repo_type=repo_type,
            token=TOKEN
        )
        
        # Extract events/comments
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
            "git_reference": getattr(discussion, 'git_reference', None),
            "events": events,
            "total_events": len(events)
        }
    except Exception as e:
        return {"error": f"Failed to get pull request details: {str(e)}"}

@mcp.tool()
def hf_upload_file_pr(
    repo_id: str,
    file_path: str,
    content: str,
    commit_message: str,
    pr_title: Optional[str] = None,
    pr_description: Optional[str] = None,
    repo_type: str = "model"
) -> Dict[str, Any]:
    """Upload a single file and create a pull request
    
    Args:
        repo_id: Repository ID (username/repo-name)
        file_path: Path where to save the file in the repo
        content: File content
        commit_message: Commit message
        pr_title: Title for the pull request
        pr_description: Description for the pull request
        repo_type: Type of repository ("model", "dataset", "space")
    """
    if READ_ONLY:
        return {"error": "❌ Read-only mode - cannot upload files or create pull requests"}
    
    if not TOKEN:
        return {"error": "❌ Authentication required - set HF_TOKEN environment variable"}
    
    try:
        # Upload file with create_pr=True
        result = upload_file(
            path_or_fileobj=content.encode('utf-8'),
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            create_pr=True,
            pr_title=pr_title or f"Add {file_path}",
            pr_description=pr_description or f"Uploaded {file_path} via HuggingMCP",
            token=TOKEN
        )
        
        return {
            "status": "success",
            "message": f"📤 Uploaded {file_path} and created pull request",
            "file_path": file_path,
            "file_size": len(content.encode('utf-8')),
            "commit_message": commit_message,
            "pr_url": result if isinstance(result, str) else None,
            "repo_id": repo_id,
            "repo_type": repo_type
        }
    except Exception as e:
        return {"error": f"Failed to upload file with pull request: {str(e)}"}

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    print("🤗 HuggingMCP Server Starting...")
    print(f"   Authenticated: {'✅' if TOKEN else '❌'}")
    print(f"   Admin Mode: {'✅' if ADMIN_MODE else '❌'}")
    print(f"   Read Only: {'✅' if READ_ONLY else '❌'}")
    print("   🚀 Ready with 23 HF tools including Pull Requests!")
    
    # Run the server
    mcp.run()
