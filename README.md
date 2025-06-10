# HuggingMCP - Enhanced Hugging Face MCP Server

A comprehensive and optimized Model Context Protocol (MCP) server for Hugging Face Hub operations, featuring 10 consolidated commands, enhanced debugging, and robust error handling.

## 🚀 Features

- **Optimized Command Structure**: Consolidated from 23+ commands to 10 main commands
- **Enhanced Debugging**: Comprehensive stderr output and logging for troubleshooting
- **Robust Error Handling**: Safe execution wrappers and detailed error reporting
- **Batch Operations**: Execute multiple operations efficiently
- **Advanced Search**: Cross-content search with popularity scoring
- **File Operations**: Read, write, edit, and delete files with chunked reading support
- **Repository Management**: Create, delete, and manage repositories with creator tracking
- **Pull Request Support**: Create and manage PRs with file changes
- **Collection Management**: Create and manage Hugging Face collections
- **Comprehensive Diagnostics**: System health checks and connectivity testing

## 📋 Prerequisites

- Python 3.8+
- Required packages:
  ```bash
  pip install mcp huggingface_hub
  ```
- Hugging Face token (set as `HF_TOKEN` environment variable)

## ⚙️ Configuration

Add to your Claude Desktop configuration file at:
`/Users/[username]/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "python3",
      "args": ["/Users/sshpro/Documents/hugmcp.py"],
      "env": {
        "HF_TOKEN": "your_hugging_face_token_here",
        "HF_ADMIN_MODE": "true",
        "HF_READ_ONLY": "false",
        "HF_WRITE_ONLY": "false",
        "HF_MAX_FILE_SIZE": "100000000"
      }
    }
  }
}
```

### Environment Variables

- `HF_TOKEN`: Your Hugging Face API token (required for write operations)
- `HF_ADMIN_MODE`: Enable admin operations like repository deletion (default: false)
- `HF_READ_ONLY`: Restrict to read-only operations (default: false)
- `HF_MAX_FILE_SIZE`: Maximum file size for operations (default: 100MB)

## 🛠️ Available Commands

### 1. `hf_system_info()`
Get system information, configuration, and test connectivity.
```python
# Returns server status, configuration, user info, and capabilities
```

### 2. `hf_repository_manager(action, repo_id, repo_type="model", **kwargs)`
Comprehensive repository management.

**Actions:**
- `create`: Create new repository
  - `private`: Make repository private (default: False)
  - `description`: Repository description
  - `space_sdk`: For Spaces - "gradio", "streamlit", "docker", "static"
  - `creator`: Repository creator (defaults to authenticated user)
- `delete`: Delete repository (requires admin mode)
- `info`: Get repository information
- `list_files`: List all files in repository

**Examples:**
```python
# Create a public model repository
hf_repository_manager("create", "my-awesome-model", "model", 
                     description="My awesome AI model")

# Create a private Gradio space
hf_repository_manager("create", "my-space", "space", 
                     private=True, space_sdk="gradio")

# Get repository info
hf_repository_manager("info", "microsoft/DialoGPT-medium")

# List files in repository
hf_repository_manager("list_files", "gpt2")
```

### 3. `hf_file_operations(action, repo_id, filename, repo_type="model", **kwargs)`
Comprehensive file operations.

**Actions:**
- `read`: Read file content
  - `max_size`: Maximum characters to read (default: 500,000)
  - `chunk_size`: Enable chunked reading
  - `chunk_number`: Chunk number to read (for chunked reading)
- `write`: Write/upload file content
  - `content`: File content to write
  - `commit_message`: Commit message
- `edit`: Edit file by replacing text
  - `old_text`: Text to replace
  - `new_text`: Replacement text
  - `commit_message`: Commit message
- `delete`: Delete file from repository

**Examples:**
```python
# Read a file (truncated to 1000 chars)
hf_file_operations("read", "gpt2", "README.md", max_size=1000)

# Read file in chunks
hf_file_operations("read", "gpt2", "config.json", chunk_size=1000, chunk_number=0)

# Write a new file
hf_file_operations("write", "my-repo", "new_file.txt", 
                  content="Hello World!", 
                  commit_message="Add new file")

# Edit existing file
hf_file_operations("edit", "my-repo", "README.md",
                  old_text="# Old Title",
                  new_text="# New Title",
                  commit_message="Update title")
```

### 4. `hf_search_hub(content_type, query=None, author=None, filter_tag=None, limit=20)`
Search Hugging Face Hub for models, datasets, or spaces.

**Examples:**
```python
# Search for transformer models
hf_search_hub("models", query="transformer", limit=10)

# Search for datasets by specific author
hf_search_hub("datasets", author="huggingface", limit=5)

# Search for Gradio spaces
hf_search_hub("spaces", filter_tag="gradio")
```

### 5. `hf_collections(action, **kwargs)`
Manage Hugging Face Collections.

**Actions:**
- `create`: Create new collection
  - `title`: Collection title (required)
  - `namespace`: Collection namespace (defaults to user)
  - `description`: Collection description
  - `private`: Make collection private
- `add_item`: Add item to collection
  - `collection_slug`: Collection identifier
  - `item_id`: Item to add (repo ID)
  - `item_type`: Type of item ("model", "dataset", "space")
  - `note`: Optional note about the item
- `info`: Get collection information
  - `collection_slug`: Collection identifier

**Examples:**
```python
# Create a new collection
hf_collections("create", title="My AI Models", 
               description="Collection of my favorite models")

# Add item to collection
hf_collections("add_item", 
               collection_slug="my-collection",
               item_id="gpt2", 
               item_type="model",
               note="Great base model")
```

### 6. `hf_pull_requests(action, repo_id, repo_type="model", **kwargs)`
Manage Pull Requests.

**Actions:**
- `create`: Create empty PR
  - `title`: PR title (required, min 3 characters)
  - `description`: PR description
- `list`: List PRs
  - `status`: Filter by status ("open", "closed", "all")
  - `author`: Filter by author
- `details`: Get PR details
  - `pr_number`: PR number to get details for
- `create_with_files`: Create PR with file changes
  - `files`: List of {path, content} dictionaries
  - `commit_message`: Commit message
  - `pr_title`: PR title
  - `pr_description`: PR description

**Examples:**
```python
# Create a simple PR
hf_pull_requests("create", "my-repo", title="Update documentation")

# Create PR with file changes
hf_pull_requests("create_with_files", "my-repo",
                 files=[{"path": "README.md", "content": "Updated content"}],
                 commit_message="Update README",
                 pr_title="Documentation update")

# List open PRs
hf_pull_requests("list", "my-repo", status="open")
```

### 7. `hf_upload_manager(action, repo_id, repo_type="model", **kwargs)`
Upload management with various options.

**Actions:**
- `single_file`: Upload one file
  - `file_path`: Path in repository
  - `content`: File content
  - `commit_message`: Commit message
- `multiple_files`: Upload multiple files
  - `files`: List of {path, content} dictionaries
  - `commit_message`: Commit message
- `with_pr`: Upload file(s) and create PR
  - `file_path`: Path in repository
  - `content`: File content
  - `commit_message`: Commit message
  - `pr_title`: PR title
  - `pr_description`: PR description

### 8. `hf_batch_operations(operation_type, operations)`
Execute multiple operations in batch.

**Operation Types:**
- `search`: Batch search operations
- `info`: Batch repository info retrieval
- `files`: Batch file listing

**Example:**
```python
# Batch search multiple content types
hf_batch_operations("search", [
    {"content_type": "models", "query": "transformer", "limit": 5},
    {"content_type": "datasets", "query": "text", "limit": 3}
])
```

### 9. `hf_advanced_search(query, search_types=["models", "datasets", "spaces"], filters=None, limit_per_type=10)`
Advanced search across multiple content types with filtering and popularity scoring.

**Example:**
```python
# Advanced search with filtering
hf_advanced_search("transformer", 
                   search_types=["models", "datasets"],
                   filters={"author": "huggingface"},
                   limit_per_type=15)
```

### 10. `hf_debug_diagnostics()`
Comprehensive debugging and diagnostic information.
```python
# Get system diagnostics, connectivity tests, and debug info
```

## 🔧 Debugging

The server includes comprehensive debugging features:

- **Stderr Output**: Real-time debugging information printed to stderr
- **Log Files**: Detailed logs written to `/tmp/hugmcp_debug.log`
- **Diagnostic Tools**: Use `hf_debug_diagnostics()` for system health checks
- **Error Tracking**: Comprehensive error handling with stack traces

### Debug Log Locations
- MCP Logs: `/Users/[username]/Library/Logs/Claude/mcp-server-huggingmcp.log`
- Debug Logs: `/tmp/hugmcp_debug.log`

## 🛡️ Security & Permissions

- **Authentication**: Requires HF_TOKEN for write operations
- **Read-Only Mode**: Set `HF_READ_ONLY=true` to prevent modifications
- **Admin Mode**: Set `HF_ADMIN_MODE=true` to enable repository deletion
- **File Size Limits**: Configurable via `HF_MAX_FILE_SIZE`

## 🐛 Troubleshooting

### Common Issues

1. **Server Disconnects Immediately**
   - Check if the script path in configuration is correct
   - Verify Python dependencies are installed
   - Check debug logs for detailed error information

2. **Authentication Errors**
   - Ensure `HF_TOKEN` is set correctly
   - Verify token has required permissions
   - Check token validity at huggingface.co

3. **File Path Errors**
   - Verify the script is at `/Users/sshpro/Documents/hugmcp.py`
   - Check file permissions (should be readable/executable)

4. **Import Errors**
   - Install required packages: `pip install mcp huggingface_hub`
   - Check Python version compatibility

### Getting Help

1. Use `hf_debug_diagnostics()` for system information
2. Check stderr output for real-time debugging
3. Review log files for detailed error traces
4. Verify configuration in Claude Desktop settings

## 📝 Version History

### v2.0.0 (Current)
- Consolidated 23+ commands into 10 optimized commands
- Enhanced debugging and error handling
- Added batch operations and advanced search
- Improved file operations with chunked reading
- Added comprehensive diagnostics
- Fixed metadata and connection stability issues

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please ensure all changes maintain the current command structure and include appropriate error handling and debugging output.
