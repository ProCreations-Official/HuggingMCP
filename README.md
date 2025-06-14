# HuggingMCP - Advanced Hugging Face MCP Server

A comprehensive and powerful Model Context Protocol (MCP) server for Hugging Face Hub operations, featuring 18+ specialized tools, AI workflow automation, and extensive ML capabilities.

## 🚀 Features

### Core Capabilities
- **Optimized Command Structure**: 18+ specialized commands covering all aspects of ML workflows
- **Enhanced Debugging**: Comprehensive stderr output and logging for troubleshooting
- **Robust Error Handling**: Safe execution wrappers with detailed error reporting and helpful guidance
- **Backward Compatibility**: All existing commands maintained with enhanced functionality

### New Advanced Features
- **🔬 Model Evaluation & Testing**: Comprehensive model analysis, validation, and comparison tools
- **🗃️ Dataset Processing**: Advanced dataset analysis, validation, and management
- **📝 License Management**: Automated license checking, compliance validation, and suggestions
- **🤝 Community Features**: Repository likes, discussions, commit history, and social interactions
- **🚀 Space Management**: Complete Hugging Face Spaces control and monitoring
- **🧠 AI Inference Tools**: Model testing and inference capabilities with multiple strategies
- **⚙️ Workflow Automation**: Automated model card generation, README creation, and bulk operations
- **📊 Advanced Analytics**: Trending analysis, recommendation engine, and ecosystem insights
- **🛠️ Repository Utilities**: Health checks, backup tools, and comprehensive repository management

### Enhanced File Operations
- **Batch Processing**: Edit multiple files with pattern matching
- **File Validation**: Format-specific validation for JSON, Markdown, Python, etc.
- **Backup System**: Automatic backup creation before destructive operations
- **Advanced Reading**: Chunked reading, encoding detection, and size management

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
- `HF_MAX_FILE_SIZE`: Maximum file size for operations in bytes (default: 104857600 = 100MB)
- `HF_ENABLE_INFERENCE`: Enable inference API features (default: true)
- `HF_INFERENCE_TIMEOUT`: Timeout for inference operations in seconds (default: 30)
- `HF_CACHE_ENABLED`: Enable caching for better performance (default: true)

## 🛠️ Available Commands

### Core Commands (Enhanced)

### 1. `hf_system_info()`
Get comprehensive system information, configuration, and test connectivity.
```python
# Returns server status, configuration, user info, capabilities, and new features
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
Enhanced file operations with advanced capabilities.

**Actions:**
- `read`: Read file content with encoding detection
  - `max_size`: Maximum characters to read (default: 500,000)
  - `chunk_size`: Enable chunked reading
  - `chunk_number`: Chunk number to read (for chunked reading)
- `write`: Write/upload file content with validation
  - `content`: File content to write
  - `commit_message`: Commit message
- `edit`: Edit file by replacing text with backup
  - `old_text`: Text to replace
  - `new_text`: Replacement text
  - `commit_message`: Commit message
- `delete`: Delete file from repository
- `validate`: Validate file format and content
- `backup`: Create backup of file before operations
- `batch_edit`: Edit multiple files with pattern matching
  - `pattern`: Text pattern to replace
  - `replacement`: Replacement text
  - `file_patterns`: File patterns to match (e.g., ["*.md", "*.txt"])

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

### 11. `hf_repo_file_manager(action, repo_id, repo_type="model", filename=None, **kwargs)`
Unified repository and file management with rename support.

**Actions:**
- `repo_create`, `repo_delete`, `repo_info`, `list_files`
- `file_read`, `file_write`, `file_edit`, `file_delete`, `file_rename`

**Example:**
```python
# Rename a file in a repository
hf_repo_file_manager("file_rename", "my-repo", filename="old.txt",
                     new_filename="new.txt", commit_message="Rename file")
```

### New Advanced Commands

### 12. `hf_model_evaluation(action, repo_id, **kwargs)`
Advanced model evaluation and testing capabilities.

**Actions:**
- `analyze`: Comprehensive model analysis including architecture, frameworks, and compatibility
- `compare`: Compare multiple models side by side
- `test_inference`: Test model inference capabilities (if supported)
- `validate_model`: Validate model integrity and completeness

**Examples:**
```python
# Analyze a model
hf_model_evaluation("analyze", "microsoft/DialoGPT-medium")

# Compare multiple models
hf_model_evaluation("compare", "gpt2", models=["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"])

# Validate model
hf_model_evaluation("validate_model", "my-model")
```

### 13. `hf_space_management(action, space_id, **kwargs)`
Advanced Hugging Face Spaces management.

**Actions:**
- `runtime_info`: Get space runtime information and status
- `restart`: Restart a space
- `pause`: Pause a space
- `set_sleep_time`: Set sleep time for a space
- `duplicate`: Duplicate a space to a new location

**Examples:**
```python
# Get space runtime info
hf_space_management("runtime_info", "gradio/chatbot")

# Restart a space
hf_space_management("restart", "my-space")

# Duplicate a space
hf_space_management("duplicate", "original-space", to_id="my-copied-space")
```

### 14. `hf_community_features(action, repo_id, repo_type="model", **kwargs)`
Community features and social interactions.

**Actions:**
- `like`: Like a repository
- `unlike`: Unlike a repository
- `get_likes`: Get user's liked repositories
- `create_discussion`: Create a discussion (non-PR)
- `get_commits`: Get repository commit history
- `get_refs`: Get repository branches and tags

**Examples:**
```python
# Like a repository
hf_community_features("like", "microsoft/DialoGPT-medium")

# Get commit history
hf_community_features("get_commits", "my-repo")

# Create a discussion
hf_community_features("create_discussion", "my-repo", 
                      title="Question about model", 
                      description="How do I use this model?")
```

### 15. `hf_dataset_processing(action, dataset_id, **kwargs)`
Advanced dataset processing and analysis tools.

**Actions:**
- `analyze`: Analyze dataset structure, size, and metadata
- `compare`: Compare multiple datasets
- `validate`: Validate dataset format and completeness

**Examples:**
```python
# Analyze a dataset
hf_dataset_processing("analyze", "squad")

# Compare datasets
hf_dataset_processing("compare", "squad", datasets=["squad", "glue", "imdb"])

# Validate dataset
hf_dataset_processing("validate", "my-dataset")
```

### 16. `hf_license_management(action, repo_id, repo_type="model", **kwargs)`
License management and compliance tools.

**Actions:**
- `check_license`: Check repository license information
- `validate_compliance`: Validate license compliance with scoring
- `suggest_license`: Suggest appropriate license based on content type and preferences

**Examples:**
```python
# Check license
hf_license_management("check_license", "my-model")

# Validate compliance
hf_license_management("validate_compliance", "my-model")

# Get license suggestions
hf_license_management("suggest_license", "my-model", 
                     content_type="model", commercial_use=True)
```

### 17. `hf_inference_tools(action, repo_id, **kwargs)`
Advanced inference and model testing tools.

**Actions:**
- `test_inference`: Test model inference with custom inputs
- `check_endpoints`: Check available inference endpoints

**Examples:**
```python
# Test inference
hf_inference_tools("test_inference", "gpt2", 
                  inputs=["Hello world", "How are you?"],
                  parameters={"max_length": 50})

# Check endpoints
hf_inference_tools("check_endpoints", "my-model")
```

### 18. `hf_ai_workflow_tools(action, **kwargs)`
Specialized AI workflow and automation tools.

**Actions:**
- `create_model_card`: Generate comprehensive model cards
- `bulk_operations`: Perform bulk operations across repositories
- `generate_readme`: Generate README files for repositories
- `validate_pipeline`: Validate complete ML pipelines

**Examples:**
```python
# Generate model card
hf_ai_workflow_tools("create_model_card", repo_id="my-model",
                    model_type="text-generation", language=["en"])

# Bulk operations
hf_ai_workflow_tools("bulk_operations", 
                    repo_list=["model1", "model2"], 
                    operation="validate")

# Generate README
hf_ai_workflow_tools("generate_readme", repo_id="my-model", repo_type="model")
```

### 19. `hf_advanced_analytics(action, **kwargs)`
Advanced analytics and insights for HuggingFace repositories.

**Actions:**
- `trending_analysis`: Analyze trending models/datasets with metrics
- `recommendation_engine`: Recommend repositories based on preferences

**Examples:**
```python
# Trending analysis
hf_advanced_analytics("trending_analysis", content_type="models", limit=50)

# Get recommendations
hf_advanced_analytics("recommendation_engine", 
                     content_type="models",
                     preferences={"tags": ["transformers", "pytorch"], "min_downloads": 1000})
```

### 20. `hf_repository_utilities(action, repo_id, repo_type="model", **kwargs)`
Advanced repository utilities and management tools.

**Actions:**
- `repository_health`: Comprehensive repository health check with scoring
- `backup_info`: Create comprehensive backup information

**Examples:**
```python
# Health check
hf_repository_utilities("repository_health", "my-model")

# Create backup info
hf_repository_utilities("backup_info", "my-model")
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

### v3.0.0 (Current) - Major Feature Release
- **Massive expansion**: Added 9+ new advanced command categories (18+ total tools)
- **🔬 Model Evaluation**: Complete model analysis, comparison, and validation system
- **🗃️ Dataset Processing**: Advanced dataset analysis and validation tools
- **📝 License Management**: Automated license checking and compliance validation
- **🤝 Community Features**: Repository likes, discussions, commit history, social interactions
- **🚀 Space Management**: Complete Hugging Face Spaces control and monitoring
- **🧠 AI Inference Tools**: Model testing and inference capabilities
- **⚙️ Workflow Automation**: Model card generation, README creation, bulk operations
- **📊 Advanced Analytics**: Trending analysis, recommendation engine, ecosystem insights
- **🛠️ Repository Utilities**: Health checks, backup tools, comprehensive management
- **Enhanced File Operations**: Batch editing, validation, backup system, format detection
- **Improved Error Handling**: Detailed validation, helpful error messages, guided troubleshooting
- **Graceful Degradation**: Feature detection for different huggingface_hub versions
- **Backward Compatibility**: All existing v2.x commands maintained and enhanced

### v2.1.0
- Consolidated 23+ commands into 11 optimized commands
- Added unified repo/file manager with rename support
- Enhanced debugging and error handling
- Added batch operations and advanced search
- Improved file operations with chunked reading
- Added comprehensive diagnostics
- Fixed metadata and connection stability issues

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please ensure all changes maintain the current command structure and include appropriate error handling and debugging output.
