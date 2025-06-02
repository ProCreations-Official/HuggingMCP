# 🤗 HuggingMCP - Hugging Face Model Context Protocol Server

**Give Claude superpowers with Hugging Face!** 🚀

HuggingMCP is a comprehensive Model Context Protocol (MCP) server that allows Claude and other MCP-compatible AI assistants to interact seamlessly with the Hugging Face ecosystem. Create models, manage datasets, edit files, organize collections, and much more - all through natural language!

## ✨ Features

### 🏗️ **Repository Management**
- **Create repositories** (models, datasets, spaces) with custom settings
- **Delete repositories** (admin mode required)
- **Get detailed repository information** and metadata
- **List repository files** and directory structures

### 📝 **Advanced File Operations**
- **Read files** from any Hugging Face repository (public/private)
- **Write/upload files** with custom content
- **Precise file editing** with exact text replacement (old_text → new_text)
- **Delete files** from repositories
- **Binary file support** for non-text files

### 🔍 **Search & Discovery**
- **Search models** with filters (author, tags, popularity)
- **Search datasets** across all of Hugging Face
- **Search Spaces** and demo applications
- **Advanced filtering** by downloads, likes, creation date
- **Comprehensive metadata** for all results

### 📚 **Collections Management**
- **Create collections** to organize repositories
- **Add items** to collections (models, datasets, spaces, papers)
- **Manage collection metadata** and descriptions
- **Get collection information** and item lists

### 🔒 **Security & Permissions**
- **Token-based authentication** with Hugging Face
- **Permission controls**: read-only, write-only, admin modes
- **File size limits** to prevent abuse
- **Comprehensive error handling**

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **Claude Desktop** application ([Download here](https://claude.ai/desktop))
- **Hugging Face account** and access token ([Get token here](https://huggingface.co/settings/tokens))

### Installation

1. **Create a project directory:**
```bash
mkdir huggingmcp && cd huggingmcp
```

2. **Save the main.py file** from the artifact in your project directory

3. **Install dependencies:**
```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv add "mcp[cli]" huggingface_hub
```

### Configuration

1. **Open Claude Desktop settings:**
   - Go to Settings → Developer
   - Click "Edit Config" to open `claude_desktop_config.json`

2. **Add HuggingMCP configuration:**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/YOUR/huggingmcp",
        "run",
        "main.py"
      ],
      "env": {
        "HF_TOKEN": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Important:** Replace `/ABSOLUTE/PATH/TO/YOUR/huggingmcp` with the actual absolute path to your project directory.

3. **Restart Claude Desktop** to load the MCP server

4. **Verify connection:** Look for the 🔨 hammer icon in Claude Desktop, indicating MCP tools are available.

## 🎯 Usage Examples

Once connected, you can use natural language to interact with Hugging Face:

### 🔐 Authentication
```
"Please login to Hugging Face with my token: hf_xxxxxxxxxxxx"
"Who am I currently logged in as?"
"What are my current permissions?"
```

### 🏗️ Repository Operations
```
"Create a new model repository called 'my-awesome-model' with a custom README"
"Show me information about the 'microsoft/DialoGPT-medium' model"
"List all files in the 'squad' dataset repository"
"Delete my test repository (admin mode required)"
```

### 📝 File Management
```
"Read the README.md file from 'gpt2' model repository"
"Create a new config.json file in my model repo with these settings: {...}"
"Edit the training script and replace 'learning_rate=0.001' with 'learning_rate=0.0001'"
"Delete the old_model.bin file from my repository"
```

### 🔍 Search & Discovery
```
"Find the top 10 most downloaded text classification models"
"Search for datasets related to sentiment analysis by huggingface"
"Show me recent Gradio spaces for image generation"
"Find models tagged with 'pytorch' and 'transformer'"
```

### 📚 Collections
```
"Create a new collection called 'My Favorite Models'"
"Add the 'bert-base-uncased' model to my collection with a note"
"Show me all items in the 'best-nlp-models' collection"
```

## 🛡️ Security & Permissions

HuggingMCP includes comprehensive permission controls:

### Permission Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Default** | Full read/write access | Development and experimentation |
| **Read Only** | Can only read repositories and files | Safe exploration mode |
| **Write Only** | Can only create/modify (no deletion) | Content creation workflows |
| **Admin Mode** | Full access including deletion | Advanced repository management |

### Setting Permissions

**Via Environment Variables:**
```bash
export HF_READ_ONLY=true   # Enable read-only mode
export HF_ADMIN_MODE=true  # Enable admin mode
```

**Via Claude Commands:**
```
"Set HuggingMCP to read-only mode"
"Enable admin mode for repository deletion"
"Show me my current permissions"
```

## 🧩 Available Tools

HuggingMCP exposes the following tools to Claude:

### Authentication
- `hf_login` - Login with Hugging Face token
- `hf_whoami` - Get current user info
- `hf_logout` - Logout from Hugging Face

### Repository Management
- `create_repository` - Create new repos (models/datasets/spaces)
- `delete_repository` - Delete repos (admin mode)
- `get_repository_info` - Get repo metadata
- `list_repository_files` - List files in repo

### File Operations
- `read_file` - Read file content
- `write_file` - Write/upload files
- `edit_file` - Precise text replacement editing
- `delete_file_from_repo` - Delete specific files

### Search & Discovery
- `search_models` - Search HF models
- `search_datasets` - Search HF datasets  
- `search_spaces` - Search HF Spaces

### Collections
- `create_hf_collection` - Create collections
- `add_to_collection` - Add items to collections
- `get_collection_info` - Get collection details

### Configuration
- `get_hf_config` - Get current config
- `set_hf_permissions` - Update permissions

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | None | Your Hugging Face access token |
| `HF_READ_ONLY` | false | Enable read-only mode |
| `HF_WRITE_ONLY` | false | Enable write-only mode |
| `HF_ADMIN_MODE` | false | Enable admin operations |
| `HF_MAX_FILE_SIZE` | 50000000 | Max file size in bytes (50MB) |

### Claude Desktop Config

**Minimal Configuration:**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "python",
      "args": ["/path/to/main.py"],
      "env": {
        "HF_TOKEN": "your_token_here"
      }
    }
  }
}
```

**Advanced Configuration:**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/huggingmcp",
        "run", "main.py"
      ],
      "env": {
        "HF_TOKEN": "hf_xxxxxxxxxxxx",
        "HF_READ_ONLY": "false",
        "HF_ADMIN_MODE": "false",
        "HF_MAX_FILE_SIZE": "100000000"
      }
    }
  }
}
```

## 🐛 Troubleshooting

### Common Issues

**1. "Connection failed" in Claude Desktop**
- Verify the absolute path in your config is correct
- Check that `main.py` exists in the specified directory
- Ensure Python/uv is accessible from the command line

**2. "Authentication required" errors**
- Verify your HF_TOKEN is valid at https://huggingface.co/settings/tokens
- Ensure the token has appropriate permissions (read/write)
- Check that the token is correctly set in environment or config

**3. "Permission denied" errors**
- Check your permission settings with `get_hf_config`
- Verify you're not in read-only mode for write operations
- Ensure admin mode is enabled for deletion operations

**4. "File too large" errors**
- Check the file size against `HF_MAX_FILE_SIZE` setting
- Increase the limit via environment variable if needed
- Consider splitting large files into smaller chunks

### Debug Logs

Check Claude Desktop MCP logs:
- **macOS:** `~/Library/Logs/Claude/mcp.log`
- **Windows:** `%APPDATA%/Claude/Logs/mcp.log`

Enable verbose logging in main.py:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. **Check the logs** in Claude Desktop's MCP log files
2. **Verify configuration** using the `get_hf_config` tool
3. **Test authentication** with `hf_whoami`
4. **Start simple** with read operations before trying writes

## 🤝 Contributing

Found a bug or want to add a feature? Here's how you can help:

1. **Report Issues:** Open an issue describing the problem
2. **Feature Requests:** Suggest new Hugging Face integrations
3. **Code Contributions:** Submit pull requests with improvements
4. **Documentation:** Help improve these docs!

## 📄 License

MIT License - feel free to use, modify, and distribute!

## 🙏 Acknowledgments

- **Anthropic** for creating the Model Context Protocol
- **Hugging Face** for their amazing platform and APIs
- **FastMCP** team for the excellent Python SDK

---

**Happy prompting with HuggingMCP!** 🤗✨

*Now Claude can be your AI pair programmer for all things Hugging Face!*
