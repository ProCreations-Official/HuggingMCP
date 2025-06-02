# HuggingMCP 🤗

**Model Context Protocol (MCP) server for comprehensive HuggingFace integration**

HuggingMCP provides Claude with full access to HuggingFace Hub functionality, enabling seamless creation, management, and interaction with spaces, models, datasets, and collections directly from your Claude conversations.

## ✨ Features

- **🔍 Search & Explore**: Find models, datasets, and spaces across HuggingFace Hub
- **📁 Repository Management**: Create, read, edit, and manage HuggingFace repositories
- **📝 File Operations**: Read, create, edit, and delete files in repositories with precise text replacement
- **🚀 Space Creation**: Create and configure HuggingFace Spaces with different SDKs (Gradio, Streamlit, etc.)
- **📊 Dataset Management**: Create and manage datasets with full API access
- **🏷️ Collection Management**: Create and manage HuggingFace collections
- **🔐 Permission Control**: Configurable access levels (read, write, admin)
- **📈 Statistics**: Get download stats, repo info, and user analytics

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- HuggingFace account with API token
- Claude Desktop application

### Step 1: Install Dependencies

```bash
pip install huggingface_hub mcp
```

### Step 2: Get Your HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **write** permissions
3. Copy the token (starts with `hf_...`)

### Step 3: Download HuggingMCP

Clone or download this repository:

```bash
git clone https://github.com/ProCreations-Official/HuggingMCP.git
cd HuggingMCP
```

### Step 4: Configure Claude Desktop

#### For macOS:
Open or create `~/Library/Application Support/Claude/claude_desktop_config.json`

#### For Windows:
Open or create `%APPDATA%/Claude/claude_desktop_config.json`

Add the following configuration:

```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "python",
      "args": ["/FULL/PATH/TO/HuggingMCP/main.py"],
      "env": {
        "HUGGINGFACE_TOKEN": "hf_your_token_here",
        "HF_MCP_PERMISSIONS": "admin"
      }
    }
  }
}
```

**Important**: Replace `/FULL/PATH/TO/HuggingMCP/main.py` with the actual full path to your main.py file.

#### Alternative Configuration Examples

**Read-only access:**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "python",
      "args": ["/FULL/PATH/TO/HuggingMCP/main.py"],
      "env": {
        "HUGGINGFACE_TOKEN": "hf_your_token_here",
        "HF_MCP_PERMISSIONS": "read"
      }
    }
  }
}
```

**Using uv (recommended):**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/FULL/PATH/TO/HuggingMCP",
        "run",
        "main.py"
      ],
      "env": {
        "HUGGINGFACE_TOKEN": "hf_your_token_here",
        "HF_MCP_PERMISSIONS": "admin"
      }
    }
  }
}
```

**Windows with WSL:**
```json
{
  "mcpServers": {
    "huggingmcp": {
      "command": "wsl.exe",
      "args": [
        "bash",
        "-c",
        "cd /path/to/HuggingMCP && python main.py"
      ],
      "env": {
        "HUGGINGFACE_TOKEN": "hf_your_token_here",
        "HF_MCP_PERMISSIONS": "admin"
      }
    }
  }
}
```

### Step 5: Restart Claude Desktop

1. Close Claude Desktop completely
2. Restart the application
3. Look for the MCP connection indicator (🔌 icon) in the interface

## 🔐 Permission Levels

Control what Claude can do with your HuggingFace account:

| Level | Permissions | Use Case |
|-------|------------|----------|
| `read` | Search, list, download, view | Safe exploration and reading |
| `write` | Read + create, edit, upload | Content creation and modification |
| `admin` | Write + delete, manage settings | Full control (default) |

Set permission level via the `HF_MCP_PERMISSIONS` environment variable.

## 🚀 Available Tools

### Search & Discovery
- `hf_search_models` - Search HuggingFace models
- `hf_search_datasets` - Search datasets
- `hf_search_spaces` - Search spaces
- `hf_whoami` - Get current user info

### Repository Management
- `hf_create_repo` - Create new repositories
- `hf_get_repo_info` - Get repository details
- `hf_list_repo_files` - List files in a repository
- `hf_update_repo_settings` - Update repository settings
- `hf_delete_repo` - Delete repositories ⚠️

### File Operations
- `hf_read_file` - Read file contents
- `hf_upload_file` - Create/upload new files
- `hf_edit_file` - Edit files with precise text replacement
- `hf_delete_file` - Delete files

### Space Management
- `hf_create_space` - Create HuggingFace Spaces
- `hf_list_user_repos` - List your repositories

### Collections
- `hf_create_collection` - Create collections
- `hf_get_collection` - Get collection info
- `hf_add_to_collection` - Add items to collections

### Analytics
- `hf_get_download_stats` - Get download statistics

## 💡 Usage Examples

### Creating a New Model Repository

```
Claude, create a new model repository called "my-username/my-awesome-model" with a public license and upload a README.md file with a description of the model.
```

### Building a Gradio Space

```
Claude, create a new Gradio space called "my-username/demo-app" and set up the basic files for a text classification demo.
```

### Editing Files with Precision

```
Claude, in my model repository "my-username/my-model", edit the README.md file and replace the text "This is a draft" with "This model is ready for production use".
```

### Managing Collections

```
Claude, create a new collection called "My AI Models" and add my top 3 models to it with descriptions.
```

### Exploring the Hub

```
Claude, search for the most popular text-to-image models on HuggingFace and show me their download stats.
```

## 🔧 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HUGGINGFACE_TOKEN` | Your HF API token | None | ✅ Yes |
| `HF_MCP_PERMISSIONS` | Permission level | `admin` | ❌ No |

## 🐛 Troubleshooting

### MCP Server Not Connecting

1. **Check the path**: Ensure the path to `main.py` is absolute and correct
2. **Verify token**: Test your HuggingFace token at [huggingface.co](https://huggingface.co)
3. **Check permissions**: Ensure your token has write access if needed
4. **View logs**: Check Claude Desktop logs for error messages

### Common Issues

**"Permission denied" errors:**
- Check your `HF_MCP_PERMISSIONS` setting
- Ensure your token has the required permissions

**"Repository not found" errors:**
- Verify the repository ID format: `username/repo-name`
- Check if the repository exists and you have access

**File editing issues:**
- Ensure `old_text` matches exactly (including whitespace)
- Use `hf_read_file` first to see the current content

### Log Files

Check these locations for MCP server logs:
- **macOS**: `~/Library/Logs/Claude/mcp-server-huggingmcp.log`
- **Windows**: `%APPDATA%/Claude/logs/mcp-server-huggingmcp.log`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [HuggingFace Hub](https://huggingface.co)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Desktop](https://claude.ai/desktop)

## ⚠️ Security Notes

- Keep your HuggingFace token secure and never share it
- Use appropriate permission levels for your use case
- Be cautious with `admin` permissions in shared environments
- Regular tokens are recommended over fine-grained tokens for broader access

---

**Made with ❤️ by ProCreations-Official**

Happy building with HuggingFace and Claude! 🎉