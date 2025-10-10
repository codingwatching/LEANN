# Slack Integration Setup Guide

This comprehensive guide will walk you through setting up Slack integration with LEANN, including troubleshooting common issues like the "users cache is not ready yet" error.

## Overview

LEANN's Slack integration uses MCP (Model Context Protocol) servers to fetch and index your Slack messages for RAG (Retrieval-Augmented Generation). This allows you to search through your Slack conversations using natural language queries.

## Prerequisites

1. **Slack Workspace Access**: You need admin or owner permissions in your Slack workspace to create apps and configure OAuth tokens.

2. **Slack MCP Server**: Install a Slack MCP server (e.g., `slack-mcp-server` via npm)

3. **LEANN**: Ensure you have LEANN installed and working

## Step 1: Create a Slack App

### 1.1 Go to Slack API Dashboard

1. Visit [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Enter your app name (e.g., "LEANN Slack Integration")
5. Select your workspace
6. Click **"Create App"**

### 1.2 Configure App Permissions

#### Bot Token Scopes

1. In your app dashboard, go to **"OAuth & Permissions"** in the left sidebar
2. Scroll down to **"Scopes"** section
3. Under **"Bot Token Scopes"**, click **"Add an OAuth Scope"**
4. Add the following scopes:
   - `channels:read` - Read public channel information
   - `channels:history` - Read messages in public channels
   - `groups:read` - Read private channel information
   - `groups:history` - Read messages in private channels
   - `im:read` - Read direct message information
   - `im:history` - Read direct messages
   - `mpim:read` - Read group direct message information
   - `mpim:history` - Read group direct messages
   - `users:read` - Read user information
   - `team:read` - Read workspace information

#### App-Level Tokens (Optional)

Some MCP servers may require app-level tokens:

1. Go to **"Basic Information"** in the left sidebar
2. Scroll down to **"App-Level Tokens"**
3. Click **"Generate Token and Scopes"**
4. Enter a name (e.g., "LEANN Integration")
5. Add the `connections:write` scope
6. Click **"Generate"**
7. Copy the token (starts with `xapp-`)

### 1.3 Install App to Workspace

1. Go to **"OAuth & Permissions"** in the left sidebar
2. Click **"Install to Workspace"**
3. Review the permissions and click **"Allow"**
4. Copy the **"Bot User OAuth Token"** (starts with `xoxb-`)

## Step 2: Install Slack MCP Server

### Option A: Using npm (Recommended)

```bash
# Install globally
npm install -g slack-mcp-server

# Or install locally
npm install slack-mcp-server
```

### Option B: Using npx (No installation required)

```bash
# Use directly without installation
npx slack-mcp-server
```

## Step 3: Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required: Bot User OAuth Token
SLACK_BOT_TOKEN=xoxb-your-bot-token-here

# Optional: App-Level Token (if your MCP server requires it)
SLACK_APP_TOKEN=xapp-your-app-token-here

# Optional: Workspace-specific settings
SLACK_WORKSPACE_ID=T1234567890  # Your workspace ID (optional)
```

## Step 4: Test the Setup

### 4.1 Test MCP Server Connection

```bash
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --test-connection \
  --workspace-name "Your Workspace Name"
```

This will test the connection and list available tools without indexing any data.

### 4.2 Index a Specific Channel

```bash
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "Your Workspace Name" \
  --channels general \
  --query "What did we discuss about the project?"
```

## Common Issues and Solutions

### Issue 1: "users cache is not ready yet" Error

**Problem**: You see this warning:
```
WARNING - Failed to fetch messages from channel random: Failed to fetch messages: {'code': -32603, 'message': 'users cache is not ready yet, sync process is still running... please wait'}
```

**Solution**: This is a common timing issue. The LEANN integration now includes automatic retry logic:

1. **Wait and Retry**: The system will automatically retry with exponential backoff (2s, 4s, 8s, etc.)
2. **Increase Retry Parameters**: If needed, you can customize retry behavior:
   ```bash
   python -m apps.slack_rag \
     --mcp-server "slack-mcp-server" \
     --max-retries 10 \
     --retry-delay 3.0 \
     --channels general \
     --query "Your query here"
   ```
3. **Keep MCP Server Running**: Start the MCP server separately and keep it running:
   ```bash
   # Terminal 1: Start MCP server
   slack-mcp-server
   
   # Terminal 2: Run LEANN (it will connect to the running server)
   python -m apps.slack_rag --mcp-server "slack-mcp-server" --channels general --query "test"
   ```

### Issue 2: "No message fetching tool found"

**Problem**: The MCP server doesn't have the expected tools.

**Solution**: 
1. Check if your MCP server is properly installed and configured
2. Verify your Slack tokens are correct
3. Try a different MCP server implementation
4. Check the MCP server documentation for required configuration

### Issue 3: Permission Denied Errors

**Problem**: You get permission errors when trying to access channels.

**Solutions**:
1. **Check Bot Permissions**: Ensure your bot has been added to the channels you want to access
2. **Verify Token Scopes**: Make sure you have all required scopes configured
3. **Channel Access**: For private channels, the bot needs to be explicitly invited
4. **Workspace Permissions**: Ensure your Slack app has the necessary workspace permissions

### Issue 4: Empty Results

**Problem**: No messages are returned even though the channel has messages.

**Solutions**:
1. **Check Channel Names**: Ensure channel names are correct (without the # symbol)
2. **Verify Bot Access**: Make sure the bot can access the channels
3. **Check Date Ranges**: Some MCP servers have limitations on message history
4. **Increase Message Limits**: Try increasing the message limit:
   ```bash
   python -m apps.slack_rag \
     --mcp-server "slack-mcp-server" \
     --channels general \
     --max-messages-per-channel 1000 \
     --query "test"
   ```

## Advanced Configuration

### Custom MCP Server Commands

If you need to pass additional parameters to your MCP server:

```bash
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server --token-file /path/to/tokens.json" \
  --workspace-name "Your Workspace" \
  --channels general \
  --query "Your query"
```

### Multiple Workspaces

To work with multiple Slack workspaces, you can:

1. Create separate apps for each workspace
2. Use different environment variables
3. Run separate instances with different configurations

### Performance Optimization

For better performance with large workspaces:

```bash
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "Your Workspace" \
  --max-messages-per-channel 500 \
  --no-concatenate-conversations \
  --query "Your query"
```

## Troubleshooting Checklist

- [ ] Slack app created with proper permissions
- [ ] Bot token (xoxb-) copied correctly
- [ ] App-level token (xapp-) created if needed
- [ ] MCP server installed and accessible
- [ ] Environment variables set correctly
- [ ] Bot invited to relevant channels
- [ ] Channel names specified without # symbol
- [ ] Sufficient retry attempts configured
- [ ] Network connectivity to Slack APIs

## Getting Help

If you continue to have issues:

1. **Check Logs**: Look for detailed error messages in the console output
2. **Test MCP Server**: Use `--test-connection` to verify the MCP server is working
3. **Verify Tokens**: Double-check that your Slack tokens are valid and have the right scopes
4. **Community Support**: Reach out to the LEANN community for help

## Example Commands

### Basic Usage
```bash
# Test connection
python -m apps.slack_rag --mcp-server "slack-mcp-server" --test-connection

# Index specific channels
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "My Company" \
  --channels general random \
  --query "What did we decide about the project timeline?"
```

### Advanced Usage
```bash
# With custom retry settings
python -m apps.slack_rag \
  --mcp-server "slack-mcp-server" \
  --workspace-name "My Company" \
  --channels general \
  --max-retries 10 \
  --retry-delay 5.0 \
  --max-messages-per-channel 2000 \
  --query "Show me all decisions made in the last month"
```

This guide should help you get Slack integration working smoothly with LEANN. The retry logic improvements should significantly reduce the "cache not ready" errors you were experiencing.
