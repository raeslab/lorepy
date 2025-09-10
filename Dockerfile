# Use Node.js LTS on Debian Trixie (slim variant for smaller image size)
FROM node:lts-trixie-slim

# Install system dependencies needed for development and VS Code devcontainer
# - git: version control (essential for most development workflows)
# - curl: downloading files and API calls
# - ca-certificates: SSL certificate validation
# - gnupg: GPG key management
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
# Using --no-fund and --no-audit flags to reduce installation noise
RUN npm install -g @anthropic-ai/claude-code --no-fund --no-audit

# Set the working directory
WORKDIR /workspace

# Verify Claude Code installation
RUN claude --version

# Keep container running for devcontainer usage
CMD ["sleep", "infinity"]