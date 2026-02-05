# Use Python 3.11 slim image as the base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Node.js (required for MCP TypeScript server)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install MCP dependencies
COPY package.json .
RUN npm install

# Copy source files
COPY *.py ./
COPY *.ts ./

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Expose the MCP server via stdio
# Note: MCP servers over stdio don't need to EXPOSE ports
CMD ["npx", "tsx", "mcp_server.ts"]
