# Use Python 3.12 slim image as the base
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install Node.js (required for MCP TypeScript server)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install MCP dependencies (including Express)
COPY package.json .
RUN npm install

# Copy EVERYTHING to ensure subdirectories like mcp/ and core/ are preserved
COPY . .

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Expose both the FastAPI port and the MCP-SSE port
EXPOSE 8000
EXPOSE 3001

# Default command remains the MCP loader
CMD ["npx", "tsx", "mcp/mcp_server.ts", "--transport=sse"]
