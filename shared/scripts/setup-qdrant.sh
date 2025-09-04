#!/bin/bash

# Qdrant Database Setup Script for AgenticRAG System
# This script creates the required collection for the AgenticRAG system

set -e

# Qdrant configuration from system.yaml
QDRANT_HOST="localhost"
QDRANT_PORT=6333
COLLECTION_NAME="embeddings"
VECTOR_SIZE=640  # For dengcaoQwen3-Embedding-0.6B model

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Qdrant collection for AgenticRAG system...${NC}"

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed.${NC}"
    exit 1
fi

# Check if Qdrant is running
if ! curl -s http://$QDRANT_HOST:$QDRANT_PORT &> /dev/null; then
    echo -e "${RED}Error: Qdrant is not running or not accessible at $QDRANT_HOST:$QDRANT_PORT${NC}"
    echo "Please start Qdrant service:"
    echo "  Docker: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
    echo "  Binary: Download from https://github.com/qdrant/qdrant/releases"
    exit 1
fi

# Check Qdrant version
echo -e "${YELLOW}Checking Qdrant version...${NC}"
VERSION_RESPONSE=$(curl -s http://$QDRANT_HOST:$QDRANT_PORT)
echo "Qdrant response: $VERSION_RESPONSE"

# Create collection
echo -e "${YELLOW}Creating collection '$COLLECTION_NAME'...${NC}"

# Delete collection if it exists
curl -s -X DELETE http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME > /dev/null 2>&1 || true

# Create new collection
curl -X PUT "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME" \
    -H "Content-Type: application/json" \
    -d "{
        \"vectors\": {
            \"size\": $VECTOR_SIZE,
            \"distance\": \"Cosine\"
        }
    }" > /dev/null && {
    echo -e "${GREEN}Collection '$COLLECTION_NAME' created successfully!${NC}"
} || {
    echo -e "${RED}Failed to create collection '$COLLECTION_NAME'.${NC}"
    exit 1
}

# Verify collection creation
echo -e "${YELLOW}Verifying collection...${NC}"
COLLECTION_INFO=$(curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME")
echo "Collection info: $COLLECTION_INFO"

# Create index optimization settings
echo -e "${YELLOW}Optimizing collection settings...${NC}"
curl -X PATCH "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME" \
    -H "Content-Type: application/json" \
    -d "{
        \"optimizers_config\": {
            \"memmap_threshold\": 20000,
            \"indexing_threshold\": 20000
        }
    }" > /dev/null && {
    echo -e "${GREEN}Collection optimization settings applied!${NC}"
} || {
    echo -e "${YELLOW}Warning: Could not apply optimization settings.${NC}"
}

echo -e "${GREEN}Qdrant setup complete!${NC}"
echo -e "${GREEN}Collection: $COLLECTION_NAME${NC}"
echo -e "${GREEN}Host: $QDRANT_HOST:$QDRANT_PORT${NC}"
echo -e "${GREEN}Vector size: $VECTOR_SIZE${NC}"
echo -e "${GREEN}Distance metric: Cosine${NC}"