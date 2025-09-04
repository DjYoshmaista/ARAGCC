#!/bin/bash

# Initialize Qdrant collection for AgenticRAG system
# This script should be run after Qdrant is started

set -e

# Configuration
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION_NAME="${COLLECTION_NAME:-embeddings}"
VECTOR_SIZE="${VECTOR_SIZE:-640}"  # For dengcaoQwen3-Embedding-0.6B

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Initializing Qdrant collection...${NC}"

# Wait for Qdrant to be ready
echo -e "${YELLOW}Waiting for Qdrant to be ready...${NC}"
for i in {1..30}; do
    if curl -s "http://$QDRANT_HOST:$QDRANT_PORT" &> /dev/null; then
        echo -e "${GREEN}Qdrant is ready!${NC}"
        break
    fi
    echo -e "${YELLOW}Waiting for Qdrant (attempt $i/30)...${NC}"
    sleep 2
done

# Check if Qdrant is accessible
if ! curl -s "http://$QDRANT_HOST:$QDRANT_PORT" &> /dev/null; then
    echo -e "${RED}Error: Qdrant is not accessible at $QDRANT_HOST:$QDRANT_PORT${NC}"
    exit 1
fi

# Delete collection if it exists
echo -e "${YELLOW}Deleting existing collection (if any)...${NC}"
curl -s -X DELETE "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME" > /dev/null 2>&1 || true

# Create collection
echo -e "${YELLOW}Creating collection '$COLLECTION_NAME'...${NC}"
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

# Apply optimization settings
echo -e "${YELLOW}Applying optimization settings...${NC}"
curl -X PATCH "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME" \
    -H "Content-Type: application/json" \
    -d "{
        \"optimizers_config\": {
            \"memmap_threshold\": 20000,
            \"indexing_threshold\": 20000
        }
    }" > /dev/null && {
    echo -e "${GREEN}Optimization settings applied!${NC}"
} || {
    echo -e "${YELLOW}Warning: Could not apply optimization settings.${NC}"
}

# Verify collection
echo -e "${YELLOW}Verifying collection...${NC}"
COLLECTION_INFO=$(curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME")
if echo "$COLLECTION_INFO" | grep -q "\"status\":\"green\""; then
    echo -e "${GREEN}Collection verification successful!${NC}"
else
    echo -e "${YELLOW}Collection created but status may need attention.${NC}"
fi

echo -e "${GREEN}Qdrant initialization complete!${NC}"
echo -e "${GREEN}Collection: $COLLECTION_NAME${NC}"
echo -e "${GREEN}Vector size: $VECTOR_SIZE${NC}"
echo -e "${GREEN}Distance metric: Cosine${NC}"