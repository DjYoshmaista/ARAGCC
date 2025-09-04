#!/bin/bash

# Combined Database Setup Script for AgenticRAG System
# This script sets up both PostgreSQL and Qdrant databases for the AgenticRAG system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}  AgenticRAG Database Setup Script${NC}"
echo -e "${BLUE}==============================================${NC}"
echo

# Function to print section headers
print_section() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check if a service is running
check_service() {
    if command_exists curl; then
        curl -s "$1" &> /dev/null
        return $?
    else
        return 1
    fi
}

print_section "Checking Prerequisites"

# Check if required tools are installed
MISSING_TOOLS=()
for tool in psql curl; do
    if ! command_exists $tool; then
        MISSING_TOOLS+=($tool)
    fi
done

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing required tools: ${MISSING_TOOLS[*]}${NC}"
    echo "Please install the missing tools:"
    for tool in "${MISSING_TOOLS[@]}"; do
        case $tool in
            psql)
                echo "  PostgreSQL client: sudo apt-get install postgresql-client"
                ;;
            curl)
                echo "  curl: sudo apt-get install curl"
                ;;
        esac
    done
    exit 1
fi

echo -e "${GREEN}All required tools are installed!${NC}"

print_section "Setting Up PostgreSQL Database"

# Source the PostgreSQL setup script
if [ -f "./setup-postgresql.sh" ]; then
    echo -e "${YELLOW}Running PostgreSQL setup...${NC}"
    ./setup-postgresql.sh
elif [ -f "/home/yosh/repos/AgenticRAGQCode/shared/scripts/setup-postgresql.sh" ]; then
    echo -e "${YELLOW}Running PostgreSQL setup...${NC}"
    /home/yosh/repos/AgenticRAGQCode/shared/scripts/setup-postgresql.sh
else
    echo -e "${RED}Error: PostgreSQL setup script not found${NC}"
    exit 1
fi

print_section "Setting Up Qdrant Database"

# Source the Qdrant setup script
if [ -f "./setup-qdrant.sh" ]; then
    echo -e "${YELLOW}Running Qdrant setup...${NC}"
    ./setup-qdrant.sh
elif [ -f "/home/yosh/repos/AgenticRAGQCode/shared/scripts/setup-qdrant.sh" ]; then
    echo -e "${YELLOW}Running Qdrant setup...${NC}"
    /home/yosh/repos/AgenticRAGQCode/shared/scripts/setup-qdrant.sh
else
    echo -e "${RED}Error: Qdrant setup script not found${NC}"
    exit 1
fi

print_section "Verifying Setup"

# Test PostgreSQL connection
echo -e "${YELLOW}Testing PostgreSQL connection...${NC}"
if PGPASSWORD=password psql -h localhost -p 5432 -U postgres -d agentic_system -c "SELECT 1;" &> /dev/null; then
    echo -e "${GREEN}PostgreSQL connection test successful!${NC}"
else
    echo -e "${RED}PostgreSQL connection test failed.${NC}"
fi

# Test Qdrant connection
echo -e "${YELLOW}Testing Qdrant connection...${NC}"
if curl -s http://localhost:6333/collections/embeddings &> /dev/null; then
    echo -e "${GREEN}Qdrant connection test successful!${NC}"
else
    echo -e "${RED}Qdrant connection test failed.${NC}"
fi

print_section "Setup Complete"

echo -e "${GREEN}Database setup completed successfully!${NC}"
echo
echo "PostgreSQL:"
echo "  - Database: agentic_system"
echo "  - Host: localhost:5432"
echo "  - User: postgres"
echo "  - Password: password"
echo
echo "Qdrant:"
echo "  - Collection: embeddings"
echo "  - Host: localhost:6333"
echo "  - Vector size: 640"
echo "  - Distance: Cosine"
echo
echo -e "${BLUE}==============================================${NC}"