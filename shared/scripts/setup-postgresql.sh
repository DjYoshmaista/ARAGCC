#!/bin/bash

# PostgreSQL Database Setup Script for AgenticRAG System
# This script creates the required database and tables for the AgenticRAG system

set -e

# Database configuration from system.yaml
DB_HOST="localhost"
DB_PORT=5432
DB_NAME="agentic_system"
DB_USER="postgres"
DB_PASSWORD="password"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up PostgreSQL database for AgenticRAG system...${NC}"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}Error: PostgreSQL is not installed.${NC}"
    echo "Please install PostgreSQL first:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "  CentOS/RHEL: sudo yum install postgresql-server postgresql-contrib"
    echo "  macOS: brew install postgresql"
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
    echo -e "${YELLOW}Warning: PostgreSQL is not running or not accessible at $DB_HOST:$DB_PORT${NC}"
    echo "Please start PostgreSQL service:"
    echo "  Ubuntu/Debian: sudo systemctl start postgresql"
    echo "  CentOS/RHEL: sudo systemctl start postgresql"
    echo "  macOS: brew services start postgresql"
    exit 1
fi

# Create database
echo -e "${YELLOW}Creating database '$DB_NAME'...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || {
    echo -e "${YELLOW}Database '$DB_NAME' already exists or creation failed.${NC}"
}

# Connect to the database and create tables
echo -e "${YELLOW}Creating tables...${NC}"

# Create documents table
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    folder_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_chunks INTEGER NOT NULL DEFAULT 0
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    overlap_tokens INTEGER NOT NULL DEFAULT 0,
    previous_chunk_id VARCHAR(255),
    next_chunk_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_number ON chunks(chunk_number);
CREATE INDEX IF NOT EXISTS idx_documents_folder_path ON documents(folder_path);
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ language 'plpgsql';

-- Create trigger to automatically update updated_at
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON TABLE documents TO $DB_USER;
GRANT ALL PRIVILEGES ON TABLE chunks TO $DB_USER;
GRANT ALL PRIVILEGES ON SEQUENCE documents_id_seq TO $DB_USER;
GRANT ALL PRIVILEGES ON SEQUENCE chunks_id_seq TO $DB_USER;

-- Show table information
\d documents
\d chunks
EOF

echo -e "${GREEN}PostgreSQL database setup completed successfully!${NC}"
echo -e "${GREEN}Database: $DB_NAME${NC}"
echo -e "${GREEN}Host: $DB_HOST:$DB_PORT${NC}"
echo -e "${GREEN}User: $DB_USER${NC}"

# Test connection
echo -e "${YELLOW}Testing database connection...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null && {
    echo -e "${GREEN}Database connection test successful!${NC}"
} || {
    echo -e "${RED}Database connection test failed.${NC}"
    exit 1
}

echo -e "${GREEN}PostgreSQL setup complete!${NC}"