#!/usr/bin/env python3
"""Test database connection with same configuration as orchestrator"""

import os
import sys
import yaml
import psycopg2
import qdrant_client

# Add orchestrator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))

def load_config():
    """Load system configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'shared', 'configs', 'system.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_postgresql():
    """Test PostgreSQL connection"""
    config = load_config()
    pg_config = config['databases']['postgresql']
    
    try:
        connection = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password']
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        
        print(f"✓ PostgreSQL connection successful: {result}")
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

def test_qdrant():
    """Test Qdrant connection"""
    config = load_config()
    qdrant_config = config['databases']['qdrant']
    
    try:
        client = qdrant_client.QdrantClient(
            host=qdrant_config['host'],
            port=qdrant_config['port']
        )
        
        # Test connection
        collections = client.get_collections()
        print(f"✓ Qdrant connection successful, collections: {len(collections.collections)}")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def main():
    print("Testing database connections...")
    
    pg_ok = test_postgresql()
    qdrant_ok = test_qdrant()
    
    if pg_ok and qdrant_ok:
        print("✅ All database connections working")
        return 0
    else:
        print("❌ Some database connections failed")
        return 1

if __name__ == "__main__":
    exit(main())