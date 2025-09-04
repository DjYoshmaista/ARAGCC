#!/usr/bin/env python3
"""Debug configuration parsing"""

import yaml
import os
from pprint import pprint

# Load system configuration like the orchestrator does
config_path = os.path.join(os.path.dirname(__file__), 'shared', 'configs', 'system.yaml')
print(f"Loading config from: {config_path}")

with open(config_path, 'r') as f:
    system_config = yaml.safe_load(f)

print("\nFull system config:")
pprint(system_config)

print("\nDatabase configuration:")
pprint(system_config.get('databases', {}))

print("\nPostgreSQL config:")
postgresql_config = system_config.get('databases', {}).get('postgresql', {})
pprint(postgresql_config)

print("\nQdrant config:")
qdrant_config = system_config.get('databases', {}).get('qdrant', {})
pprint(qdrant_config)

print("\nTesting DatabaseManager initialization:")
try:
    # Add the orchestrator directory to Python path for local imports
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'orchestrator'))
    
    from database_utils import DatabaseManager
    
    print("Creating DatabaseManager...")
    db_manager = DatabaseManager(postgresql_config, qdrant_config)
    
    print("Testing PostgreSQL connection...")
    result = db_manager.connect_postgresql()
    print(f"PostgreSQL connection result: {result}")
    
    if not result:
        print("PostgreSQL connection failed. Checking config values:")
        for key, value in postgresql_config.items():
            print(f"  {key}: {value} (type: {type(value)})")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
