#!/usr/bin/env python3
"""
AgenticRAG CLI - Modular Version
"""

import asyncio
import sys
from cli.config_manager import ConfigManager, CLIConfig
from cli.command_parser import CommandParser
from cli.api_client import APIClient

async def handle_query_command(args, config_manager: ConfigManager, api_client: APIClient):
    """Handle query command"""
    print(f"Query: {args.prompt}")
    print("This command will be fully implemented in the next iteration.")

async def handle_ingest_command(args, config_manager: ConfigManager, api_client: APIClient):
    """Handle ingestion command"""
    print(f"Ingesting: {args.paths}")
    print("This command will be fully implemented in the next iteration.")

async def handle_status_command(args, config_manager: ConfigManager, api_client: APIClient):
    """Handle status command"""
    print("Checking system status...")
    health = await api_client.health_check()
    
    for service, status in health.items():
        status_icon = "✅" if status.get('status') == 'healthy' else "❌"
        print(f"{status_icon} {service}: {status.get('status', 'unknown')}")

async def main():
    """Main CLI entry point"""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    parser = CommandParser(config)
    api_client = APIClient(config)
    
    try:
        args = parser.parse_args()
        
        # Update config based on args
        if args.verbose:
            config.verbose = True
        
        # Handle commands
        if not args.command:
            parser.parser.print_help()
            return
        
        if args.command == 'query':
            await handle_query_command(args, config_manager, api_client)
        elif args.command == 'ingest':
            await handle_ingest_command(args, config_manager, api_client)
        elif args.command == 'status':
            await handle_status_command(args, config_manager, api_client)
        else:
            print(f"Command '{args.command}' is not yet implemented.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await api_client.close()

if __name__ == "__main__":
    asyncio.run(main())