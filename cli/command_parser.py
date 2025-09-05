"""
Command Parser for the AgenticRAG CLI
"""

import argparse
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class CLIConfig:
    """CLI configuration structure"""
    # Service endpoints
    orchestrator_url: str = "http://localhost:8001"
    model_gateway_url: str = "http://localhost:8070"
    vector_engine_url: str = "http://localhost:8080"
    
    # Interface settings
    verbose: bool = False
    color_output: bool = True

class CommandParser:
    """Command line argument parser"""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            prog="agentic-rag",
            description="Advanced Multi-Agent RAG System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument('--version', action='version', version='agentic-rag 1.0.0')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--no-color', action='store_true', help='Disable colored output')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Query command
        query_parser = subparsers.add_parser('query', help='Submit a query to the system')
        query_parser.add_argument('prompt', help='The query prompt')
        
        # Ingest command
        ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into RAG database')
        ingest_parser.add_argument('paths', nargs='+', help='Files or directories to ingest')
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments"""
        return self.parser.parse_args(args)