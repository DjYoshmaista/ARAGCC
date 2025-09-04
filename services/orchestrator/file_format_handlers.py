"""
File format handlers for additional document types
Extends the existing document processing system with support for:
.csv, .tsv, .xls, .pdf, .docx, .doc, .json, .jsonl, .md, and .py files
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
import traceback

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Limited XLS/XLSX processing capabilities.")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
        PYPDF2_AVAILABLE = True
    except ImportError:
        PYPDF2_AVAILABLE = False
        logging.warning("PyPDF2 not available. PDF processing will be disabled.")

try:
    from docx import Document
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False
    logging.warning("python-docx not available. DOCX processing will be disabled.")

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False
    logging.warning("xlrd not available. XLS processing will be disabled.")

logger = logging.getLogger(__name__)

def process_csv_file(file_path: str, chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process CSV files and convert to structured text
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows per chunk
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        row_count = 0
        
        # Increase CSV field size limit to handle very large fields (10MB)
        try:
            csv.field_size_limit(10 * 1024 * 1024)  # Set to 10MB for very large Reddit posts
            logger.debug(f"CSV field size limit set to 10MB for {file_path}")
        except OverflowError:
            # If system can't handle 10MB, use maximum available
            import sys
            csv.field_size_limit(sys.maxsize)
            logger.debug(f"CSV field size limit set to system maximum for {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Detect delimiter with fallback options
            sample = f.read(1024)
            f.seek(0)
            
            delimiter = ','  # Default fallback
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            except csv.Error:
                # Try common delimiters
                for test_delimiter in [',', '\t', ';', '|', ' ']:
                    try:
                        f.seek(0)
                        test_reader = csv.reader(f, delimiter=test_delimiter)
                        first_row = next(test_reader)
                        if len(first_row) > 1:  # Multi-column indicates good delimiter
                            delimiter = test_delimiter
                            break
                    except (StopIteration, csv.Error):
                        continue
                f.seek(0)
            
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames or []
            
            for row in reader:
                row_count += 1
                
                # Handle very large fields by chunking them if necessary
                processed_fields = []
                for k, v in row.items():
                    if v and len(str(v)) > 5000:  # Chunk fields larger than 5KB
                        # Split very large fields into smaller chunks
                        field_chunks = [str(v)[i:i+5000] for i in range(0, len(str(v)), 5000)]
                        for i, chunk in enumerate(field_chunks):
                            processed_fields.append(f"{k}_part{i+1}: {chunk}")
                        logger.debug(f"Split large field '{k}' into {len(field_chunks)} chunks")
                    elif v:
                        processed_fields.append(f"{k}: {v}")
                
                # Convert row to readable text format
                row_text = ", ".join(processed_fields)
                
                if row_text.strip():
                    doc_chunk = {
                        'content': f"CSV Row {row_count}: {row_text}",
                        'chunk_index': row_count,
                        'metadata': {
                            'file_type': 'csv',
                            'row_number': row_count,
                            'headers': headers,
                            'has_large_fields': any(len(str(v)) > 5000 for v in row.values())
                        },
                        'tags': ['csv', 'tabular-data', 'row-data']
                    }
                    chunks.append(doc_chunk)
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= chunk_size:
                    yield chunks
                    chunks = []
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed CSV file {file_path} with {row_count} rows")
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {e}")
        yield [{'content': f"Error processing CSV file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_tsv_file(file_path: str, chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process TSV files (tab-separated values)
    
    Args:
        file_path: Path to the TSV file
        chunk_size: Number of rows per chunk
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        row_count = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter='\t')
            headers = reader.fieldnames or []
            
            for row in reader:
                row_count += 1
                # Convert row to readable text format
                row_text = "\t".join([f"{k}: {v}" for k, v in row.items() if v])
                
                if row_text.strip():
                    doc_chunk = {
                        'content': f"TSV Row {row_count}: {row_text}",
                        'chunk_index': row_count,
                        'metadata': {
                            'file_type': 'tsv',
                            'row_number': row_count,
                            'headers': headers
                        },
                        'tags': ['tsv', 'tabular-data', 'row-data']
                    }
                    chunks.append(doc_chunk)
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= chunk_size:
                    yield chunks
                    chunks = []
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed TSV file {file_path} with {row_count} rows")
        
    except Exception as e:
        logger.error(f"Error processing TSV file {file_path}: {e}")
        yield [{'content': f"Error processing TSV file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_xls_file(file_path: str, chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process XLS files (Excel 97-2003 format)
    
    Args:
        file_path: Path to the XLS file
        chunk_size: Number of rows per chunk
        
    Yields:
        List of document chunks with content and metadata
    """
    if not XLRD_AVAILABLE:
        error_msg = "xlrd library not available for XLS processing"
        logger.error(error_msg)
        yield [{'content': error_msg, 'chunk_index': 0, 'metadata': {'error': error_msg}, 'tags': ['error']}]
        return
    
    try:
        import xlrd
        file_path_obj = Path(file_path)
        chunks = []
        row_count = 0
        
        workbook = xlrd.open_workbook(file_path)
        
        for sheet_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(sheet_name)
            
            # Get headers from first row if possible
            headers = []
            if sheet.nrows > 0:
                headers = [str(sheet.cell_value(0, col)) for col in range(sheet.ncols)]
            
            # Process data rows
            start_row = 1 if headers else 0
            for row_idx in range(start_row, sheet.nrows):
                row_count += 1
                row_data = []
                
                for col_idx in range(sheet.ncols):
                    cell_value = sheet.cell_value(row_idx, col_idx)
                    header = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                    row_data.append(f"{header}: {cell_value}")
                
                row_text = ", ".join(row_data)
                
                if row_text.strip():
                    doc_chunk = {
                        'content': f"XLS Sheet '{sheet_name}' Row {row_count}: {row_text}",
                        'chunk_index': row_count,
                        'metadata': {
                            'file_type': 'xls',
                            'sheet_name': sheet_name,
                            'row_number': row_count,
                            'headers': headers
                        },
                        'tags': ['xls', 'excel', 'tabular-data', 'spreadsheet']
                    }
                    chunks.append(doc_chunk)
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= chunk_size:
                    yield chunks
                    chunks = []
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed XLS file {file_path} with {row_count} rows across {len(workbook.sheet_names())} sheets")
        
    except Exception as e:
        logger.error(f"Error processing XLS file {file_path}: {e}")
        yield [{'content': f"Error processing XLS file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_pdf_file(file_path: str, chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process PDF files and extract text content
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Number of characters per chunk (approx)
        
    Yields:
        List of document chunks with content and metadata
    """
    if not PYPDF2_AVAILABLE:
        error_msg = "PyPDF2 library not available for PDF processing"
        logger.error(error_msg)
        yield [{'content': error_msg, 'chunk_index': 0, 'metadata': {'error': error_msg}, 'tags': ['error']}]
        return
    
    try:
        file_path_obj = Path(file_path)
        chunks = []
        
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1} of {file_path}: {e}")
        
        # Split text into chunks
        if full_text:
            # Simple chunking by character count
            start = 0
            chunk_index = 0
            
            while start < len(full_text):
                end = min(start + chunk_size, len(full_text))
                chunk_text = full_text[start:end]
                
                if chunk_text.strip():
                    doc_chunk = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'file_type': 'pdf',
                            'total_pages': total_pages,
                            'chunk_size': len(chunk_text)
                        },
                        'tags': ['pdf', 'document', 'text']
                    }
                    chunks.append(doc_chunk)
                    chunk_index += 1
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= 10:  # Limit chunks per yield
                    yield chunks
                    chunks = []
                
                start = end
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed PDF file {file_path} with {total_pages} pages")
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        yield [{'content': f"Error processing PDF file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_docx_file(file_path: str, chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process DOCX files (Word documents)
    
    Args:
        file_path: Path to the DOCX file
        chunk_size: Number of characters per chunk (approx)
        
    Yields:
        List of document chunks with content and metadata
    """
    if not PYTHON_DOCX_AVAILABLE:
        error_msg = "python-docx library not available for DOCX processing"
        logger.error(error_msg)
        yield [{'content': error_msg, 'chunk_index': 0, 'metadata': {'error': error_msg}, 'tags': ['error']}]
        return
    
    try:
        file_path_obj = Path(file_path)
        chunks = []
        
        doc = Document(file_path)
        
        full_text = ""
        paragraph_count = 0
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text += f"{paragraph.text}\n"
                paragraph_count += 1
        
        # Extract text from tables
        table_count = 0
        for table in doc.tables:
            table_count += 1
            full_text += f"\n--- Table {table_count} ---\n"
            for row in table.rows:
                row_text = " | ".join([cell.text for cell in row.cells])
                if row_text.strip():
                    full_text += f"{row_text}\n"
        
        # Split text into chunks
        if full_text:
            # Simple chunking by character count
            start = 0
            chunk_index = 0
            
            while start < len(full_text):
                end = min(start + chunk_size, len(full_text))
                chunk_text = full_text[start:end]
                
                if chunk_text.strip():
                    doc_chunk = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'file_type': 'docx',
                            'paragraphs': paragraph_count,
                            'tables': table_count,
                            'chunk_size': len(chunk_text)
                        },
                        'tags': ['docx', 'word', 'document']
                    }
                    chunks.append(doc_chunk)
                    chunk_index += 1
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= 10:  # Limit chunks per yield
                    yield chunks
                    chunks = []
                
                start = end
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed DOCX file {file_path} with {paragraph_count} paragraphs and {table_count} tables")
        
    except Exception as e:
        logger.error(f"Error processing DOCX file {file_path}: {e}")
        yield [{'content': f"Error processing DOCX file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_json_file(file_path: str, chunk_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process JSON files
    
    Args:
        file_path: Path to the JSON file
        chunk_size: Number of JSON objects per chunk
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        object_count = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                for i, item in enumerate(data):
                    object_count += 1
                    item_text = json.dumps(item, indent=2)
                    
                    doc_chunk = {
                        'content': f"JSON Object {i+1}:\n{item_text}",
                        'chunk_index': object_count,
                        'metadata': {
                            'file_type': 'json',
                            'object_index': i,
                            'object_type': type(item).__name__
                        },
                        'tags': ['json', 'structured-data']
                    }
                    chunks.append(doc_chunk)
                    
                    # Yield chunks when we reach the chunk size
                    if len(chunks) >= chunk_size:
                        yield chunks
                        chunks = []
            
            elif isinstance(data, dict):
                # Single object
                object_count = 1
                item_text = json.dumps(data, indent=2)
                
                doc_chunk = {
                    'content': f"JSON Object:\n{item_text}",
                    'chunk_index': object_count,
                    'metadata': {
                        'file_type': 'json',
                        'object_type': 'dict'
                    },
                    'tags': ['json', 'structured-data']
                }
                chunks.append(doc_chunk)
            
            else:
                # Primitive value
                object_count = 1
                doc_chunk = {
                    'content': f"JSON Value: {data}",
                    'chunk_index': object_count,
                    'metadata': {
                        'file_type': 'json',
                        'object_type': type(data).__name__
                    },
                    'tags': ['json', 'structured-data']
                }
                chunks.append(doc_chunk)
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed JSON file {file_path} with {object_count} objects")
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        yield [{'content': f"Invalid JSON file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]
    except Exception as e:
        logger.error(f"Error processing JSON file {file_path}: {e}")
        yield [{'content': f"Error processing JSON file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_jsonl_file(file_path: str, chunk_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process JSONL files (JSON Lines - one JSON object per line)
    
    Args:
        file_path: Path to the JSONL file
        chunk_size: Number of JSON objects per chunk
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        line_count = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    line_count += 1
                    
                    item_text = json.dumps(data, indent=2)
                    
                    doc_chunk = {
                        'content': f"JSONL Line {line_num}:\n{item_text}",
                        'chunk_index': line_count,
                        'metadata': {
                            'file_type': 'jsonl',
                            'line_number': line_num,
                            'object_type': type(data).__name__
                        },
                        'tags': ['jsonl', 'structured-data', 'line-delimited']
                    }
                    chunks.append(doc_chunk)
                    
                    # Yield chunks when we reach the chunk size
                    if len(chunks) >= chunk_size:
                        yield chunks
                        chunks = []
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} of {file_path}: {e}")
                    continue
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed JSONL file {file_path} with {line_count} valid JSON objects")
        
    except Exception as e:
        logger.error(f"Error processing JSONL file {file_path}: {e}")
        yield [{'content': f"Error processing JSONL file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_md_file(file_path: str, chunk_size: int = 2000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process Markdown files
    
    Args:
        file_path: Path to the Markdown file
        chunk_size: Number of characters per chunk (approx)
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Simple chunking for markdown
        if content:
            start = 0
            chunk_index = 0
            
            while start < len(content):
                end = min(start + chunk_size, len(content))
                
                # Try to break at section boundaries
                if end < len(content):
                    # Look for section breaks (double newlines or headers)
                    section_breaks = ['\n\n', '\n#', '\n##', '\n###']
                    best_break = end
                    
                    for break_type in section_breaks:
                        break_pos = content.rfind(break_type, start, end)
                        if break_pos > start + chunk_size // 2:  # Only break if we're past halfway
                            best_break = break_pos + len(break_type)
                            break
                    
                    end = best_break
                
                chunk_text = content[start:end]
                
                if chunk_text.strip():
                    doc_chunk = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'file_type': 'md',
                            'chunk_size': len(chunk_text)
                        },
                        'tags': ['markdown', 'documentation']
                    }
                    chunks.append(doc_chunk)
                    chunk_index += 1
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= 5:  # Limit chunks per yield
                    yield chunks
                    chunks = []
                
                start = end
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed Markdown file {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing Markdown file {file_path}: {e}")
        yield [{'content': f"Error processing Markdown file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

def process_py_file(file_path: str, chunk_size: int = 2000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Process Python files with code-aware chunking
    
    Args:
        file_path: Path to the Python file
        chunk_size: Number of characters per chunk (approx)
        
    Yields:
        List of document chunks with content and metadata
    """
    try:
        file_path_obj = Path(file_path)
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Simple chunking for Python code
        if content:
            start = 0
            chunk_index = 0
            
            while start < len(content):
                end = min(start + chunk_size, len(content))
                
                # Try to break at logical boundaries
                if end < len(content):
                    # Look for function/class boundaries or empty lines
                    section_breaks = ['\n\nclass ', '\n\ndef ', '\n\nif __name__']
                    best_break = end
                    
                    for break_type in section_breaks:
                        break_pos = content.rfind(break_type, start, end)
                        if break_pos > start + chunk_size // 2:  # Only break if we're past halfway
                            best_break = break_pos
                            break
                    
                    # If no logical break, try to break at empty lines
                    if best_break == end:
                        break_pos = content.rfind('\n\n', start, end)
                        if break_pos > start + chunk_size // 2:
                            best_break = break_pos + 2
                    
                    end = best_break
                
                chunk_text = content[start:end]
                
                if chunk_text.strip():
                    doc_chunk = {
                        'content': chunk_text,
                        'chunk_index': chunk_index,
                        'metadata': {
                            'file_type': 'py',
                            'chunk_size': len(chunk_text)
                        },
                        'tags': ['python', 'code', 'source']
                    }
                    chunks.append(doc_chunk)
                    chunk_index += 1
                
                # Yield chunks when we reach the chunk size
                if len(chunks) >= 5:  # Limit chunks per yield
                    yield chunks
                    chunks = []
                
                start = end
        
        # Yield any remaining chunks
        if chunks:
            yield chunks
            
        logger.debug(f"Processed Python file {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing Python file {file_path}: {e}")
        yield [{'content': f"Error processing Python file: {str(e)}", 'chunk_index': 0, 'metadata': {'error': str(e)}, 'tags': ['error']}]

# File format handler registry
FILE_FORMAT_HANDLERS = {
    'csv': process_csv_file,
    'tsv': process_tsv_file,
    'xls': process_xls_file,
    'pdf': process_pdf_file,
    'docx': process_docx_file,
    'json': process_json_file,
    'jsonl': process_jsonl_file,
    'md': process_md_file,
    'py': process_py_file
}

def get_file_handler(file_extension: str):
    """
    Get the appropriate handler function for a file extension
    
    Args:
        file_extension: File extension without the dot (e.g., 'csv', 'pdf')
        
    Returns:
        Handler function or None if not supported
    """
    return FILE_FORMAT_HANDLERS.get(file_extension.lower())