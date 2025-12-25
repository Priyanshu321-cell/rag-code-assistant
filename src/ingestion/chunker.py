from src.ingestion.parser import parse_directory
from typing import List,Union
from loguru import logger
from pathlib import Path

def chunk_by_function(functions)->List[dict]:
    """One function one chunk"""
    chunks = []
    for i, function in enumerate(functions):
        chunk = {
            'id': f'chunk_{i}',
            'text': f"Function: {function['name']}\nDoc:{function['docstring']}\nCode:{function['code'][:500] if len(function['code']) > 500 else function['code']}",
            'metadata': {'file':function['file'],'line':function['line_start'],'function':function['name'],
                         'is_async':function['type']=='async'}
        }
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks from {len(functions)} functions")
    return chunks


def format_function_chunk(function_info, include_code:bool = False, max_length=None):
    """Format a single function into text"""
    parts = []
    async_func = "async " if function_info['type'] == 'async' else ""
    parts.append(f"{async_func}def {function_info['name']}()")
    if function_info['docstring']:
        parts.append(f"Docs : {function_info['docstring']}")
    filename = Path(function_info['file']).name
    parts.append(f"File : {filename}")
    
    if include_code and function_info['code']:
        code  = function_info['code']
        if max_length and len(code)>max_length:
            code = truncate_code(code, max_length,strategy='trucate_end')
        parts.append(f"Code:\n{code}")
    
    return '\n'.join(parts)
    
def truncate_code(code, max_length, strategy:str = 'truncate_end')->str:
    """Truncate code to max length using specified strategy"""
    if len(code)<=max_length:
        return code
    
    if strategy == 'truncate_end':
        return code[:max_length] + '...'
    elif strategy == 'sig_doc':
        lines = code.split('\n')
        return '\n'.join(lines[:10]) + "\n..."
    elif strategy == 'mul_chunks':
        return code[:max_length] + '...'
    else:
        return code[:max_length] + '...'

def extract_metadata(function_info: dict) -> dict:
    """Extract clean metadata for vector store."""
    return {
        'function': function_info['name'],
        'file': function_info['file'],
        'line_start': function_info['line_start'],
        'line_end': function_info['line_end'],
        'is_async': function_info['type'] == 'async',
        'has_docstring': bool(function_info['docstring'])
    }

if __name__ == '__main__':
   
    directory = Path("data/raw/fastapi/fastapi")
    functions = parse_directory(directory_path=directory)
    
    print(f"Parsed {len(functions)} functions\n")

    print("=" * 50)
    print("Single function formatted:")
    print("=" * 50)
    print(format_function_chunk(functions[0], include_code=True, max_length=200))
    
    print("\n" + "=" * 50)
    print("Creating chunks:")
    print("=" * 50)
    chunks = chunk_by_function(functions[:5])
    
    print(f"\nFirst chunk:")
    print(chunks[0]['text'])
    print(f"\nMetadata: {chunks[0]['metadata']}")
    
    