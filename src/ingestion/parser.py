import ast
from loguru import logger
from pathlib import Path    
from typing import List, Dict, Optional


def parse_python_file(filepath:Path)->List[Dict]:
    """Parses a single file and extract all functions"""
    try:
        with open(filepath,'r',encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read {filepath} : {e}")
        return []
    
    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logger.error(f"Syntax error in {filepath} : {e}")
        return []
    
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                function_code = ast.get_source_segment(content, node)
            except:
                function_code = ""
            
            functions.append({
                'name' : node.name,
                "type" : 'async' if isinstance(node, ast.AsyncFunctionDef) else 'sync',
                "docstring" : ast.get_docstring(node) or '',
                "line_start" : node.lineno,
                "line_end": node.end_lineno,
                "code" :function_code or '',
                "file" : str(filepath),
            })
    return functions
        
print(len(parse_python_file(Path("data/raw/fastapi/fastapi/applications.py"))))

def parse_directory(directory_path: Path, exclude_patterns:Optional[List[str]]=None):
    """Parse all files in directory"""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.pyc', 'test_']
    
    directory = Path(directory_path)
    python_files = list(directory.rglob('*.py'))
    
    python_files = [
        f for f in python_files 
        if not any(pattern in str(f) for pattern in exclude_patterns)
    ]
    print(len(python_files))
    
    logger.info(f"Found {len(python_files)} Python files in {directory}")
    all_functions = []
    for filepath in python_files:
        functions = parse_python_file(filepath)
        all_functions.extend(functions)
    
    logger.info(f"Parsed {len(all_functions)} total functions")
    return all_functions

all_functions = parse_directory("data/raw/fastapi/fastapi")
def format_function_for_embedding(function_info: Dict) -> str:
    """Format function metadata into suitable text for embedding ."""
    parts = []
    
    async_prefix = "async" if function_info['type'] == 'async' else ""
    parts.append(f"{async_prefix}def {function_info['name']}()")
    
    if function_info['docstring']:
        parts.append(f"Description: {function_info['docstring']}")
        
    filename = Path(function_info['file']).name
    parts.append(f"File : {filename}")
    
    code = function_info['code']
    if(len(code) > 500):
        code = code[:500] + "..."
    parts.append(f"Code:\n{code}")
    
    return "\n".join(parts)
        
    
if __name__=="__main__":
    from pathlib import Path
    
    functions = parse_directory(Path("data/raw/fastapi/fastapi"))
    print(f"Found {len(functions)} functions")
    
    if functions:
        formatted = format_function_for_embedding(functions[10])
        print(formatted)

