#!/usr/bin/env python3
"""Script to update project dependencies based on imports."""
import ast
import os
from pathlib import Path
from typing import Set

import toml

def get_imports_from_file(file_path: Path) -> Set[str]:
    """Extract import statements from a Python file."""
    imports: Set[str] = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return imports

def get_all_imports(src_dir: Path) -> Set[str]:
    """Get all imports from Python files in the source directory."""
    all_imports: Set[str] = set()
    
    for file_path in src_dir.rglob('*.py'):
        all_imports.update(get_imports_from_file(file_path))
    
    return all_imports

def update_pyproject_toml(imports: Set[str]) -> None:
    """Update pyproject.toml with new dependencies."""
    pyproject_path = Path('pyproject.toml')
    
    if not pyproject_path.exists():
        print("pyproject.toml not found")
        return
    
    # Read current pyproject.toml
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        pyproject = toml.load(f)
    
    # Get current dependencies
    current_deps = set(dep.split('>=')[0] for dep in pyproject['project']['dependencies'])
    
    # Add new dependencies
    new_deps = imports - current_deps
    if new_deps:
        print(f"Adding new dependencies: {new_deps}")
        for dep in new_deps:
            pyproject['project']['dependencies'].append(f"{dep}>=0.0.0")
    
    # Write updated pyproject.toml
    with open(pyproject_path, 'w', encoding='utf-8') as f:
        toml.dump(pyproject, f, indent=4)

def main() -> None:
    """Main function to update dependencies."""
    src_dir = Path('src')
    if not src_dir.exists():
        print("src directory not found")
        return
    
    imports = get_all_imports(src_dir)
    update_pyproject_toml(imports)

if __name__ == '__main__':
    main() 