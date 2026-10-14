"""Set notebook code and data paths independently of the working directory."""
from pathlib import Path
import os
import sys

def setup_notebook():
    code_root = Path(__file__).resolve().parents[1]
    project_root = Path(os.environ.get('XENIUM_ROOT', code_root)).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f'XENIUM_ROOT does not exist: {project_root}')
    os.environ.setdefault('XENIUM_ROOT', str(project_root))
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    os.chdir(project_root)
    return code_root, project_root
