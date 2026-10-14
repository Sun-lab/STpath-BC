"""Report saved batch outputs without a notebook frontend."""
from pathlib import Path

def display(*values):
    for value in values:
        print(value.to_string(index=False) if hasattr(value, 'to_string') else value)

def Image(filename=None, **kwargs):
    return Path(filename) if filename else 'Figure saved'

DisplayImage = Image
FileLink = Path
Markdown = str
HTML = str

def clear_output(**kwargs):
    pass
