import nbformat
import sys
import os
import re

def make_student_version(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        # Clean code cells
        if cell.cell_type == 'code' and '### your code here' in cell.source:
            cell.source = '### your code here'
        # Clean markdown title
        if cell.cell_type == 'markdown':
            lines = cell.source.splitlines()
            if lines and re.match(r'^# .*- solution\s*$', lines[0], re.IGNORECASE):
                # Remove ' - solution' (case-insensitive, with or without leading/trailing spaces)
                lines[0] = re.sub(r'\s*-\s*solution\s*$', '', lines[0], flags=re.IGNORECASE)
                cell.source = '\n'.join(lines)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}-student{ext}"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Student version saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_student_notebook.py <notebook.ipynb>")
        sys.exit(1)
    make_student_version(sys.argv[1])