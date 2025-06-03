import nbformat
import sys
import os
import re

def make_student_version(input_path):
    """
    Generate a student version of a Jupyter notebook by removing solution code and solution markers.

    This function reads a Jupyter notebook (.ipynb) file, removes code in cells marked with
    '### your code here' (leaving only the marker), and cleans up markdown titles that indicate
    a solution (removing ' - solution' from the first line of such cells). The processed notebook
    is saved with '-student' appended to the original filename.

    Args:
        input_path (str): Path to the input Jupyter notebook file.

    Returns:
        None

    Example:
        >>> make_student_version("assignment1.ipynb")
        Student version saved as: assignment1-student.ipynb

    Notes:
        - The function expects the input file to be a valid Jupyter notebook (version 4).
        - The output file will be saved in the same directory as the input file.
    """
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