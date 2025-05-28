import os
import json

## Run this in the container!

def update_kernel_metadata(notebook_path, kernel_name):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    notebook['metadata']['kernelspec'] = {
        "name": kernel_name,
        "display_name": f"Python 3 ({kernel_name})",
        "language": "python"
    }

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

def update_all_notebooks(directory, kernel_name):
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return

    found_notebooks = False
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                found_notebooks = True
                notebook_path = os.path.join(root, file)
                print(f"Updating notebook: {notebook_path}")  # Debug print
                update_kernel_metadata(notebook_path, kernel_name)

    if not found_notebooks:
        print("No .ipynb files found in the directory.")

# Update all notebooks in the specified directory
update_all_notebooks('/mnt/jbook', 'jbook')