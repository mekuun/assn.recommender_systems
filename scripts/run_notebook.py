"""Execute all cells in a fresh kernel and save the executed notebook."""
import os
from pathlib import Path
# Set BLAS limits before the notebook kernel is launched.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / 'experiments.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}},
                        on_cell_start=lambda cell, cell_index: print(f'Cell {cell_index}', flush=True))
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Executed:', path)
