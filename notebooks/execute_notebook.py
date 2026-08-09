"""
Script to execute notebooks/calibration_benchmark_tutorial.ipynb and save outputs inline.
"""

from pathlib import Path
from nbclient import NotebookClient
import nbformat


def run():
    nb_path = Path(__file__).parent / "calibration_benchmark_tutorial.ipynb"
    print(f"Reading {nb_path}...")
    nb = nbformat.read(str(nb_path), as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print("Executing notebook cells...")
    client.execute()

    print(f"Saving executed notebook back to {nb_path}...")
    nbformat.write(nb, str(nb_path))
    print("✅ Notebook execution completed successfully!")


if __name__ == "__main__":
    run()
