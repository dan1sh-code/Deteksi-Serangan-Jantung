import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.eda import run_eda_polars

if __name__ == "__main__":
    run_eda_polars()
