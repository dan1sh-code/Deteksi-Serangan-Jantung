import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.train_model import train_and_evaluate, preprocess_data_polars_combined

if __name__ == "__main__":
    train_and_evaluate()
