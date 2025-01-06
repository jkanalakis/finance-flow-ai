
import pandas as pd

def load_dataset(file_path: str):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
