import pandas as pd
import numpy as np
import os

# Function to check if the file is empty
def is_file_empty(file_path):
    """Check if file is empty by reading first character in it"""
    # Using 'with' automatically closes the file after the block execution
    with open(file_path, 'r') as f:
        first_char = f.read(1)
        if not first_char:
            return True
    return False

def append_column(path,column_name,data):
    if os.path.exists(path) and not is_file_empty(path):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print("CSV file is empty. Creating a new DataFrame.")
            df = pd.DataFrame()
    else:
        print(f"File '{path}' does not exist or is empty. Creating a new file.")
        df = pd.DataFrame()

    if column_name in df.columns:
        print(f"Column '{column_name}' exists. It will be replaced.")
    else:
        print(f"Column '{column_name}' does not exist. It will be added.")

    df[column_name] = data
    df = df.sort_index(axis=1, ascending=False)
    df.to_csv(path, index=False)