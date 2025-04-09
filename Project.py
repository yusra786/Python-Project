import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\yusra\Downloads\Pyhton project\heart_attack_predictions.csv')
# Step 1: Initial Data Exploration
print("Initial Data Exploration")
print("First 5 rows of the dataset:")
print(df.head())
print("\nSummary of the dataset:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())


