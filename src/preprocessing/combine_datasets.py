import pandas as pd
import os

malware_path = '../../******.csv'
benign_path = '../../*******.csv'
output_path = 'output_features/******.csv'


if not os.path.exists(malware_path):
    raise FileNotFoundError(f"{malware_path} not found.")
if not os.path.exists(benign_path):
    raise FileNotFoundError(f"{benign_path} not found.")

df_malware = pd.read_csv(malware_path)
df_benign = pd.read_csv(benign_path)

if list(df_malware.columns) != list(df_benign.columns):
    raise ValueError("Malware and benign columns are not equal. Please check.")

df_combined = pd.concat([df_malware, df_benign], ignore_index=True)

df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


df_combined.to_csv(output_path, index=False)
print(f"Combined file '{output_path}' is saved.")
