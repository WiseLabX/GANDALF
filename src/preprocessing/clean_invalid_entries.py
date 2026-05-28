import pandas as pd

df = pd.read_csv("../../******.csv")

numeric_columns_to_fill_zero = [
    'NrIntReceivers',
    'NrIntActivities',
    'NrServices',
    'Activities',
    'NrIntServices',
    'TotalIntentFilters',
    'NrIntReceiversActions'
]

df[numeric_columns_to_fill_zero] = df[numeric_columns_to_fill_zero].fillna(0)

if 'MalFamily' in df.columns:
    df['MalFamily'] = df['MalFamily'].fillna("Unknown")

print(f" number of columns: {df.shape[1]}, number of records: {df.shape[0]}")
print(f"🔍 NaN values: {df.isnull().sum().sum()}")


output_path = 'output_features/******.csv'
df.to_csv(output_path, index=False)

