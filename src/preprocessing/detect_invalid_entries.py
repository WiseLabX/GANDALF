import pandas as pd

def analyze_dataframe_for_tensor_issues(df, name="DataFrame"):

    print("\n1. Nested Ccells:")
    nested_cols = []
    for col in df.columns:
        nested_count = df[col].apply(lambda x: isinstance(x, (pd.DataFrame, pd.Series, list, dict))).sum()
        if nested_count > 0:
            nested_cols.append((col, nested_count))
            print(f"  {col} → {nested_count} .")
    if not nested_cols:
        print("No nested.")

    print("\n🔍 2. non numeric:")
    non_numeric_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
            print(f" {col} → dtype: {df[col].dtype}")
    if not non_numeric_cols:
        print("All numeric.")

    print("\n(NaN) values:")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            print(f" {col} → {count}  NaN values.")
    else:
        print("no (NaN) values.")

    print("\nfinihed.\n")


features = pd.read_csv("../../*****.csv")
labels = features['Malware']

analyze_dataframe_for_tensor_issues(features, name="Features")
analyze_dataframe_for_tensor_issues(labels.to_frame(), name="Labels")