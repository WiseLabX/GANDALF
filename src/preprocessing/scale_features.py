import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # 🆕 eklendi
import numpy as np

malware_df = pd.read_csv('../../*******.csv')
benign_df = pd.read_csv('../../******.csv')


df_combined = pd.concat([malware_df, benign_df], ignore_index=True)

binary_cols = [col for col in df_combined.columns
               if set(df_combined[col].dropna().unique()).issubset({0,1})]

continuous_cols = [col for col in df_combined.select_dtypes(include=['int64','float64']).columns
                   if col not in binary_cols]

print(f" Binary values will not be scaled: {len(binary_cols)}")
print(f" numerical values will be scaled: {len(continuous_cols)}")

print("\n Before Scaling Min-Max ---")
print(df_combined[continuous_cols].agg(['min','max']))

scaler = MinMaxScaler(feature_range=(-1,1))
df_combined[continuous_cols] = scaler.fit_transform(np.log1p(df_combined[continuous_cols]))

joblib.dump(scaler, 'output_features/scaler_continuous_columns.pkl')
print("\n Scaler saved: output_features/scaler_continuous_columns.pkl")

print("\n After Scaling Min-Max ---")
print(df_combined[continuous_cols].agg(['min','max']))

print("\n Scaled columns:")
print(continuous_cols)

df_combined.to_csv('output_features/*****.csv', index=False)

malware_df_scaled = df_combined[df_combined['Malware']==1]
benign_df_scaled = df_combined[df_combined['Malware']==0]

malware_df_scaled.to_csv('output_features/******.csv', index=False)
benign_df_scaled.to_csv('output_features/******.csv', index=False)


