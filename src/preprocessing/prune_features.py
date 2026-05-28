import pandas as pd

malware_file = '../../****.csv'
benign_file = '../../*****.csv'


malware_df = pd.read_csv(malware_file)
benign_df = pd.read_csv(benign_file)

common_cols = malware_df.columns.intersection(benign_df.columns)

malware_df = malware_df[common_cols]
benign_df = benign_df[common_cols]

def find_constant_columns(df):
    return [col for col in df.columns if df[col].nunique() == 1]

def find_high_cardinality_columns(df, threshold=0.9):
    high_card_cols = []
    n = len(df)
    for col in df.columns:
        unique_ratio = df[col].nunique() / n
        if unique_ratio > threshold:
            high_card_cols.append(col)
    return high_card_cols

def find_date_time_like_columns(cols):
    date_keywords = ['date', 'time', 'timestamp', 'mod', 'clock', 'timedate', 'updated']
    return [col for col in cols if any(k in col.lower() for k in date_keywords)]


def find_leak_columns(cols):
    leak_keywords = ['ratio', 'scanner', 'detection', 'leak']
    return [col for col in cols if any(k in col.lower() for k in leak_keywords)]

constant_malware = set(find_constant_columns(malware_df))
constant_benign = set(find_constant_columns(benign_df))

constant_both = constant_malware.intersection(constant_benign)

constant_only_malware = constant_malware - constant_both
constant_only_benign = constant_benign - constant_both

full_df = pd.concat([malware_df, benign_df], ignore_index=True)
high_card_cols = find_high_cardinality_columns(full_df)
date_time_cols = find_date_time_like_columns(full_df.columns)
leak_cols = find_leak_columns(full_df.columns)


drop_cols = set(constant_both) | set(high_card_cols) | set(leak_cols)

label_cols = ['Package', 'Malware', 'MalFamily']
label_cols = [col for col in label_cols if col in full_df.columns]
drop_cols = drop_cols.difference(label_cols)


print("--- CONSTANT COLUMNS ---")
for c in constant_both:
    print(f"- {c}")

print("\n--- CONSTANT COLUMNS ---")
for c in sorted(constant_only_malware | constant_only_benign):
    print(f"- {c}")

print("\n--- DROP COLUMNS ---")
for c in drop_cols:
    print(f"- {c}")

pruned_full_df = full_df.drop(columns=drop_cols)

malware_pruned_df = pruned_full_df[pruned_full_df['Malware'] == 1].copy()
benign_pruned_df = pruned_full_df[pruned_full_df['Malware'] == 0].copy()

malware_pruned_df.to_csv('output_features/*****.csv', index=False)
benign_pruned_df.to_csv('output_features/******.csv', index=False)


