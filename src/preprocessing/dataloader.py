

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

class MalwareDataset(Dataset):
    def __init__(self, features, labels):
        if hasattr(features, "values"):
            features = features.values
        if hasattr(labels, "values"):
            labels = labels.values

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def csv_fileload(featureselection=False):
    global train_loader
    df = pd.read_csv(STATIC_MALWARE_PATH)

    label_column = 'Malware'

    # columns to be removed
    label_columns = ['Malware', 'MalFamily', 'Package','EarliestModDate','HighestModDate']

    # ---------------------------
    # Select only malware samples from the 2008–2016 period for the training set
    df['HighestModDate'] = pd.to_datetime(df['HighestModDate'], errors='coerce')

    df = df[
        (df['Malware'] == 1) &
        (df['HighestModDate'].dt.year >= 2008) &
        (df['HighestModDate'].dt.year <= 2016)
        ]

    print(f"Trainset malware number: {len(df)}")

    selected_malware = df.sample(
        n=30000,
        random_state=42,
        replace=False
    )

    y = selected_malware[label_column]  # Label
    X = selected_malware.drop(columns=label_columns)

    X_test = X.copy()
    y_test = y.copy()

    if featureselection:
        correlation_threshold = 0.9
        corr_matrix = X_test.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
        print(f"Features to be removed (high correlation): {to_drop}")

        X_reduced = X_test.drop(columns=to_drop)
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        selector = RFE(model, n_features_to_select=75)
        selector = selector.fit(X_test, y)

        selected_features = X_test.columns[selector.support_]
        print(f"Selected Features: {selected_features}")
        X_rfe = X[:, selector.support_]
        dataset = MalwareDataset(X_rfe, y)
    else:
        dataset = MalwareDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
