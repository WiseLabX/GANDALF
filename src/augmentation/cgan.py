
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn.utils import spectral_norm


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=5):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # x: batch_size x in_features
        batch_size = x.size(0)
        M = x.matmul(self.T.view(self.in_features, -1))  # batch_size x (out_features*kernel_dims)
        M = M.view(batch_size, self.out_features, self.kernel_dims)

        # Pairwise L1 distance
        out = []
        for i in range(batch_size):
            diff = torch.abs(M[i] - M)  # batch_size x out_features x kernel_dims
            exp_neg = torch.exp(-diff.sum(2))  # batch_size x out_features
            out.append(exp_neg.sum(0) - 1)  # kendi kendine farkı çıkar

        out = torch.stack(out)
        # x ile concat et
        return torch.cat([x, out], dim=1)

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DiscriminatorNet, self).__init__()
        embed_dim = 32
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.pre_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim + embed_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )

        # Minibatch discrimination
        self.minibatch_layer = MinibatchDiscrimination(in_features=128, out_features=16, kernel_dims=3)

        self.post_net = nn.Sequential(
            spectral_norm(nn.Linear(128 + 16, 64)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], dim=1)

        x = self.pre_net(x)
        x = self.minibatch_layer(x)
        x = self.post_net(x)
        return x


class GeneratorNet(nn.Module):
    def __init__(self, z_dim, num_classes, output_dim):
        super(GeneratorNet, self).__init__()
        embed_dim = 32
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, output_dim)
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)  # label → embedding
        x = torch.cat([z, c], dim=1)  # birleştir
        x = self.net(x)
        return torch.tanh(x)


def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates, labels)

    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

#Training Loop for conditional_gan
def train_conditional_gan(train_loader, generator, discriminator, optimizer_g, optimizer_d, z_dim, device, label_probs, epochs=epochs, lambda_gp=10, n_critic=5 ):
    d_loss_arr = []
    g_loss_arr = []

    print("Label probs:", label_probs)

    for epoch in range(epochs):
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (real_data_mal, labels) in enumerate(train_loader):
            real_data = real_data_mal.to(device)
            labels = labels.to(device).long()
            batch_size = real_data.size(0)

            #Discriminator (Critic) Training
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim).to(device)

                fake_labels = torch.multinomial(label_probs, batch_size, replacement=True).to(device)
                fake_data = generator(noise, fake_labels).detach()

                real_validity = discriminator(real_data, labels)
                fake_validity = discriminator(fake_data, fake_labels)

                gp = compute_gradient_penalty(discriminator, real_data, fake_data, fake_labels, device)
                d_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

            #Generator Training
            noise = torch.randn(batch_size, z_dim).to(device)
            gen_labels = torch.multinomial(label_probs, batch_size, replacement=True).to(device)
            gen_data = generator(noise, gen_labels)
            g_loss = -discriminator(gen_data, gen_labels).mean()

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

        # Epoch average
        avg_d_loss = d_epoch_loss / num_batches
        avg_g_loss = g_epoch_loss / num_batches
        d_loss_arr.append(avg_d_loss)
        g_loss_arr.append(avg_g_loss)

        print(f"[{epoch + 1}/{epochs}] Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")

    # Mode collapse checking
    results = check_mode_collapse(
        generator,
        latent_dim=z_dim,
        n_samples=1000,
        device=device
    )
    print(results)

    plot_loss_curves(d_loss_arr, g_loss_arr)  # Batch based
    plot_loss_curves_new(d_loss_arr, g_loss_arr)  # Epoch based
    plot_loss_curves_v2(d_loss_arr, g_loss_arr)  # Epoch based
    plot_loss_curves_smoothed(d_loss_arr, g_loss_arr, smooth_window=5)  # Smoothed epoch loss
    return d_loss_arr, g_loss_arr



def generate_synthetic_malware ():
    global generator
    global train_set
    global train_period_benign, train_period_malware

    # DATASET
    df = pd.read_csv(STATIC_COMBINED_PATH)
    df['Malware'] = df['Malware'].astype(int)
    df['HighestModDate'] = pd.to_datetime(df['HighestModDate'], errors='coerce')

    malware_df = df[df['Malware'] == 1]
    benign_df = df[df['Malware'] == 0]

    # Malware families TO be included in the test set
    # to evaluate the model against previously unseen malware families
    forced_unseen_families = [
        "Android MarsDaemon", "AppAd", "Code4hk/xRAT", "Fakengry",
        "Hiddapp", "HiddenApp", "HighsterSpy", "Minimob",
        "Mobtes", "Pandaad", "PornVideo", "Resharer",
        "Secneo", "Shedun", "TiSpy", "Trackplus", "Zdtad"
    ]

    # TRAIN SET (2008–2016)
    train_period_malware = malware_df[
        (malware_df['HighestModDate'].dt.year >= 2008) &
        (malware_df['HighestModDate'].dt.year <= 2016)
        ]

    train_period_benign = benign_df[
        (benign_df['HighestModDate'].dt.year >= 2008) &
        (benign_df['HighestModDate'].dt.year <= 2016)
        ]

    train_benign = train_period_benign.sample(n=8000, random_state=42)
    train_malware = train_period_malware.sample(n=220, random_state=42)

    train_set = pd.concat([train_benign, train_malware]) \
        .sample(frac=1, random_state=42) \
        .reset_index(drop=True)

    print(train_set.shape)


    # VALIDATION SET
    val_benign = train_period_benign.drop(train_benign.index)
    val_malware = train_period_malware.drop(train_malware.index)

    val_benign = val_benign.sample(n=5000, random_state=42)
    val_malware = val_malware.sample(n=5000, random_state=42)

    val_set = pd.concat([val_benign, val_malware]).sample(frac=1, random_state=42)

    print("TRAIN → benign:", len(train_benign))
    print("TRAIN → malware:", len(train_malware))
    print("TRAIN total:", len(train_set))

    # TEST SET (2017–2020)
    test_period_malware = malware_df[
        (malware_df['HighestModDate'].dt.year >= 2017) &
        (malware_df['HighestModDate'].dt.year <= 2020)
        ]

    test_period_benign = benign_df[
        (benign_df['HighestModDate'].dt.year >= 2017) &
        (benign_df['HighestModDate'].dt.year <= 2020)
        ]

    forced_test_malware = test_period_malware[
        test_period_malware['MalFamily'].isin(forced_unseen_families)
    ]

    remaining_pool = test_period_malware[
        ~test_period_malware.index.isin(forced_test_malware.index)
    ]

    remaining_needed = 440 - len(forced_test_malware)

    random_test_malware = remaining_pool.sample(
        n=remaining_needed,
        random_state=42
    )

    test_malware = pd.concat([forced_test_malware, random_test_malware])

    test_benign = test_period_benign.sample(n=3995, random_state=42)

    test_set = pd.concat([test_benign, test_malware]) \
        .sample(frac=1, random_state=42) \
        .reset_index(drop=True)

    print("TEST → benign:", len(test_benign))
    print("TEST → malware:", len(test_malware))
    print("Forced unseen in test:", len(forced_test_malware))
    print("TEST total:", len(test_set))

    train_families = set(train_set[train_set['Malware'] == 1]['MalFamily'])
    test_families = set(test_set[test_set['Malware'] == 1]['MalFamily'])

    unseen = test_families - train_families
    print("Number of real unseen family:", len(unseen))

    train_family_counts = (
        train_set[train_set['Malware'] == 1]['MalFamily']
        .value_counts()
        .reset_index()
    )

    train_family_counts.columns = ['MalwareFamily', 'Count']

    test_family_counts = (
        test_set[test_set['Malware'] == 1]['MalFamily']
        .value_counts()
        .reset_index()
    )

    test_family_counts.columns = ['MalwareFamily', 'Count']

    output_path = "malwarefamily_dataset_statistics.xlsx"

    with pd.ExcelWriter(output_path) as writer:
        train_family_counts.to_excel(
            writer,
            sheet_name='Train_Malware_Family',
            index=False
        )

        test_family_counts.to_excel(
            writer,
            sheet_name='Test_Malware_Family',
            index=False
        )

    print(f" Statistics have been successfully written to the Excel file: {output_path}")


    label_columns = ['Malware', 'MalFamily', 'Package', 'EarliestModDate', 'HighestModDate']
    test_malfamily= test_set['MalFamily']

    y_train_global = train_set['Malware']
    X_train_global = train_set.drop(columns=label_columns)
    y_test_global = test_set['Malware']
    X_test_global = test_set.drop(columns=label_columns)

    label_columns = ['Malware', 'MalFamily', 'Package', 'EarliestModDate', 'HighestModDate']

    X_train = train_set.drop(columns=label_columns)
    y_train = train_set['Malware']

    X_val = val_set.drop(columns=label_columns)
    y_val = val_set['Malware']
    val_malfamily = val_set['MalFamily']

    # VALIDATION model
    model_val = RandomForestClassifier(n_estimators=100, random_state=42)
    model_val.fit(X_train, y_train)
    y_val_pred = model_val.predict(X_val)

    recall_dict = {}

    df_val = pd.DataFrame({
        'y_true': y_val.values,
        'y_pred': y_val_pred,
        'MalFamily': val_malfamily.values
    })

    malware_df = df_val[df_val['y_true'] == 1]

    for family, group in malware_df.groupby('MalFamily'):
        tp = (group['y_pred'] == 1).sum()
        fn = (group['y_pred'] == 0).sum()
        recall = tp / (tp + fn + 1e-6)
        recall_dict[family] = recall

    print("VAL SMSreg recall:", recall_dict.get("SMSreg"))

    # WEIGHT calculation
    weights = {}
    for k, v in recall_dict.items():
        weights[k] = max(0.05, 1 - v)

    total = sum(weights.values())

    print("Difficulty weights ready")

    print("Weights:", weights)

    label_probs = build_label_probs(weights, family_to_idx, num_classes, device)

    # train the CGAN
    d_losses, g_losses = train_conditional_gan(
        train_loader=train_loader,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        z_dim=z_dim,
        device=device,
        label_probs=label_probs,
        epochs=epochs,
        lambda_gp=lambdagp,
        n_critic=n_critic  # One generator update for every 5 discriminator updates
    )

    # WEIGHTED generation
    n_i_dict = {}

    for family, weight in weights.items():
        n_i = int(weight * synthetic_malware_size)
        n_i = max(n_i, 10)
        n_i_dict[family] = n_i

    total_assigned = sum(n_i_dict.values())

    if total_assigned > synthetic_malware_size:
        scale = synthetic_malware_size / total_assigned
        for k in n_i_dict:
            n_i_dict[k] = int(n_i_dict[k] * scale)

    remaining = synthetic_malware_size - sum(n_i_dict.values())

    sorted_families = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    i = 0
    while remaining > 0:
        family = sorted_families[i % len(sorted_families)][0]
        n_i_dict[family] += 1
        remaining -= 1
        i += 1

    all_samples = []

    all_labels = []

    for family, n_i in n_i_dict.items():

        if n_i == 0:
            continue

        if family not in family_to_idx:
            print(f"Skip: {family} (not present)")
            continue  # unseen family crash

        label_idx = family_to_idx[family]

        z = torch.randn(n_i, z_dim).to(device)
        labels = torch.full((n_i,), label_idx).to(device).long()

        fake = generator(z, labels).detach().cpu()
        all_samples.append(fake)
        all_labels.extend([label_idx] * n_i)

    synthetic_malware = torch.cat(all_samples, dim=0)

    from collections import Counter
    print("Fake label distribution:", Counter(all_labels))

    fake_data_binary=synthetic_malware

    # Result
    print(fake_data_binary)


def get_validation_performance(X_train, y_train, X_val, y_val, val_malfamily):
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return y_pred

def build_label_probs(weights, family_to_idx, num_classes, device):
    probs = torch.zeros(num_classes)

    for family, idx in family_to_idx.items():
        probs[idx] = weights.get(family, 0.0)

    probs = probs / probs.sum()
    return probs.to(device)

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

