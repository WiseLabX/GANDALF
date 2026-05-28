
def plot_loss_curves(d_loss_arr, g_loss_arr):
    plt.plot(d_loss_arr, label='Discriminator Loss')
    plt.plot(g_loss_arr, label='Generator Loss')
    plt.legend()
    plt.title('WGAN Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Moving average
def moving_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_loss_curves_new(d_loss, g_loss):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    d_smooth = moving_average(d_loss)
    g_smooth = moving_average(g_loss)
    # Discriminator Loss
    axs[0].plot(d_loss, marker='o', color='lightcoral', label='Raw D Loss')
    axs[0].plot(range(len(d_smooth)), d_smooth, color='red', linewidth=2, label='Smoothed D Loss')
    axs[0].set_title('Discriminator Loss (with Smoothing)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()

    # Generator Loss
    axs[1].plot(g_loss, marker='o', color='skyblue', label='Raw G Loss')
    axs[1].plot(range(len(g_smooth)), g_smooth, color='blue', linewidth=2, label='Smoothed G Loss')
    axs[1].set_title('Generator Loss (with Smoothing)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_loss_curves_v2(d_loss_arr, g_loss_arr):
    # Discriminator Loss graph
    plt.figure(figsize=(10, 4))
    plt.plot(d_loss_arr, label='Discriminator Loss', color='red')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Generator Loss graph
    plt.figure(figsize=(10, 4))
    plt.plot(g_loss_arr, label='Generator Loss', color='blue')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curves_smoothed(d_loss_arr, g_loss_arr, smooth_window=30):

    plt.figure(figsize=(12, 6))

    steps = np.arange(len(d_loss_arr))

    def smooth_curve(arr, window):
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    smooth_d = smooth_curve(d_loss_arr, smooth_window)
    smooth_g = smooth_curve(g_loss_arr, smooth_window)

    plt.plot(smooth_d, label='Discriminator Loss (smoothed)', color='red')
    plt.plot(smooth_g, label='Generator Loss (smoothed)', color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Wasserstein GAN Losses")
    plt.grid(True)
    plt.legend()
    epoch_steps = len(d_loss_arr) // epochs
    for e in range(1, epochs):
        plt.axvline(x=e * epoch_steps, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_real_vs_synthetic(real_loader, fake_data_tensor):

    real_samples = []
    for batch_data, _ in real_loader:
        real_samples.append(batch_data)
        if len(real_samples) * batch_data.size(0) > 5000:
            break
    real_data = torch.cat(real_samples, dim=0)[:5000].cpu().numpy()

    fake_data = fake_data_tensor.detach().cpu().numpy()

    all_data = np.concatenate([real_data, fake_data], axis=0)
    labels = np.array([0]*len(real_data) + [1]*len(fake_data))  # 0: Gerçek, 1: Sahte

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(all_data)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    data_tsne = tsne.fit_transform(all_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(data_pca[labels==0][:, 0], data_pca[labels==0][:, 1], label='Real Malware', alpha=0.6)
    axes[0].scatter(data_pca[labels==1][:, 0], data_pca[labels==1][:, 1], label='Synthetic Malware', alpha=0.6)
    axes[0].set_title('PCA Visualization')
    axes[0].legend()

    # t-SNE
    axes[1].scatter(data_tsne[labels==0][:, 0], data_tsne[labels==0][:, 1], label='Real Malware', alpha=0.6)
    axes[1].scatter(data_tsne[labels==1][:, 0], data_tsne[labels==1][:, 1], label='Synthetic Malware', alpha=0.6)
    axes[1].set_title('t-SNE Visualization')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def visualize_malware_benign_synthetic(train_loader, fake_data_tensor):
    dfbenign = pd.read_csv(STATIC_BENIGN_PATH, dtype=str)


    label_columns = ['Malware', 'MalFamily', 'Package',
                     'EarliestModDate', 'HighestModDate']
    X_benign = dfbenign.drop(columns=label_columns, errors='ignore')

    # Remove non-numeric columns
    X_benign = X_benign.apply(pd.to_numeric, errors='coerce')

    # Remove rows containing NaN values
    X_benign = X_benign.dropna()

    X_All = X_benign.to_numpy(dtype=np.float32)

    print("Benign shape:", X_All.shape)

    # Dummy label
    y_All = np.zeros(len(X_All), dtype=np.float32)

    datasetbenign = MalwareDataset(X_All, y_All)
    train_loader_benign = DataLoader(
        datasetbenign,
        batch_size=batch_size,
        shuffle=True
    )

    real_samples = []
    total_real_samples = 0
    for batch_data, labels in train_loader:
        real_samples.append(batch_data)

        total_real_samples += batch_data.size(0)
        if total_real_samples >= 5000:
            break
    real_data = torch.cat(real_samples, dim=0)[:5000].cpu().numpy()

    for batch_data, labels in train_loader:
        real_samples.append(batch_data)
        total_real_samples += batch_data.size(0)
        if total_real_samples >= 5000:
            break
    real_data = torch.cat(real_samples, dim=0)[:5000].cpu().numpy()

    fake_data = fake_data_tensor.detach().cpu().numpy()
    benign_samples = []
    total_benign_samples = 0
    for batch_data, labels in train_loader_benign:
        benign_samples.append(batch_data)
        total_benign_samples += batch_data.size(0)
        if total_benign_samples >= 5000:
            break
    benign_samples = torch.cat(benign_samples, dim=0)[:5000].cpu().numpy()

    benign_samples = np.array(benign_samples)
    print(f"Real: {real_data.shape}, Fake: {fake_data.shape}, Benign: {benign_samples.shape}")

    if benign_samples.ndim == 3 and benign_samples.shape[0] >= 1:
        benign_samples = benign_samples.squeeze(0)

    print("Real:", real_data.shape)
    print("Fake:", fake_data.shape)
    print("Benign:", benign_samples.shape)

    all_data = np.concatenate([real_data, fake_data, benign_samples], axis=0)
    labels = np.array(
        [0] * len(real_data) +     # 0: Gerçek Malware
        [1] * len(fake_data) +     # 1: Sahte Malware (GAN)
        [2] * len(benign_samples)  # 2: Gerçek Benign
    )

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(all_data)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    data_tsne = tsne.fit_transform(all_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    renkler = ['red', 'blue', 'green']
    etiketler = ['Real Malware', 'Synthetic Malware (GAN)', 'Real Benign']

    # PCA
    for i in range(3):
        axes[0].scatter(data_pca[labels==i][:, 0], data_pca[labels==i][:, 1], label=etiketler[i], alpha=0.6, color=renkler[i])
    axes[0].set_title('PCA Visualization')
    axes[0].legend()

    # t-SNE
    for i in range(3):
        axes[1].scatter(data_tsne[labels==i][:, 0], data_tsne[labels==i][:, 1], label=etiketler[i], alpha=0.6, color=renkler[i])
    axes[1].set_title('t-SNE Visualization')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


    real_centroid = np.mean(real_data, axis=0)
    fake_centroid = np.mean(fake_data, axis=0)
    benign_centroid = np.mean(benign_samples, axis=0)

    # Calculate Distance
    dist_fake_real = np.linalg.norm(fake_centroid - real_centroid)
    dist_fake_benign = np.linalg.norm(fake_centroid - benign_centroid)

    print("\n Distribution Distance Analysis")
    print(f"Distance(Synthetic, Real Malware): {dist_fake_real:.4f}")
    print(f"Distance(Synthetic, Benign): {dist_fake_benign:.4f}")

    if dist_fake_real < dist_fake_benign:
        print("Synthetic malware distribution is closer to real malware distribution.")
    else:
        print("Synthetic malware distribution overlaps with benign distribution.")

def check_synthetic_malware_similarity(synthetic_malware, k=4):
    sahte_np = synthetic_malware.cpu().numpy()

    # 1. Pairwise matrix (Euclidean)
    dist_matrix = pairwise_distances(sahte_np, metric='euclidean')

    # 2. Pairwise Distance Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap='viridis')
    plt.title("Pairwise Distance Heatmap")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()

    # 3. K-distance
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(sahte_np)
    distances, _ = nbrs.kneighbors(sahte_np)
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.title(f"{k}-Distance Graph (for DBSCAN)")
    plt.xlabel("Data Points sorted by distance")
    plt.ylabel(f"{k}-NN Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    q1 = np.percentile(k_distances, 25)
    q3 = np.percentile(k_distances, 75)
    iqr = q3 - q1
    low_eps = max(q1 - 1.5 * iqr, 1e-4)  # negatif olmasın
    high_eps = q3 + 1.5 * iqr
    print(f"\n🔧 Suggested eps range based on IQR: [{low_eps:.3f}, {high_eps:.3f}]\n")

    eps_values = np.linspace(low_eps, high_eps, num=10)

    print(" DBSCAN Diversity:\n")
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5, metric='precomputed')
        clusters = dbscan.fit_predict(dist_matrix)

        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        cluster_sizes = collections.Counter(clusters)
        noise_count = cluster_sizes.get(-1, 0)

        print(f"eps = {eps:.3f}")
        print(f"  ➤ Number of clusters found: {num_clusters}")
        print(f"  ➤ Noise points (label -1): {noise_count}")
        print(f"  ➤ Cluster sizes (label: count): {dict(cluster_sizes)}\n")

        # 6. Visualise with PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(sahte_np)

        plt.figure(figsize=(8, 6))
        unique_labels = set(clusters)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for label in unique_labels:
            indices = clusters == label
            if label == -1:
                plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
                            c='lightgrey', marker='x', label='Noise')
            else:
                plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
                            color=colors(label), label=f'Cluster {label}')

        plt.title(f"DBSCAN Clustering (eps={eps:.3f})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            ncol=2
        )
        plt.tight_layout()
        plt.show()

def check_mode_collapse(generator, latent_dim, n_samples=1000, device='cpu', visualize=True):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        fake_samples = generator(z).cpu().numpy()

    # Basic diversity measurements
    std_per_feature = np.std(fake_samples, axis=0)
    overall_std = np.mean(std_per_feature)
    pairwise_dists = np.linalg.norm(fake_samples - fake_samples.mean(axis=0), axis=1)
    diversity_score = np.mean(pairwise_dists)

    # Clustering analysis using DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(fake_samples)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    if overall_std < 0.01 or diversity_score < 0.01:
        diversity_report = "Very low diversity: High risk of mode collapse."
    elif n_clusters < 2:
        diversity_report = "Low diversity: Generated samples are highly similar to each other."
    else:
        diversity_report = "Diversity is sufficient: No apparent mode collapse."

    report = {
        "feature_std_mean": overall_std,
        "diversity_score": diversity_score,
        "n_samples": n_samples,
        "n_clusters_dbscan": n_clusters,
        "n_noise_dbscan": n_noise,
        "diversity_report": diversity_report
    }

    print("Mode Collapse / Diversity Analysis:")
    print(f"• Average std per feature: {overall_std:.6f}")
    print(f"• Average sample distance (diversity score): {diversity_score:.6f}")
    print(f"• Number of DBSCAN clusters: {n_clusters}")
    print(f"• Number of DBSCAN noise points: {n_noise}")
    print(f"• Report: {diversity_report}")

    # Visualization
    if visualize:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(fake_samples[:1000])

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.6)
        plt.title("t-SNE Visualization - Generated Fake Samples")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    return report


