



from torch import nn
from torch.nn.utils import spectral_norm
import torch
import numpy as np

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=5):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        # Öğrenilebilir parametre: T matrisleri
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
            out.append(exp_neg.sum(0) - 1)

        out = torch.stack(out)
        # x ile concat et
        return torch.cat([x, out], dim=1)

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorNet, self).__init__()

        # İlk katmanlar (özellik çıkarımı)
        self.pre_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )

        # 🔥 Minibatch discrimination katmanı
        self.minibatch_layer = MinibatchDiscrimination(in_features=128, out_features=16, kernel_dims=3)

        # Son katmanlar (karar verici)
        self.post_net = nn.Sequential(
            spectral_norm(nn.Linear(128 + 16, 64)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.2),

            nn.Linear(64, 1)  # WGAN için: aktivasyon yok
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.minibatch_layer(x)
        x = self.post_net(x)
        return x


class GeneratorNet(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(GeneratorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
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

    def forward(self, z):
        x = self.net(z)
        return torch.tanh(x)

def initialize_gan():
    global generator, discriminator

    # Define Generator and Discriminator
    generator = GeneratorNet(z_dim=z_dim, output_dim=input_dim).to(device)
    discriminator = DiscriminatorNet(input_dim=input_dim).to(device)

    # Optimization
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0.0, 0.9),weight_decay=5e-5)

    return generator, discriminator, optimizer_g, optimizer_d

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    # fake = torch.ones(real_samples.size(0), 1).to(device)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # gradients = gradients.view(gradients.size(0), -1)
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#Training Loop for WGAN
def train_wgan_gp(train_loader, generator, discriminator, optimizer_g, optimizer_d, z_dim, device, epochs=epochs, lambda_gp=10, n_critic=5):
    d_loss_arr = []
    g_loss_arr = []

    for epoch in range(epochs):
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (real_data_mal, _) in enumerate(train_loader):
            real_data = real_data_mal.to(device)
            batch_size = real_data.size(0)

            # Discriminator (Critic) Training
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_data = generator(noise).detach()

                real_validity = discriminator(real_data)
                fake_validity = discriminator(fake_data)

                # Gradient Penalty calculation
                gp = compute_gradient_penalty(discriminator, real_data.data, fake_data.data, device)
                d_loss = fake_validity.mean() - real_validity.mean() + lambda_gp * gp

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

            #Generator Training
            noise = torch.randn(batch_size, z_dim).to(device)
            gen_data = generator(noise)
            g_loss = -discriminator(gen_data).mean()

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

        # Epoch averages
        avg_d_loss = d_epoch_loss / num_batches
        avg_g_loss = g_epoch_loss / num_batches
        d_loss_arr.append(avg_d_loss)
        g_loss_arr.append(avg_g_loss)

        print(f"[{epoch + 1}/{epochs}] Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")

    # Mode collapse analysis
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

