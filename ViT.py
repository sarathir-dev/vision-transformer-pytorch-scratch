import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader


# patch embeddings
class PatchEmbeddings(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        # dimensionality of model
        self.d_model = d_model
        # image size
        self.img_size = img_size
        # patch size
        self.patch_size = patch_size
        # number of channels
        self.n_channels = n_channels

        self.linear_project = nn.Conv2d(
            self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    # B - Batch size
    # C - Image Channels
    # H - Image Height
    # # W - Image Width
    # p_col - Patch Column
    # p_row - Patch Row

    def forward(self, x):
        x = self.linear_project(x)  # (B, C, H, W) -> (B, d_model, P)
        x = x.flatten(2)  # (B, d_model, p_col, p_row) -> (B, d_model, P)
        x = x.transpose(1, 2)  # (B, d_models, P) -> (B, P, d_model)
        return x


# Class Token and Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        # classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # creating postional encoding
        positional_encoding = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    positional_encoding[pos][i] = np.sin(
                        pos / (10000 ** (i / d_model)))
                else:
                    positional_encoding[pos][i] = np.cos(
                        pos / (10000 ** ((i - 1) / d_model)))

        self.register_buffer('positional_encoding',
                             positional_encoding.unsqueeze(0))

    def forward(self, x):
        # expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        # adding class tokens to the begining of each embedding
        x = torch.cat((tokens_batch, x), dim=1)
        # add positional encodings to embeddings
        x = x + self.positional_encoding
        return x


# Class Attention Head
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # obtaining queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # dot product of queries and keys
        attention = Q @ K.transpose(-2, -1)

        # scaling
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        # combine attentio heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # sub-layer 1 normalization
        self.ln1 = nn.LayerNorm(d_model)

        # multi-head attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # sub-layer 2 normalization
        self.ln2 = nn.LayerNorm(d_model)

        # multilayer perceptraon
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x):
        # residual connection after sub-layer 1
        out = x + self.mha(self.ln1(x))
        # residual connection after sub-layer 2
        out = out * self.mlp(self.ln2(out))
        return out


# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[
            1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model  # dimensionality of model
        self.n_classes = n_classes  # number of classes
        self.img_size = img_size  # image size
        self.patch_size = patch_size  # patch size
        self.n_channels = n_channels  # number of channels
        self.n_heads = n_heads  # number of channels

        self.n_patches = (
            self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbeddings(
            self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.postional_encoding = PositionalEncoding(
            self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])

        # classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.postional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x


# Training Parameters
d_model = 9
n_classes = 10
img_size = (32, 32)
patch_size = (16, 16)
n_channels = 1
n_heads = 3
n_layers = 3
batch_size = 128
epochs = 5
alpha = 0.005


# Loading MNIST Dataset
transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])

train_set = MNIST(
    root="./../datasets", train=True, download=True, transform=transform
)
test_set = MNIST(
    root="./../datasets", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(
    device)})" if torch.cuda.is_available() else "")
transformer = VisionTransformer(
    d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
optimizer = Adam(transformer.parameters(), lr=alpha)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    training_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = transformer(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    print(f'Epoch: {
          epoch + 1} / {epochs}, Loss: {training_loss / len(train_loader):.3f}')


# Testing
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = transformer(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'\nModel Accuracy:  {100 * correct // total} %')


'''
Obtained Output:
Epoch: 1 / 5, Loss: 1.771
Epoch: 2 / 5, Loss: 1.649
Epoch: 3 / 5, Loss: 1.645
Epoch: 4 / 5, Loss: 1.650
Epoch: 5 / 5, Loss: 1.654
Model Accuracy:  81 %
'''
