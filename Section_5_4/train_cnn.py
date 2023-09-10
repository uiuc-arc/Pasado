import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np

import argparse

from model import Conv

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Train MNIST CNN')
parser.add_argument('--net', choices=['small', 'med', 'big'], help='CNN architecture', default='big')
args = parser.parse_args()

net_type = args.net

transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor()])

train_val_dataset = datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)

num_train = len(train_val_dataset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=128, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=1024, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if net_type == 'small':
    model = Conv(device, 28, [(16, 4, 2, 0), (32, 4, 2, 0)], [100, 10]).to(device)  # ConvSmall.
elif net_type == 'med':
    model = Conv(device, 28, [(16, 4, 2, 1), (32, 4, 2, 1)], [1000, 10]).to(device)  # ConvMed.
elif net_type == 'big':
    model = Conv(device, 28, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 3, 1, 1), (64, 4, 2, 1)], [512, 512, 10]).to(
        device)  # ConvBig.
else:
    raise ValueError("Undefined CNN architecture!")

print(f'======== Training Conv{net_type.capitalize()} ========')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

max_grad_norm = 10.


def train(model, loader, criterion, optimizer, device):
    model.train()

    pbar = tqdm(loader)
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        pbar.set_description(f"Train Epoch {epoch}: Loss {running_loss / ((i + 1) * loader.batch_size):.3f}")


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    print(f'Validation loss: {avg_loss:.3f}')
    return avg_loss


def test(model, loader, device):
    model.eval()
    correct = 0
    correct_indices = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            matches = (predicted == labels).cpu().numpy()
            correct += (predicted == labels).sum().item()
            correct_indices.extend(idx for idx, match in enumerate(matches) if match)

    print(f'Test Accuracy: {100 * correct / len(loader.dataset)} %')
    return correct_indices


if __name__ == '__main__':

    n_epochs = 100

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.3f} --> {val_loss:.3f}).  Saving model ...')
            torch.save(model.state_dict(), f'trained/conv{net_type}.pth')
            best_val_loss = val_loss

    model.load_state_dict(torch.load(f'trained/conv{net_type}.pth'))
    correct_indices = test(model, test_loader, device)
    torch.save(correct_indices, f'trained/indices_conv{net_type}.pth')
