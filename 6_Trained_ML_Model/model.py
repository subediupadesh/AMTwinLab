from datetime import datetime; print("Starting date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torchmetrics import Accuracy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

## Loading Dataset
path = os.path.abspath('data/')
X, y = np.load(path+'/X.npy'), np.load(path+'/y.npy')

## Train Validation Test Split
# Reshape X and y to match PyTorch's Conv2d input format: (batch_size, channels, width, height)
X_reshaped = X[:, np.newaxis, :, :] 
y_reshaped = y[:, np.newaxis, :, :] 

# Split data into training, testing, validation sets
X_, X_test, y_, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.1, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.109, random_state=69)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

print(f'Feature [batch, channel, width, height] => Train: {X_train.shape} | Test: {X_test.shape} | Validation {X_val.shape}', flush=True) 
print(f'Label   [batch, channel, width, height] => Train: {y_train.shape} | Test: {y_test.shape} | Validation {y_val.shape}', flush=True) 


# UNet model definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.middle = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        middle = self.middle(self.pool(enc4))

        dec4 = self.upconv4(middle)
        dec4 = self.pad_and_crop(enc4, dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.pad_and_crop(enc3, dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.pad_and_crop(enc2, dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.pad_and_crop(enc1, dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final(dec1)
        return out

    def pad_and_crop(self, target, tensor):
        _, _, target_height, target_width = target.size()
        _, _, tensor_height, tensor_width = tensor.size()

        # Padding
        pad_h = max(0, target_height - tensor_height)
        pad_w = max(0, target_width - tensor_width)

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # Cropping
        delta_h = (tensor.size(2) - target_height) // 2
        delta_w = (tensor.size(3) - target_width) // 2

        return tensor[:, :, delta_h: delta_h + target_height, delta_w: delta_w + target_width]
    

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # feature = torch.tensor(self.features[idx], dtype=torch.float32)
        # label = torch.tensor(self.labels[idx], dtype=torch.float32)
        feature = self.features[idx].clone().detach().requires_grad_(True)
        label = self.labels[idx].clone().detach()
        return feature, label
    
# Function to calculate accuracy (for binary classification)
def calculate_accuracy(output, target):
    preds = torch.sigmoid(output) > 0.5
    correct = (preds == target).float()
    acc = correct.sum() / torch.numel(correct)
    return acc

# Initialize data loaders
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=13, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=13, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=13, shuffle=False)

# Custom progress bar function
def show_progress_bar(current, total, bar_length=20):
    """Displays a progress bar with arrows."""
    fraction = current / total
    arrow_count = int(fraction * bar_length)
    bar = '=' * arrow_count + '>' + ' ' * (bar_length - arrow_count - 1)
    progress_message = f'[{bar}] {current}/{total} batches'
    print(progress_message, end='\r', flush=True) 


# Training loop with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, history_path='training_history.npy'):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    bar_length = 20  # Length of the progress bar
    
    for epoch in range(num_epochs):
        t1 = datetime.now()
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        # Training phase
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += calculate_accuracy(outputs, targets).item() * inputs.size(0)

            # Show custom progress bar
            show_progress_bar(i + 1, len(train_loader), bar_length)

        print()  # New line after the progress bar

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_acc / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_acc += calculate_accuracy(outputs, targets).item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_acc / len(val_loader.dataset)

        # Store loss and accuracy for this epoch
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}", flush=True) 
        t2 = datetime.now()
        print(f"Training time for Epoch {epoch+1} is : {(t2-t1).total_seconds()} seconds", flush=True) 

    # Save training history to a numpy file
    np.save(history_path, history)
    print(f'Training history saved to {history_path}', flush=True) 

# Initialize the model, loss function, optimizer
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
num_epochs = 25
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

torch.save(model, 'Unet_model3.pth')
print('Model saved successfully!', flush=True) 

print("Ending date and time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
