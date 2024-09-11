#  split -b 25M model.pth model_part_ ## use in command line to split file to 25MB chunks

import streamlit as st
import torch
import os
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from skimage.transform import resize
import requests
# import cv2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# path = os.path.abspath('')

# st.write("Current working directory:", os.getcwd())


st.set_page_config(layout="wide")

st.title("Phase from thermal history | A Test Web App")


option = st.selectbox("Select Data to use:", 
    options=[("Select Test Sample", 1), ("Upload Temperature Distribution", 0)],
    format_func=lambda x: x[0])[1]

# def transform_array(arr, target_shape=(201, 401)):
#     arr_resized = cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
#     return arr_resized

def transform_array(arr, target_shape=(201, 401)):
    arr_resized = resize(arr, (target_shape[0], target_shape[1]), order=0, mode='reflect', anti_aliasing=False)
    return arr_resized

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
        feature = self.features[idx].clone().detach().requires_grad_(True)
        label = self.labels[idx].clone().detach()
        return feature, label
    
# Function to calculate accuracy (for binary classification)
def calculate_accuracy(output, target):
    preds = torch.sigmoid(output) > 0.5
    correct = (preds == target).float()
    acc = correct.sum() / torch.numel(correct)
    return acc

# Initialize the model, loss function, optimizer
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load the entire model
# model = torch.load('6_Machine_Learning/trained_model/model.pth')
# model.eval()

# Load the state dict model for inference only
# model.load_state_dict(torch.load('6_Machine_Learning/trained_model/model.pth'))


# Function to join split files
def join_files(output_file, parts_dir, parts):
    with open(output_file, 'wb') as outfile:
        for part in parts:
            part_path = os.path.join(parts_dir, part)
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())

parts_dir = '6_Machine_Learning/trained_model'
parts = ['model_part_aa', 'model_part_ab', 'model_part_ac', 'model_part_ad','model_part_ae']
output_file = 'model.pth'
join_files(output_file, parts_dir, parts)

model.load_state_dict(torch.load(output_file))
model.eval()

if option == 0:
    # Display file uploader
    X_test_file = st.file_uploader("Upload Array", type="npy")
    if X_test_file:
        X_test_array = np.load(X_test_file)
        st.write(f"Temperature Distribution array file: {X_test_file.name}")
        # st.write(f"Using sample test data: {X_test_array.shape}")
        if X_test_array.shape != (201, 401):
            X_test_array = transform_array(X_test_array)
        X_test_array = X_test_array[np.newaxis, np.newaxis, :, :] 
        X_test = torch.tensor(X_test_array, dtype=torch.float32)
        # st.write(f"Using sample test data: {X_test_array.shape}")

        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                prediction = torch.sigmoid(outputs)
                prediction = (prediction > 0.5).float()
                prediction = prediction.cpu().numpy()

        laser_pos = np.argmax(X_test_array[0,0,0,:])
        st.divider()
        cm1, cm2 = st.columns(2)
        # Plot the predictions, ground truth, and error
        fig1, (ax1) = plt.subplots(1, 1, figsize=(12, 4))
        cmap = plt.get_cmap('RdYlBu_r')
        cmap.set_under('white', alpha=0)
        hmap1 = ax1.imshow(X_test_array[0][0], cmap='RdYlBu_r', vmin=300, aspect=0.5,  interpolation='quadric')
        ax1.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        ax1.set_title('Thermal Hisotry', pad=80, loc='left', fontsize=25, weight='bold')
        ax1.set_xlabel('$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold',)
        ax1.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='k', ha='center', va='center')
        ax1.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
        ax1.set_ylim(201, -1);  ax1.set_xlim(-1,401)

        contour_levels = [450, 700,  1337]
        label_colors = ['k', 'k', 'k']
        contour = ax1.contour(X_test_array[0][0], levels=contour_levels, colors='crimson', linewidths=3, linestyles='dashdot')
        labels = plt.clabel(contour, inline=True, fontsize=24, fmt='%1.0f K')
        for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color) 
        
        fig2, (ax2) = plt.subplots(1, 1, figsize=(12, 4))
        cmap = plt.get_cmap('RdYlGn_r')
        cmap.set_under('white', alpha=0)
        hmap2a = ax2.imshow(prediction[0][0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
        cmap = plt.get_cmap('Wistia')
        cmap.set_under('white', alpha=0) 
        hmap2b = ax2.imshow(1-prediction[0][0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
        ax2.set_title('ML Prediction', pad=80, loc='left', fontsize=25, weight='bold')
        ax2.set_xlabel('$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold', color='none')
        ax2.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='none', ha='center', va='center')
        ax2.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
        ax2.set_ylim(201, -1);  ax1.set_xlim(-1,401)

        contour_levels = [450, 700,  1337]
        label_colors = ['k', 'k', 'aqua']
        contour = ax2.contour(X_test_array[0][0], levels=contour_levels, colors='crimson', linewidths=3, linestyles='dashdot')
        labels = plt.clabel(contour, inline=True, fontsize=24, fmt='%1.0f K')
        for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color) 

        cm1.pyplot(fig1)
        cm2.pyplot(fig2)
       

    else:
        st.write("Please upload the Temperature Distribution 2D numpy array.")

elif option == 1:
    X_test_array = np.load('6_Machine_Learning/data/X_test.npy')
    y_test_array = np.load('6_Machine_Learning/data/y_test.npy')
    observation = st.slider("Select observation", min_value=1, max_value=X_test_array.shape[0], value=154) - 1
    X_test_array =  X_test_array[observation:observation+1,:,:,:]
    y_test_array = y_test_array[observation:observation+1,:,:,:]
    X_test = torch.tensor(X_test_array, dtype=torch.float32)
    y_test = torch.tensor(y_test_array, dtype=torch.float32)

    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # st.write(f"Using sample test data: {observation}")
    # st.write(f"Using sample test data: {X_test_array.shape}")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            prediction = torch.sigmoid(outputs)
            prediction = (prediction > 0.5).float()
            prediction = prediction.cpu().numpy()

    pred_error = prediction - np.array(y_test)

    laser_pos = np.argmax(X_test_array[0,0,0,:])

    cm1, cm2 = st.columns(2)
    # Plot the predictions, ground truth, and error
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('white', alpha=0)
    hmap1 = ax1.imshow(X_test_array[0][0], cmap='RdYlBu_r', vmin=300, aspect=0.5,  interpolation='quadric')
    ax1.set_title('Thermal Hisotry', pad=80, loc='left', fontsize=25, weight='bold')
    ax1.set_xlabel('$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold',)
    ax1.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='k', ha='center', va='center')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax1.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax1.set_ylim(201, -1);  ax1.set_xlim(-1,401)
    ax1.spines[:].set_linewidth(4)

    contour_levels = [450, 700,  1337]
    label_colors = ['k', 'k', 'k']
    contour = ax1.contour(X_test_array[0][0], levels=contour_levels, colors='crimson', linewidths=3, linestyles='dashdot')
    labels = plt.clabel(contour, inline=True, fontsize=24, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color) 

    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap2 = ax2.imshow(pred_error[0][0], cmap=cmap, vmin=-1, vmax=1, aspect=0.5)
    ax2.set_title('Prediction Error', pad=10, loc='left', fontsize=25, weight='bold')
    ax2.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax5 = fig1.add_axes([0.18, 0.02, 0.65, 0.1]) 
    cbar = fig1.colorbar(hmap2, cax=ax5, orientation='horizontal')
    # ax5.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar.ax.tick_params(labelsize=20, direction='inout', length=20, width=5, rotation=90) 
    cbar.set_ticks([-0.99,0,0.99], labels=['False Negative','Correct Pred.','False Positive'], weight='bold')
    ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='none', length_includes_head=True, clip_on=False)
    ax2.set_ylim(201, -1);  ax2.set_xlim(-1,401)
    ax2.spines[:].set_linewidth(4)

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap3a = ax3.imshow(y_test[0][0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap3b = ax3.imshow(1-y_test[0][0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax3.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax3.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax3.set_ylim(201, -1);  ax2.set_xlim(-1,401)
    ax3.set_title('Simulation Result', pad=80, loc='left', fontsize=25, weight='bold')
    ax3.set_xlabel('$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold', color='none')
    ax3.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='none', ha='center', va='center')
    ax3.spines[:].set_linewidth(4)

    contour_levels = [450, 700,  1337]
    label_colors = ['k', 'k', 'aqua']
    contour = ax3.contour(X_test_array[0][0], levels=contour_levels, colors='crimson', linewidths=3, linestyles='dashdot')
    labels = plt.clabel(contour, inline=True, fontsize=24, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color) 

    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap4a = ax4.imshow(prediction[0][0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap4b = ax4.imshow(1-prediction[0][0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax4.tick_params(axis='both', labelcolor='black', labelsize=65, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax4.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='none', length_includes_head=True, clip_on=False)
    ax4.set_ylim(201, -1);  ax2.set_xlim(-1,401)
    ax4.set_title('ML Predicted', pad=10, loc='left', fontsize=25, weight='bold')
    ax4.spines[:].set_linewidth(4)

    contour_levels = [450, 700,  1337]
    label_colors = ['k', 'k', 'aqua']
    contour = ax4.contour(X_test_array[0][0], levels=contour_levels, colors='crimson', linewidths=3, linestyles='dashdot')
    labels = plt.clabel(contour, inline=True, fontsize=24, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color) 


    cm1.pyplot(fig1)
    cm2.pyplot(fig2)

    # cm1.write(np.max(pred_error))
    # cm1.write(np.min(pred_error))
st.divider()

# Plotting function for losses and accuracies
def plot_training_history(history_path='6_Machine_Learning/trained_model/training_history.npy'):
    # Load the training history
    history = np.load(history_path, allow_pickle=True).item()
    
    epochs = range(1, len(history['train_loss']) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting loss
    ax1.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plotting accuracy
    ax2.plot(epochs, history['train_acc'], 'b', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    
    # Display the figure using Streamlit
    st.pyplot(fig)

st.title("ML Model Training History Visualization")
plot_training_history()