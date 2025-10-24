import os
import numpy as np
import time as tm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(layout="wide", page_title="Phase & Velocity Prediction Viewer")

device = "cpu"
path = os.path.abspath('../..')

# -----------------------------------------------
# Model Definition
# -----------------------------------------------
class PhaseVelNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_in  = self.pool1(enc1_out)
        enc2_out = self.enc2(enc2_in)
        enc3_in  = self.pool2(enc2_out)
        enc3_out = self.enc3(enc3_in)
        bott_in  = self.pool3(enc3_out)
        bott_out = self.bottleneck(bott_in)
        up3 = self.up3(bott_out)
        dec3_in = self.crop_and_concat(up3, enc3_out)
        dec3_out = self.dec3(dec3_in)
        up2 = self.up2(dec3_out)
        dec2_in = self.crop_and_concat(up2, enc2_out)
        dec2_out = self.dec2(dec2_in)
        up1 = self.up1(dec2_out)
        dec1_in = self.crop_and_concat(up1, enc1_out)
        dec1_out = self.dec1(dec1_in)
        logits = self.final(dec1_out)  # (B,1,H,W)
        return logits

    def crop_and_concat(self, upsampled, encoder_features):
        up_h, up_w = upsampled.size(2), upsampled.size(3)
        enc_h, enc_w = encoder_features.size(2), encoder_features.size(3)
        dh, dw = enc_h - up_h, enc_w - up_w
        if dh > 0 or dw > 0:
            pad_left   = max(0, dw // 2)
            pad_right  = max(0, dw - pad_left)
            pad_top    = max(0, dh // 2)
            pad_bottom = max(0, dh - pad_top)
            upsampled = F.pad(upsampled, (pad_left, pad_right, pad_top, pad_bottom))
        if dh < 0 or dw < 0:
            crop_top  = (-dh) // 2
            crop_left = (-dw) // 2
            upsampled = upsampled[:, :, crop_top:crop_top + enc_h, crop_left:crop_left + enc_w]
        return torch.cat([upsampled, encoder_features], dim=1)

# ------------------------------
# Test Data Prediction Function
# ------------------------------
def Test_Data_Prediction():
    st.header("Prediction for Test Data: All Beams")
    Laser_type = st.selectbox("Select Laser Type", ("Gaussian", "FlatTop", "Bessel", "Ring"), index=2)
    t_step = st.slider("Select time step", 0, 12, 8)

    time = np.load(f'../Test_Data/{Laser_type}_time.npy')[t_step]
    laser_speed = 30
    laser_pos = (125 + time*laser_speed)* 401/1000  # Laser actual position in true dimension
    
    temperature = np.load(path+f'/7_Virtual_Digital_Twin/Test_Data/individual_temp_data/{Laser_type}_temp_{t_step}.npy')[0]
    phase = np.load(path+f'/7_Virtual_Digital_Twin/Test_Data/individual_phase_data/{Laser_type}_phase_{t_step}.npy')[0]
    velocity = np.load(path+f'/7_Virtual_Digital_Twin/Test_Data/individual_vel_data/{Laser_type}_vel_{t_step}.npy')[0]
    
    temp_mean, temp_std , eps = 622.8411254882812, 433.029052734375, 1e-8
    temperature_norm = (temperature - temp_mean) / (temp_std + eps)
    
    # -----------------------------------------------
    # Phase Prediction
    # -----------------------------------------------
    model_PN = PhaseVelNet(in_channels=1, out_channels=1).to(device)
    # state = torch.load(path+f'/6_Trained_ML_Models/Trained_Models/PhaseNet.pt", weights_only=False, map_location=device)
    state = torch.load(path+"/7_Virtual_Digital_Twin/src/PhaseNet.pt", weights_only=False, map_location=device)
    model_PN.load_state_dict(state)
    
    temperature_norm = torch.tensor(temperature_norm, dtype=torch.float32)[np.newaxis, np.newaxis, :, :]
    pred_phase = (torch.sigmoid(model_PN(temperature_norm)) > 0.5).float().numpy()[0]
    pred_error = pred_phase[0] - phase

    
    # -----------------------------------------------
    # Velocity Prediction
    # -----------------------------------------------
    model_VN = PhaseVelNet(in_channels=2, out_channels=1).to(device)
    # model_VN.load_state_dict(torch.load(path+f'/6_Trained_ML_Models/Trained_Models/VelNet.pt", map_location="cpu", weights_only=True))
    model_VN.load_state_dict(torch.load(path+"/7_Virtual_Digital_Twin/src/VelNet.pt", map_location="cpu", weights_only=True))
    Input_TempVel = np.stack([temperature_norm[0], phase[np.newaxis]], axis=1)
    Input_TempVel = torch.tensor(Input_TempVel, dtype=torch.float32)
    with torch.no_grad():
        pred_vel = model_VN(Input_TempVel).numpy()
        pred_vel = np.round(np.where(pred_vel < 1e-3, 0, pred_vel), decimals=2)[0,0]
    
    error_vel = np.abs(pred_vel - velocity)
    Liq_Area_Pred = ((1000*250)/(201*401)) * np.sum(pred_phase[0])
    Liq_Area_GT = ((1000*250)/(201*401)) * np.sum(phase)
    Perc_Area_Diff = np.abs(((Liq_Area_Pred-Liq_Area_GT)/Liq_Area_GT)*100)
    
    # -----------------------------------------------
    # Visualization
    # -----------------------------------------------
        
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(24, 20), frameon=True)
    
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('white', alpha=0)
    hmap1 = ax1.imshow(temperature, cmap='RdYlBu_r', vmin=300, aspect=0.5,  interpolation='quadric')
    ax1.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax1.set_ylim(201, -1);  ax1.set_xlim(-1,401)
    ax1.set_title(r't='+f'{time:.2f} s', pad=80, loc='left', fontsize=35, weight='bold', fontname='ADLaM Display')
    contour_levels = [450, 700, 1337]
    label_colors = ['white', 'yellow', 'red']
    contour = ax1.contour(temperature, levels=contour_levels, colors='black', linewidths=3, linestyles='dashed')
    labels = plt.clabel(contour, inline=True, fontsize=20, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color)
    ax1.spines[:].set_linewidth(4)
    ax1a = fig.add_axes([0.485, 0.742, 0.02, 0.108])
    cbar = fig.colorbar(hmap1, cax=ax1a)
    cbar.ax.tick_params(labelsize=20,length=0)
    cbar.set_ticks([np.min(temperature)*1.15, (np.max(temperature)+300)/2, np.max(temperature)*0.97], labels=[f'{np.min(temperature):.0f}', f'{(np.max(temperature)+300)/2 :.0f}', f'{np.max(temperature):.0f}'], weight='bold') 
    ax1a.spines[:].set_linewidth(0)
    ax1.text(412, 180, r'$\mathbf{\leftarrow}$T [K]$\mathbf{\rightarrow}$', fontsize=30, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    ax1.set_zorder(1)
    
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap2 = ax2.imshow(phase, cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap2 = ax2.imshow(1-phase, cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax2.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='none', length_includes_head=True, clip_on=False)
    ax2.set_ylim(201, -1);  ax4.set_xlim(-1,401)
    # ax2.set_title('FEM', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax2.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')   
    ax2.spines[:].set_linewidth(4)
    ax2.text(435, 190, r'FCC    LIQ', fontsize=33, color='k', rotation=90, weight='bold', fontfamily='ADLaM Display')
    ax2a = fig.add_axes([0.907, 0.742, 0.02, 0.108])
    cmap = ListedColormap(['#8B0000', '#FFA500'])
    data = np.array([[0], [1]])
    cbar = ax2a.imshow(data, cmap=cmap, aspect='auto')
    ax2a.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2a.spines[:].set_linewidth(0)
    
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap3a = ax3.imshow(pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap3b = ax3.imshow(1-pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax3.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax3.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='none', length_includes_head=True, clip_on=False)
    ax3.set_ylim(201, -1);  ax3.set_xlim(-1,401)
    # ax3.set_title('ML', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax3.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax3.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax3.spines[:].set_linewidth(4)
    ax3.text(435, 190, r'FCC    LIQ', fontsize=33, color='k', rotation=90, weight='bold', fontfamily='ADLaM Display')
    ax3a = fig.add_axes([0.485, 0.541, 0.02, 0.108])
    cmap = ListedColormap(['#8B0000', '#FFA500'])
    data = np.array([[0], [1]])
    cbar = ax3a.imshow(data, cmap=cmap, aspect='auto')
    ax3a.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax3a.spines[:].set_linewidth(0)
    
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap4 = ax4.imshow(pred_error, cmap=cmap, vmin=-1, vmax=1, aspect=0.5)
    # ax4.set_title('Error', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax4.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax4.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax4.set_ylim(201, -1);  ax1.set_xlim(-1,401)
    ax4.spines[:].set_linewidth(4)
    ax4a = fig.add_axes([0.907, 0.541, 0.02, 0.108])
    cbar = fig.colorbar(hmap4, cax=ax4a, orientation='vertical')
    cbar.ax.tick_params(labelsize=20, direction='inout', length=20, width=5, rotation=0) 
    cbar.set_ticks([-0.97,0,0.97], labels=['FN','CP','FP'], weight='bold', color='black', size=25, fontname='ADLaM Display')
    ax4a.spines[:].set_linewidth(0)
    
    hmap5 = ax5.imshow(pred_vel, cmap='gist_ncar_r', aspect=0.5,  interpolation='bilinear')
    ax5.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax5.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax5.set_ylim(201, -1);  ax5.set_xlim(-1,401)
    ax5.spines[:].set_linewidth(4)
    # ax5.set_title('ML', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax5.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax5a = fig.add_axes([0.485, 0.341, 0.02, 0.108])
    ax5a.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(hmap5, cax=ax5a, orientation='vertical')
    cbar.ax.tick_params(labelsize=25, direction='in', length=0) 
    ax5.text(412, 150, r'$\mathbf{[\mu m/s]}$', fontsize=25, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    cbar.set_ticks([np.max(pred_vel)*0.06, (np.max(pred_vel))/2, np.max(pred_vel)*0.96], labels=[f'{np.min(pred_vel):.0f}', f'{np.max(pred_vel)/2 :.0f}', f'{np.floor(np.max(pred_vel)):.0f}'], weight='bold') 
    ax5a.spines[:].set_linewidth(0)
    ax5.set_zorder(1)
    
    hmap6 = ax6.imshow(velocity, cmap='gist_ncar_r', aspect=0.5,  interpolation='bilinear')
    ax6.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax6.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax6.set_ylim(201, -1);  ax5.set_xlim(-1,401)
    ax6.spines[:].set_linewidth(4)
    # ax6.set_title('FEM', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax6.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax6a = fig.add_axes([0.907, 0.341, 0.02, 0.108])
    ax6a.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(hmap6, cax=ax6a, orientation='vertical')
    cbar.ax.tick_params(labelsize=25, direction='in', length=0) 
    ax6.text(412, 150, r'$\mathbf{[\mu m/s]}$', fontsize=25, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    cbar.set_ticks([np.max(velocity)*0.06, (np.max(velocity))/2, np.max(velocity)*0.96], labels=[f'{np.min(velocity):.0f}', f'{np.max(velocity)/2 :.0f}', f'{np.max(velocity):.0f}'], weight='bold') 
    ax6a.spines[:].set_linewidth(0)
    ax6.set_zorder(1)
    
    hmap7 = ax7.imshow(error_vel, cmap='hot_r', aspect=0.5,  interpolation='bilinear')
    ax7.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax7.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax7.set_ylim(201, -1);  ax5.set_xlim(-1,401)
    ax7.spines[:].set_linewidth(4)
    # ax7.set_title('Error', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax7.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax7a = fig.add_axes([0.485, 0.14, 0.02, 0.108])
    ax7a.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(hmap7, cax=ax7a, orientation='vertical')
    cbar.ax.tick_params(labelsize=25, direction='in', length=0) 
    ax7.text(412, 150, r'$\mathbf{[\mu m/s]}$', fontsize=25, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    cbar.set_ticks([np.max(error_vel)*0.06, (np.max(error_vel))/2, np.max(error_vel)*0.96], labels=[f'{np.min(error_vel):.0f}', f'{np.max(error_vel)/2 :.0f}', f'{np.max(error_vel):.0f}'], weight='bold') 
    ax7a.spines[:].set_linewidth(0)
    ax7.set_zorder(1)
    ax7.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='k', ha='center', va='center')
    ax7.set_xlabel(r'$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold',)
    
    # ax8.axis('off')               # hides axes, ticks, and frame
    # ax8.set_facecolor('none')     # makes background fully transparent
    
    ax8.set_position([0.55, 0.145, 0.35, 0.10])
    ax8.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax8.spines[:].set_linewidth(0)
    
    ax8.text(0.1, 0.8, f'FEM Calculated LIQUID Area: {Liq_Area_GT:.0f}'+r'$\mathbf{\mu m^2}$', size=30, color='green', weight='bold', rotation=0, fontname='Play')
    ax8.text(0.1, 0.5, f'ML     Predicted  LIQUID Area: {Liq_Area_Pred:.0f}'+r'$\mathbf{\mu m^2}$', size=30, color='blue', weight='bold', rotation=0, fontname='Play')
    ax8.text(0.1, 0.2, f'Area Prediction Error: {Perc_Area_Diff:.2f}%', size=30, color='red', weight='bold', rotation=0, fontname='Play')
    
    
    ax1.text(0, 240, r'(a) T-Distribution', size=30, weight='bold', rotation=0, fontname='Play')
    ax2.text(2, 190, r'(b) Phase Evolution (FEM)', size=30, weight='bold', rotation=0, fontname='Play')
    ax3.text(2, 190, r'(c) Predicted Phase (ML)', size=30, weight='bold', rotation=0, fontname='Play')
    ax4.text(2, 190, r'(d) Phase Prediction Error', size=30, weight='bold', rotation=0, fontname='Play')
    ax5.text(2, 190, r'(e) Predicted Velocity (ML)', size=30, weight='bold', rotation=0, fontname='Play')
    ax6.text(2, 190, r'(f) Meltpool Velocity (FEM)', size=30, weight='bold', rotation=0, fontname='Play')
    ax7.text(2, 185, r'(g) Velocity Prediction Error', size=30, weight='bold', rotation=0, fontname='Play')
    
    fig.text(0.44, 0.9, f'{Laser_type} HS', size=50, weight='bold', rotation=0, fontname='Play', color ='red')
    
        
    # plt.show()
    
    # -----------------------------------------------
    # Streamlit Display
    # -----------------------------------------------
    st.pyplot(fig)
    st.success(f"✅ Visualization complete for t_step = {t_step}, time = {time:.2f} s")


# ------------------------------
# Unseen Data Prediction Function (to fill later)
# ------------------------------

def run_prediction_step(t_step):
    """
    This function contains your actual prediction code for a given t_step.
    Everything related to data loading, model inference, and plotting goes here.
    """
    st.write(f"🧩 Running prediction for t_step = {t_step}")
    time = np.load(path+f'/7_Virtual_Digital_Twin/Unknown_Data/Bessel_time.npy')[t_step]
    laser_speed = 30
    laser_pos = (125 + time*laser_speed)* 401/1000  # Laser actual position in true dimension
    temperature = np.load(path+f'/7_Virtual_Digital_Twin/Unknown_Data/individual_temp_data/Bessel_temp_{t_step}.npy')[0]
    temp_mean, temp_std , eps = 622.8411254882812, 433.029052734375, 1e-8
    temperature_norm = (temperature - temp_mean) / (temp_std + eps)
    model_PN = PhaseVelNet(in_channels=1, out_channels=1).to(device)
    # state = torch.load(path+"/6_Trained_ML_Models/Trained_Models/PhaseNet.pt", weights_only=False, map_location=device)
    state = torch.load(path+"/7_Virtual_Digital_Twin/src/PhaseNet.pt", weights_only=False, map_location=device)
    model_PN.load_state_dict(state)
    temperature_norm = torch.tensor(temperature_norm, dtype=torch.float32)[np.newaxis, np.newaxis, :, :]
    pred_phase = (torch.sigmoid(model_PN(temperature_norm)) > 0.5).float().numpy()[0]
    model_VN = PhaseVelNet(in_channels=2, out_channels=1).to(device)
    # model_VN.load_state_dict(torch.load(path+"/6_Trained_ML_Models/Trained_Models/VelNet.pt", map_location="cpu", weights_only=True))
    model_VN.load_state_dict(torch.load(path+"/7_Virtual_Digital_Twin/src/VelNet.pt", map_location="cpu", weights_only=True))
    Input_TempVel = np.stack([temperature_norm[0], pred_phase], axis=1)
    Input_TempVel = torch.tensor(Input_TempVel, dtype=torch.float32)
    with torch.no_grad():
        pred_vel = model_VN(Input_TempVel).numpy()
        pred_vel = np.round(np.where(pred_vel < 1e-3, 0, pred_vel), decimals=2)[0,0]
    Liq_Area_Pred = ((1000*250)/(201*401)) * np.sum(pred_phase[0])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 8), frameon=True)
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('white', alpha=0)
    hmap1 = ax1.imshow(temperature, cmap='RdYlBu_r', vmin=300, aspect=0.5,  interpolation='quadric')
    ax1.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax1.set_ylim(201, -1);  ax1.set_xlim(-1,401)
    ax1.set_title(r't='+f'{time:.2f} s', pad=80, loc='left', fontsize=35, weight='bold', fontname='ADLaM Display')
    contour_levels = [450, 700, 1337]
    label_colors = ['white', 'yellow', 'red']
    contour = ax1.contour(temperature, levels=contour_levels, colors='black', linewidths=3, linestyles='dashed')
    labels = plt.clabel(contour, inline=True, fontsize=20, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color)
    ax1.spines[:].set_linewidth(4)
    ax1a = fig.add_axes([0.485, 0.57, 0.02, 0.27]) 
    cbar = fig.colorbar(hmap1, cax=ax1a)
    cbar.ax.tick_params(labelsize=20,length=0)
    cbar.set_ticks([np.min(temperature)*1.15, (np.max(temperature)+300)/2, np.max(temperature)*0.97], labels=[f'{np.min(temperature):.0f}', f'{(np.max(temperature)+300)/2 :.0f}', f'{np.max(temperature):.0f}'], weight='bold') 
    ax1a.spines[:].set_linewidth(0)
    ax1.text(412, 180, r'$\mathbf{\leftarrow}$T [K]$\mathbf{\rightarrow}$', fontsize=30, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    ax1.set_zorder(1)
    
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap2 = ax2.imshow(pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap2 = ax2.imshow(1-pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax2.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='none', length_includes_head=True, clip_on=False)
    ax2.set_ylim(201, -1);  ax4.set_xlim(-1,401)
    # ax2.set_title('FEM', pad=10, loc='center', fontsize=35, weight='bold', fontname='ADLaM Display')
    ax2.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax2.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')   
    ax2.spines[:].set_linewidth(4)
    ax2.text(435, 190, r'FCC    LIQ', fontsize=33, color='k', rotation=90, weight='bold', fontfamily='ADLaM Display')
    ax2a = fig.add_axes([0.907, 0.57, 0.02, 0.27]) 
    cmap = ListedColormap(['#8B0000', '#FFA500'])
    data = np.array([[0], [1]])
    cbar = ax2a.imshow(data, cmap=cmap, aspect='auto')
    ax2a.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2a.spines[:].set_linewidth(0)
    
    hmap4 = ax4.imshow(pred_vel, cmap='gist_ncar_r', aspect=0.5,  interpolation='bilinear')
    ax4.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax4.arrow(laser_pos, -80, 0, 76,  width = 8.5, color='red', length_includes_head=True, clip_on=False)
    ax4.set_ylim(201, -1);  ax3.set_xlim(-1,401)
    ax4.spines[:].set_linewidth(4)
    ax4.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax4a = fig.add_axes([0.907, 0.15, 0.02, 0.27]) 
    ax4a.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(hmap4, cax=ax4a, orientation='vertical')
    cbar.ax.tick_params(labelsize=25, direction='in', length=0) 
    ax4.text(412, 150, r'$\mathbf{[\mu m/s]}$', fontsize=25, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    cbar.set_ticks([np.max(pred_vel)*0.06, (np.max(pred_vel))/2, np.max(pred_vel)*0.96], labels=[f'{np.min(pred_vel):.0f}', f'{np.max(pred_vel)/2 :.0f}', f'{np.floor(np.max(pred_vel)):.0f}'], weight='bold') 
    ax4a.spines[:].set_linewidth(0)
    ax4.set_zorder(1)
    ax4.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='k', ha='center', va='center')
    ax4.set_xlabel(r'$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold',)
    
    ax3.set_position([0.13, 0.145, 0.35, 0.25])
    ax3.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax3.spines[:].set_linewidth(0)
    
    ax1.text(10, 260, r'Input T-Data', size=50, weight='bold', rotation=0, fontname='Play')
    ax2.text(10, 180, r'ML Pred. Phase', size=50, weight='bold', rotation=0, fontname='Play')
    ax3.text(0.15, 0.45, f'Predicted  LIQUID Area: {Liq_Area_Pred:.0f}'+r'$\mathbf{\mu m^2}$', size=40, weight='bold', rotation=0, fontname='Play')
    ax4.text(10, 180, r'ML Pred. Velocity', size=50, weight='bold', rotation=0, fontname='Play')


    # -----------------------------------------------
    # Streamlit Display
    # -----------------------------------------------
    st.pyplot(fig)
    st.success(f"✅ Visualization complete for t_step = {t_step}, time = {time:.2f} s")
    tm.sleep(0.3)  # just to simulate work time


def Unseen_Data_Prediction():
    st.header("Data From T-only Physics: Using Bessel Beam")

    # --- Mode selection ---
    mode = st.radio(
        "Select Prediction Mode:",
        ("🖐️ Manual Time Step Selection", "⚙️ Auto Run Predictions")
    )

    # --- Manual mode ---
    if mode == "🖐️ Manual Time Step Selection":
        t_step = st.slider("Select time step", 0, 49, 0)
        run_prediction_step(t_step)

    # --- Auto mode ---
    else:
        st.write("⏳ Auto-running predictions for t_step = 0 → 49 → 0 continuously...")
        start = st.button("▶️ Start Auto Run")

        if "auto_running" not in st.session_state:
            st.session_state.auto_running = False

        if start:
            st.session_state.auto_running = True

        stop = st.button("⏹️ Stop Auto Run")
        if stop:
            st.session_state.auto_running = False

        placeholder = st.empty()
        if st.session_state.auto_running:
            t_step = 0
            while st.session_state.auto_running:
                for t_step in range(50):
                    if not st.session_state.auto_running:
                        break
                    with placeholder.container():
                        run_prediction_step(t_step)
                    tm.sleep(0.3)
                # restart loop
                placeholder.info("🔁 Restarting from t_step = 0")


# ------------------------------
def UserUploaded_T_Data_Prediction():
    st.header("T-Data Uploaded by User in numpy tensor (From Thermal Camera)")

    uploaded_file = st.file_uploader("📂 Upload Temperature Data (.npy file)", type=["npy"])
    
    if uploaded_file is not None:
        temperature = np.load(uploaded_file)[0]
        st.success(r"✅ File uploaded successfully! Dimension: 1000$\mu m$ $\times$ 250 $\mu m$")
    else:
        st.warning("⚠️ Please upload a .npy temperature file to proceed.")
        st.stop()
    
    # temperature = np.load(f'../Unknown_Data/individual_temp_data/Bessel_temp_{t_step}.npy')[0]
    
    temp_mean, temp_std , eps = 622.8411254882812, 433.029052734375, 1e-8
    temperature_norm = (temperature - temp_mean) / (temp_std + eps)
    model_PN = PhaseVelNet(in_channels=1, out_channels=1).to(device)
    # state = torch.load("../../6_Trained_ML_Models/Trained_Models/PhaseNet.pt", weights_only=False, map_location=device)
    state = torch.load(path+"/7_Virtual_Digital_Twin/src/PhaseNet.pt", weights_only=False, map_location=device)
    model_PN.load_state_dict(state)
    temperature_norm = torch.tensor(temperature_norm, dtype=torch.float32)[np.newaxis, np.newaxis, :, :]
    pred_phase = (torch.sigmoid(model_PN(temperature_norm)) > 0.5).float().numpy()[0]
    model_VN = PhaseVelNet(in_channels=2, out_channels=1).to(device)
    # model_VN.load_state_dict(torch.load("../../6_Trained_ML_Models/Trained_Models/VelNet.pt", map_location="cpu", weights_only=True))
    model_VN.load_state_dict(torch.load(path+"/7_Virtual_Digital_Twin/src/VelNet.pt", map_location="cpu", weights_only=True))
    Input_TempVel = np.stack([temperature_norm[0], pred_phase], axis=1)
    Input_TempVel = torch.tensor(Input_TempVel, dtype=torch.float32)
    with torch.no_grad():
        pred_vel = model_VN(Input_TempVel).numpy()
        pred_vel = np.round(np.where(pred_vel < 1e-3, 0, pred_vel), decimals=2)[0,0]
    Liq_Area_Pred = ((1000*250)/(201*401)) * np.sum(pred_phase[0])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 8), frameon=True)
    cmap = plt.get_cmap('RdYlBu_r')
    cmap.set_under('white', alpha=0)
    hmap1 = ax1.imshow(temperature, cmap='RdYlBu_r', vmin=300, aspect=0.5,  interpolation='quadric')
    ax1.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.set_ylim(201, -1);  ax1.set_xlim(-1,401)
    contour_levels = [450, 700, 1337]
    label_colors = ['white', 'yellow', 'red']
    contour = ax1.contour(temperature, levels=contour_levels, colors='black', linewidths=3, linestyles='dashed')
    labels = plt.clabel(contour, inline=True, fontsize=20, fmt='%1.0f K')
    for label, color in zip(labels, label_colors): label.set_fontweight('bold');label.set_color(color)
    ax1.spines[:].set_linewidth(4)
    ax1a = fig.add_axes([0.485, 0.57, 0.02, 0.27]) 
    cbar = fig.colorbar(hmap1, cax=ax1a)
    cbar.ax.tick_params(labelsize=20,length=0)
    cbar.set_ticks([np.min(temperature)*1.15, (np.max(temperature)+300)/2, np.max(temperature)*0.97], labels=[f'{np.min(temperature):.0f}', f'{(np.max(temperature)+300)/2 :.0f}', f'{np.max(temperature):.0f}'], weight='bold') 
    ax1a.spines[:].set_linewidth(0)
    ax1.text(412, 180, r'$\mathbf{\leftarrow}$T [K]$\mathbf{\rightarrow}$', fontsize=30, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    ax1.set_zorder(1)
    
    cmap = plt.get_cmap('RdYlGn_r')
    cmap.set_under('white', alpha=0)
    hmap2 = ax2.imshow(pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.0, aspect=0.5)
    cmap = plt.get_cmap('Wistia')
    cmap.set_under('white', alpha=0) 
    hmap2 = ax2.imshow(1-pred_phase[0], cmap=cmap, vmin=0.5, vmax=1.5, aspect=0.5, interpolation='quadric')
    ax2.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.set_ylim(201, -1);  ax4.set_xlim(-1,401)
    ax2.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')   
    ax2.spines[:].set_linewidth(4)
    ax2.text(435, 190, r'FCC    LIQ', fontsize=33, color='k', rotation=90, weight='bold', fontfamily='ADLaM Display')
    ax2a = fig.add_axes([0.907, 0.57, 0.02, 0.27]) 
    cmap = ListedColormap(['#8B0000', '#FFA500'])
    data = np.array([[0], [1]])
    cbar = ax2a.imshow(data, cmap=cmap, aspect='auto')
    ax2a.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2a.spines[:].set_linewidth(0)
    
    hmap4 = ax4.imshow(pred_vel, cmap='gist_ncar_r', aspect=0.5,  interpolation='bilinear')
    ax4.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax4.set_ylim(201, -1);  ax3.set_xlim(-1,401)
    ax4.spines[:].set_linewidth(4)
    ax4.contour(temperature, levels=[1337], colors='black', linewidths=3, linestyles='dashed')
    ax4a = fig.add_axes([0.907, 0.15, 0.02, 0.27]) 
    ax4a.tick_params(axis='both', labelcolor='black', labelsize=1, bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(hmap4, cax=ax4a, orientation='vertical')
    cbar.ax.tick_params(labelsize=25, direction='in', length=0) 
    ax4.text(412, 150, r'$\mathbf{[\mu m/s]}$', fontsize=25, color='k', rotation=90, weight='bold', fontname='ADLaM Display',  zorder=20)
    cbar.set_ticks([np.max(pred_vel)*0.06, (np.max(pred_vel))/2, np.max(pred_vel)*0.96], labels=[f'{np.min(pred_vel):.0f}', f'{np.max(pred_vel)/2 :.0f}', f'{np.floor(np.max(pred_vel)):.0f}'], weight='bold') 
    ax4a.spines[:].set_linewidth(0)
    ax4.set_zorder(1)
    ax4.annotate(r'$\mathbf{\leftarrow}$ 250 $\mathbf{\mu m}$ $\mathbf{\rightarrow}$', xy=(0.5, 0.5), xytext=(-0.025, 0.5), rotation=90, xycoords='axes fraction', textcoords='axes fraction', fontsize = 22, weight = 'bold', color='k', ha='center', va='center')
    ax4.set_xlabel(r'$\mathbf{\longleftarrow}$                 1000 $ \mathbf{\mu m}$                $\mathbf{\longrightarrow}$', fontsize = 27, weight ='bold',)
    
    ax3.set_position([0.13, 0.145, 0.35, 0.25])
    ax3.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax3.spines[:].set_linewidth(0)
    
    ax1.text(10, 260, r'Input T-Data', size=50, weight='bold', rotation=0, fontname='Play')
    ax2.text(10, 180, r'ML Pred. Phase', size=50, weight='bold', rotation=0, fontname='Play')
    ax3.text(0.15, 0.45, f'Predicted  LIQUID Area: {Liq_Area_Pred:.0f}'+r'$\mathbf{\mu m^2}$', size=40, weight='bold', rotation=0, fontname='Play')
    ax4.text(10, 180, r'ML Pred. Velocity', size=50, weight='bold', rotation=0, fontname='Play')

    # plt.show()

    # -----------------------------------------------
    # Streamlit Display
    # -----------------------------------------------
    st.pyplot(fig)
    st.success(f"✅ Visualization complete for uploaded Temperature Tensor Data")



# ------------------------------
# Main Streamlit App
# ------------------------------
st.title("Virtual Digital Twin: Phase & Velocity Prediction")

option = st.sidebar.radio( "Select Mode:", ("Upload T-Tensor Data", "Unseen Data Prediction", "Test Data Prediction"))

if option == "Upload T-Tensor Data":
    st.write("Uploade Temperature Data Tensor in Numpy Format")
    UserUploaded_T_Data_Prediction()
elif option == "Unseen Data Prediction":
    Unseen_Data_Prediction()
elif option == "Test Data Prediction":
    Test_Data_Prediction()