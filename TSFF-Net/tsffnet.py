# import torch
# import numpy as np
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from preprocessing import load_bci_iv_2a_dataset
# from preprocessing import EEGDataset


# def raw_depth_attention(x):
#     """ x: input features with shape [N, C, H, W] """
#     N, C, H, W = x.size()
#     # K = W if W % 2 else W + 1
#     k = 7
#     adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
#     conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True).to(x.device)  # original kernel k
#     softmax = nn.Softmax(dim=-2)
#     x_pool = adaptive_pool(x)
#     x_transpose = x_pool.transpose(-2, -3)
#     y = conv(x_transpose)
#     y = softmax(y)
#     y = y.transpose(-2, -3)
#     return y * C * x


# class TSFF(nn.Module):

#     def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2, samples=1000, channels=3, avepool=25):
#         super(TSFF, self).__init__()
#         self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
#         nn.init.xavier_uniform_(self.channel_weight.data)

#         self.num_classes = num_classes
#         self.img_weight = img_weight

#         self.raw_time_conv = nn.Sequential(
#             nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
#             nn.BatchNorm2d(24),
#             nn.Conv2d(24, 24, kernel_size=(1, 75), groups=24, bias=False),
#             nn.BatchNorm2d(24),
#             nn.GELU(),
#         )

#         self.raw_chanel_conv = nn.Sequential(
#             nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
#             nn.BatchNorm2d(9),
#             nn.Conv2d(9, 9, kernel_size=(channels, 1), groups=9, bias=False),
#             nn.BatchNorm2d(9),
#             nn.GELU(),
#         )

#         self.raw_norm = nn.Sequential(
#             nn.AvgPool3d(kernel_size=(1, 1, avepool)),
#             nn.Dropout(p=0.65),
#         )

#         # raw features
#         raw_eeg = torch.ones((1, 1, channels, samples))
#         raw_eeg = torch.einsum('bdcw, hdc->bhcw', raw_eeg, self.channel_weight)
#         out_raw_eeg = self.raw_time_conv(raw_eeg)
#         out_raw_eeg = self.raw_chanel_conv(out_raw_eeg)
#         out_raw_eeg = self.raw_norm(out_raw_eeg)
#         out_raw_eeg_shape = out_raw_eeg.cpu().data.numpy().shape
#         print('out_raw_eeg_shape: ', out_raw_eeg_shape)
#         n_out_raw_eeg = out_raw_eeg_shape[-1] * out_raw_eeg_shape[-2] * out_raw_eeg_shape[-3]

#         self.frequency_features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=8),
#             nn.Dropout(p=0.25),

#             nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3),
#             nn.Dropout(p=0.25),

#             nn.Conv2d(32, out_raw_eeg_shape[-1], kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_raw_eeg_shape[-1]),
#             nn.Conv2d(out_raw_eeg_shape[-1], out_raw_eeg_shape[-1], kernel_size=4,
#                       groups=out_raw_eeg_shape[-1], bias=False, padding=2),
#             nn.BatchNorm2d(out_raw_eeg_shape[-1]),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3),
#             nn.Dropout(p=0.25),
#         )

   
#         img_eeg = torch.ones((1, 3, width, length))
#         out_img = self.frequency_features(img_eeg)
#         out_img_shape = out_img.cpu().data.numpy().shape
#         n_out_img = out_img_shape[-1] * out_img_shape[-2] * out_img_shape[-3]
#         print('n_out_img shape: ', out_img_shape)

#         self.classifier = nn.Sequential(
#             nn.Linear(n_out_img, num_classes),
#         )


#     def forward(self, x_raw, x_frequency):
#         # features for frequency graph
#         x_frequency = self.frequency_features(x_frequency)
#         x_frequency = x_frequency.view(x_frequency.size(0), -1)

#         x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)  
#         x_raw = self.raw_time_conv(x_raw)
#         x_raw = self.raw_chanel_conv(x_raw)
#         x_raw = raw_depth_attention(x_raw)
#         x_raw = self.raw_norm(x_raw)
#         # raw features and img features weighted fusion
#         # Check the order of magnitudes for both features.
#         x_raw_flatten = x_raw.view(x_raw.size(0), -1)

#         weighted_features = x_raw_flatten * (1-self.img_weight) + x_frequency * self.img_weight

#         x = self.classifier(weighted_features)

#         return x,  x_raw_flatten, x_frequency

# data_dir = "BCICIV_2b_gdf"
# eeg_data, labels = load_bci_iv_2a_dataset(data_dir)

# # Prepare raw EEG data (x_raw) and spectrograms (x_frequency)
# x_raw = np.expand_dims(eeg_data, axis=1)  # Shape: (trials, 1, channels, samples)

# # Generate spectrograms for x_frequency
# from scipy.signal import spectrogram

# def generate_spectrogram(eeg_data, fs=250):
#     spectrograms = []
#     for trial in eeg_data:
#         trial_spectrogram = []
#         for channel in trial:
#             f, t, Sxx = spectrogram(channel, fs=fs, nperseg=128, noverlap=64)
#             trial_spectrogram.append(Sxx)
#         spectrograms.append(np.array(trial_spectrogram))
#     return np.array(spectrograms)

# x_frequency = generate_spectrogram(eeg_data)  # Shape: (trials, channels, freq_bins, time_bins)
# x_frequency = np.transpose(x_frequency, (0, 2, 3, 1))  # Shape: (trials, freq_bins, time_bins, channels)

# # Convert to PyTorch tensors
# x_raw_tensor = torch.tensor(x_raw, dtype=torch.float32)
# x_frequency_tensor = torch.tensor(x_frequency, dtype=torch.float32).permute(0, 3, 1, 2)
# labels_tensor = torch.tensor(labels - 1, dtype=torch.long)  # Subtract 1 to make labels 0-indexed

# # Create dataset and DataLoader
# dataset = TensorDataset(x_raw_tensor, x_frequency_tensor, labels_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 4. Train TSFFNet
# model = TSFF(
#     img_weight=0.02,
#     width=x_frequency_tensor.shape[-2],
#     length=x_frequency_tensor.shape[-1],
#     num_classes=len(np.unique(labels)),
#     samples=x_raw_tensor.shape[-1],
#     channels=x_raw_tensor.shape[-2]
# )

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for x_raw_batch, x_freq_batch, labels_batch in dataloader:
#         optimizer.zero_grad()
#         outputs = model(x_raw_batch, x_freq_batch)
#         loss = criterion(outputs, labels_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


import os
import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Preprocess the Data
def preprocess_data(data_dir, tmin=0, tmax=4, filter_low=4., filter_high=38., epochs_duration=4):
    all_data = []
    all_labels = []

    for file in os.listdir(data_dir):
        if file.endswith(".gdf"):
            raw = mne.io.read_raw_gdf(os.path.join(data_dir, file), preload=True)
            
            # Bandpass filter the data using a 200-order Blackman window filter
            raw.filter(filter_low, filter_high, fir_design='firwin', skip_by_annotation='edge')
            
            # Find events and create epochs
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks='eeg', baseline=None, preload=True, event_repeated='merge')
            
            # Normalize each trial to [-1, 1] for all channels (based on max absolute value in the trial)
            data = epochs.get_data()  # Shape: (n_trials, n_channels, n_samples)
            data = normalize_data(data)
            
            # Labels are encoded in the event field (assuming 2 classes: 1 and 2)
            labels = epochs.events[:, -1] - 1  # Convert to 0-based labels
            all_data.append(data)
            all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_data, all_labels

def normalize_data(data):
    """
    Normalize each trial to [-1, 1] across all channels.
    """
    n_trials, n_channels, n_samples = data.shape
    for trial in range(n_trials):
        for channel in range(n_channels):
            # Normalize the trial for each channel
            max_val = np.max(np.abs(data[trial, channel, :]))
            if max_val != 0:  # Avoid division by zero
                data[trial, channel, :] = data[trial, channel, :] / max_val
    return data

# 2. Define the TSFF Model (same as provided)
class TSFF(nn.Module):
    def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2, samples=1000, channels=3, avepool=25):
        super(TSFF, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.num_classes = num_classes
        self.img_weight = img_weight

        self.raw_time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 75), groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.raw_chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(channels, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        self.raw_norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(p=0.25),

            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 9, kernel_size=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=4, groups=9, bias=False, padding=2),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(9 * 7 * 7, num_classes),  # Adjust dimensions if necessary
        )

    def forward(self, x_raw, x_frequency):
        # Frequency features
        x_frequency = self.frequency_features(x_frequency)
        x_frequency = x_frequency.view(x_frequency.size(0), -1)

        # Raw EEG features
        x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)
        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = self.raw_norm(x_raw)
        x_raw_flatten = x_raw.view(x_raw.size(0), -1)

        # Feature fusion
        weighted_features = x_raw_flatten * (1 - self.img_weight) + x_frequency * self.img_weight

        # Classification
        x = self.classifier(weighted_features)
        return x

# 3. Initialize and Train the Model
data_dir = "BCICIV_2b_gdf"  # Replace with your directory path
eeg_data, labels = preprocess_data(data_dir)

# Prepare raw EEG data (x_raw) and frequency features (x_frequency)
x_raw = np.expand_dims(eeg_data, axis=1)  # Shape: (trials, 1, channels, samples)

# Generate spectrograms for x_frequency (time-frequency representations)
def generate_spectrogram(eeg_data, fs=250):
    spectrograms = []
    for trial in eeg_data:
        trial_spectrogram = []
        for channel in trial:
            f, t, Sxx = mne.time_frequency.tfr_multitaper(channel, freqs=np.arange(4, 38, 1), time_bandwidth=2, n_cycles=4, return_itc=False)
            trial_spectrogram.append(Sxx)
        spectrograms.append(np.array(trial_spectrogram))
    return np.array(spectrograms)

x_frequency = generate_spectrogram(eeg_data)
x_frequency = np.transpose(x_frequency, (0, 2, 3, 1))  # Shape: (trials, freq_bins, time_bins, channels)

# Convert to PyTorch tensors
x_raw_tensor = torch.tensor(x_raw, dtype=torch.float32)
x_frequency_tensor = torch.tensor(x_frequency, dtype=torch.float32).permute(0, 3, 1, 2)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Create dataset and DataLoader
dataset = TensorDataset(x_raw_tensor, x_frequency_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the TSFF model
model = TSFF(
    img_weight=0.02,
    width=x_frequency_tensor.shape[-2],
    length=x_frequency_tensor.shape[-1],
    num_classes=len(np.unique(labels)),
    samples=x_raw_tensor.shape[-1],
    channels=x_raw_tensor.shape[-2]
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_raw_batch, x_freq_batch, labels_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_raw_batch, x_freq_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
