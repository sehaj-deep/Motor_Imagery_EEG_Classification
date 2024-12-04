import os
import numpy as np
import torch
import torch.nn as nn
import mne 
from torch.utils.data import DataLoader, TensorDataset

# # 1. Load and preprocess the BCI IV 2a dataset
def load_bci_iv_2a_dataset(data_dir, tmin=0, tmax=4, filter_low=8., filter_high=30.):
    """
    Load and preprocess all .gdf files from the BCI IV 2a dataset.
    """
    all_data = []
    all_labels = []

    for file in os.listdir(data_dir):
        if file.endswith(".gdf"):
            # Load raw data
            raw = mne.io.read_raw_gdf(os.path.join(data_dir, file), preload=True)
            
            # Bandpass filter
            raw.filter(filter_low, filter_high, fir_design='firwin', skip_by_annotation='edge')
            
            # Extract events and labels
            events, event_id = mne.events_from_annotations(raw)
            
            # Extract epochs
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks='eeg', baseline=None, preload=True, event_repeated='merge')
            
            # Collect data and labels
            data = epochs.get_data()  # Shape: (trials, channels, samples)
            labels = epochs.events[:, -1]  # Corresponding labels
            
            all_data.append(data)
            all_labels.append(labels)

    # Concatenate data and labels from all subjects/files
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_data, all_labels

# class EEGDataset(Dataset):
#     def __init__(self, data_dir, channels=22, samples=178, transform=None):
#         self.data_dir = data_dir
#         self.channels = channels
#         self.samples = samples
#         self.transform = transform
#         self.data, self.labels = self.load_bci_iv_2a_dataset()

#     def load_bci_iv_2a_dataset(self):
#         # Load data using MNE (adjust path and loading procedure as per dataset)
#         eeg_data = []
#         labels = []
        
#         for file in os.listdir(self.data_dir):
#             if file.endswith('.gdf'):
#                 raw = mne.io.read_raw_gdf(os.path.join(self.data_dir, file), preload=True)
#                 raw.filter(1, 40, fir_design='firwin')  # Filter for EEG range (1-40Hz)
#                 events, _ = mne.events_from_annotations(raw)
#                 epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=3, baseline=None, preload=True)
#                 eeg_data.append(epochs.get_data())  # Raw EEG data (trials x channels x time)
#                 labels.append(epochs.events[:, -1])  # Event labels (trial labels)
        
#         return np.concatenate(eeg_data, axis=0), np.concatenate(labels, axis=0)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         label = self.labels[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # Initialize the dataset and DataLoader


