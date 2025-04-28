import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T

class AudioFeatureExtractor(nn.Module):
	def __init__(self, sample_rate=48000, n_fft=2048, n_mels=128):
		super().__init__()
		self.mel_spectrogram = T.MelSpectrogram(
			sample_rate=sample_rate,
			n_fft=n_fft,
			hop_length=512,
			n_mels=n_mels,
			normalized=True
		)
		
	def forward(self, x):
		mel_spec = self.mel_spectrogram(x)
		return mel_spec


class SynthParameterPredictor(nn.Module):
	def __init__(self, input_dim=128, hidden_dim=256, output_dim=3):
		super().__init__()
				
		self.conv_layers = nn.Sequential(
			#Conv layers
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),

			#Linear layers
			nn.Flatten(),
			nn.LazyLinear(hidden_dim),
			#nn.Linear(64 * 16 * 23, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hidden_dim // 2, output_dim)
		)
		
	def forward(self, x):
		return self.conv_layers(x)
	
def normalize_params(params, param_ranges = {
		'frequency': (110, 3520),    # A2 to A5
		'phase': (0, 1),
		'volume': (0.2, 1.0)
	}):
	"""Normalize parameters to [0, 1] range"""
	normalized = torch.zeros_like(params)
	normalized[0] = (params[0] - param_ranges['frequency'][0]) / (param_ranges['frequency'][1] - param_ranges['frequency'][0])  
	normalized[1] = params[1]  
	normalized[2] = (params[2] - 0.2) / (1.0 - 0.2)  
	return normalized

def denormalize_params(params, param_ranges = {
		'frequency': (110, 3520),    # A2 to A5
		'phase': (0, 1),
		'volume': (0.2, 1.0)
	}):
	"""Convert normalized parameters back to original range"""
	denorm = torch.zeros_like(params)
	denorm[0] = params[0] * (param_ranges['frequency'][1] - param_ranges['frequency'][0]) + param_ranges['frequency'][0]
	denorm[1] = params[1]
	denorm[2] = params[2] * (param_ranges['volume'][1] - param_ranges['volume'][0]) + param_ranges['volume'][0]
	return denorm

def normalize_batch(batch, param_ranges = {
		'frequency': (110, 3520),    # A2 to A5
		'phase': (0, 1),
		'volume': (0.2, 1.0)
	}):
	return torch.stack([normalize_params(params, param_ranges) for params in batch])

def denormalize_batch(batch, param_ranges = {
		'frequency': (110, 3520),    # A2 to A5
		'phase': (0, 1),
		'volume': (0.2, 1.0)
	}):
	return torch.stack([denormalize_params(params, param_ranges) for params in batch])