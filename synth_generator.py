import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from synth import Synth, Wave

class WaveIterableDataset(IterableDataset):
	def __init__(self, param_mask=[True, True, True], duration=2.0, sample_rate=48000, param_ranges = {
			'frequency': (110, 3520),    # A2 to A5
			'phase': (0, 1),
			'volume': (0.2, 1.0)
		}):
		self.param_mask = param_mask
		self.duration = duration
		self.sample_rate = sample_rate
		self.default_params = {
			'frequency': 440,  # A4
			'phase': 0.5,     # Half-wavelength shift
			'volume': 0.8     # 80% volume
		}
		self.param_ranges = param_ranges
		self.param_names = ['frequency', 'phase', 'volume']

	def generate_sample(self, params):
		"""Generate a single training sample."""
		wave = Wave(
			frequency=params['frequency'],
			phase=params['phase'],
			volume=params['volume'],
			sampleFrequency=self.sample_rate
		)
		audio = wave.sample(duration=self.duration)
		return audio, [params['frequency'], params['phase'], params['volume']]

	def __iter__(self):
		"""Iterator method to continuously yield single samples."""
		while True:
			params = self.default_params.copy()
			for i, vary in enumerate(self.param_mask):
				if vary:
					param_name = self.param_names[i]
					params[param_name] = np.random.uniform(*self.param_ranges[param_name])

			# Generate a single sample
			audio, target = self.generate_sample(params)

			# Convert to tensors for compatibility
			audio = torch.tensor(audio, dtype=torch.float32)
			target = torch.tensor(target, dtype=torch.float32)

			yield audio, target
	
	def __getitem__(self, index):
		return self.__iter__().__next__()
