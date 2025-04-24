import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from synth import Synth, Wave

class WaveIterableDataset(IterableDataset):
    def __init__(self, batch_size=100, param_mask=[True, True, True], duration=2.0, sample_rate=48000):
        self.batch_size = batch_size
        self.param_mask = param_mask
        self.duration = duration
        self.sample_rate = sample_rate
        self.default_params = {
            'frequency': 440,  # A4
            'phase': 0.5,     # Half-wavelength shift
            'volume': 0.8     # 80% volume
        }
        self.param_ranges = {
            'frequency': (110, 880),    # A2 to A5
            'phase': (0, 1),
            'volume': (0.2, 1.0)
        }
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
        """Iterator method to continuously yield data in batches."""
        while True:
            param_configs = []
            # Generate parameter configurations for the batch
            for _ in range(self.batch_size):
                params = self.default_params.copy()
                for i, vary in enumerate(self.param_mask):
                    if vary:
                        param_name = self.param_names[i]
                        params[param_name] = np.random.uniform(*self.param_ranges[param_name])
                param_configs.append(params)

            # Use ThreadPoolExecutor to parallelize sample generation
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [executor.submit(self.generate_sample, params) for params in param_configs]

                # Collect results
                batch_results = [future.result() for future in futures]

            # Separate inputs and targets from the results
            batch_x, batch_y = zip(*batch_results)

            # Convert batches to numpy arrays for compatibility
            batch_x = torch.tensor(batch_x, dtype=torch.float32)
            batch_y = torch.tensor(batch_y, dtype=torch.float32)

            yield batch_x, batch_y