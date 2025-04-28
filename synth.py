import numpy as np
import torch

# Implement the Synth and Wave classes with optimizations
class Synth:
	def __init__(self, sampleFrequency=48000):
		self.sampleFrequency = sampleFrequency
	
	def __call__(self, t):
		"""Sample the synthesizer at a specific point in time t (in seconds)"""
		raise NotImplementedError("Subclasses must implement __call__")
	
	def sample(self, duration, startTime=0):
		"""Create an array of samples for the given duration starting at startTime"""
		num_samples = int(duration * self.sampleFrequency)
		times = np.linspace(startTime, startTime + duration, num_samples, endpoint=False)
		return np.array([self(t) for t in times])

class Wave(Synth):
	def __init__(self, frequency=440, phase=0, volume=1, sampleFrequency=48000):
		super().__init__(sampleFrequency)
		self.frequency = frequency
		self.phase = phase
		self.volume = volume
	
	def __call__(self, t):
		"""Generate a sine wave sample at time t with the specified phase and volume using torch"""
		phase_radians = self.phase * 2 * np.pi
		return self.volume * torch.sin(2 * np.pi * self.frequency * t + phase_radians)
	
	def sample(self, duration, startTime=0):
		"""Optimized version that generates the entire array at once using torch"""
		self.phase = torch.tensor(self.phase, requires_grad=True)
		self.frequency = torch.tensor(self.frequency, requires_grad=True)
		self.volume = torch.tensor(self.volume, requires_grad=True)
		num_samples = int(duration * self.sampleFrequency)
		times = torch.linspace(startTime, startTime + duration, num_samples)
		phase_radians = self.phase * 2 * np.pi
		return self.volume * torch.sin(2 * np.pi * self.frequency * times + phase_radians)
	
	@staticmethod
	def from_tensor(tensor):
		"""Create a Wave instance from a tensor"""
		return Wave(frequency=tensor[0], phase=tensor[1], volume=tensor[2])