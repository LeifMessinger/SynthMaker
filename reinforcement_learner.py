import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from synth import Wave
from synth_generator import WaveIterableDataset

class SynthRL:
	class SynthParamNetwork(nn.Module):
		"""Neural network to predict parameter adjustments for Wave synth"""
		def __init__(self, input_size, output_size):
			super().__init__()

			self.network = nn.Sequential(
				nn.Linear(input_size, 512),
				nn.ReLU(),
				nn.Linear(512, 256),
				nn.ReLU(),
				nn.Linear(256, 128),
				nn.ReLU(),
				nn.Linear(128, output_size),
				nn.Tanh()
			)
			
		def forward(self, audio, params):
			combined_input = torch.cat([audio, params])
			return self.network(combined_input)
	
	def __init__(self, device, dataset = WaveIterableDataset(), param_ranges=None, learning_rate=0.001, discount_factor=0.95, exploration_rate=0.3):
		self.device = device

		self.dataset = dataset

		self.param_ranges = dataset.param_ranges
		
		self.param_scales = torch.tensor([
			self.param_ranges['frequency'][1] - self.param_ranges['frequency'][0],
			self.param_ranges['phase'][1] - self.param_ranges['phase'][0],
			self.param_ranges['volume'][1] - self.param_ranges['volume'][0]
		], device=device)
		
		self.policy_net = self.SynthParamNetwork(96000 + 3, 3).to(device)
		
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
		self.discount_factor = discount_factor
		self.exploration_rate = exploration_rate
		self.sample_rate = self.dataset.sample_rate
		
	def _generate_audio(self, params):
		"""Generate audio using the current parameters"""
		wave = Wave(
			frequency=params[0].item(),
			phase=params[1].item(),
			volume=params[2].item(),
			sampleFrequency=self.sample_rate
		)
		return torch.tensor(wave.sample(duration=self.dataset.duration), 
						   device=self.device, dtype=torch.float32)
	
	def _calculate_reward(self, target_audio, predicted_audio, old_error=None):
		"""Calculate reward based on improvement in audio match"""
		new_error = torch.mean((target_audio - predicted_audio) ** 2)
		
		if old_error is None:
			return -new_error
		
		improvement = old_error - new_error
		return improvement
	
	def _normalize_params(self, params):
		"""Normalize parameters to [0,1] range based on param_ranges"""
		normalized = torch.zeros_like(params)
		
		normalized[0] = (params[0] - self.param_ranges['frequency'][0]) / self.param_scales[0]
		normalized[1] = (params[1] - self.param_ranges['phase'][0]) / self.param_scales[1]
		normalized[2] = (params[2] - self.param_ranges['volume'][0]) / self.param_scales[2]
		
		return normalized
		
	def _denormalize_params(self, normalized_params):
		"""Convert normalized parameters back to their original range"""
		denormalized = torch.zeros_like(normalized_params)
		

		denormalized[0] = normalized_params[0] * self.param_scales[0] + self.param_ranges['frequency'][0]
		denormalized[1] = normalized_params[1] * self.param_scales[1] + self.param_ranges['phase'][0]
		denormalized[2] = normalized_params[2] * self.param_scales[2] + self.param_ranges['volume'][0]
		
		return denormalized
	
	def _apply_adjustment(self, params, adjustment):
		"""Apply adjustment to parameters and ensure they stay within bounds"""
		scaled_adjustment = adjustment * (self.param_scales * 0.1)
		new_params = params + scaled_adjustment
	
		new_params[0] = torch.clamp(new_params[0], 
								   self.param_ranges['frequency'][0], 
								   self.param_ranges['frequency'][1])
		new_params[1] = torch.clamp(new_params[1], 
								   self.param_ranges['phase'][0], 
								   self.param_ranges['phase'][1])
		new_params[2] = torch.clamp(new_params[2], 
								   self.param_ranges['volume'][0], 
								   self.param_ranges['volume'][1])
		
		return new_params
	
	def train_step(self, target_audio, initial_params=None, max_steps=10):
		"""Train on a single audio sample"""
		self.policy_net.train()
		
		if initial_params is None:
			current_params = torch.tensor([
				np.random.uniform(*self.param_ranges['frequency']),
				np.random.uniform(*self.param_ranges['phase']),
				np.random.uniform(*self.param_ranges['volume'])
			], device=self.device, dtype=torch.float32)
		else:
			current_params = initial_params.clone().detach().to(self.device)
	
		target_audio = target_audio.to(self.device)
	
		transitions = []

		current_audio = self._generate_audio(current_params)
		current_error = torch.mean((target_audio - current_audio) ** 2)
		
		for step in range(max_steps):

			if np.random.random() < self.exploration_rate:
				adjustment = torch.tensor(np.random.uniform(-1, 1, size=3), 
										device=self.device, dtype=torch.float32)
			else:

				with torch.no_grad():
					adjustment = self.policy_net(target_audio, 
											   self._normalize_params(current_params))
			
			next_params = self._apply_adjustment(current_params, adjustment)
			next_audio = self._generate_audio(next_params)
			next_error = torch.mean((target_audio - next_audio) ** 2)
			reward = self._calculate_reward(target_audio, next_audio, current_error)
			
			transitions.append((
				(target_audio, current_params),
				adjustment,
				reward,
				(target_audio, next_params),
				step == max_steps - 1
			))
			
			current_params = next_params
			current_audio = next_audio
			current_error = next_error
		
		self._update_policy(transitions)
		
		return current_params
	
	def _update_policy(self, transitions):
		"""Update policy network using a batch of transitions"""
		self.optimizer.zero_grad()
		
		total_loss = 0
		for state, action, reward, next_state, done in transitions:
			target_audio, current_params = state
			next_target_audio, next_params = next_state
			
			predicted_adjustment = self.policy_net(
				target_audio, 
				self._normalize_params(current_params)
			)

			action_loss = torch.mean((predicted_adjustment - action) ** 2)

			reward_factor = max(0, reward.item())
			
			transition_loss = action_loss - reward_factor * 0.1
			total_loss += transition_loss
		
		avg_loss = total_loss / len(transitions)
		avg_loss.backward()
		self.optimizer.step()
	
	def predict(self, audio, params=None, num_steps=20):
		"""Find the best parameters to match the target audio"""
		self.policy_net.eval()
		
		audio = audio.to(self.device)
		
		if params is None:
			current_params = torch.tensor([
				440.0,
				0.5,
				0.8
			], device=self.device, dtype=torch.float32)
		else:
			current_params = params.clone().detach().to(self.device)
			print(f"Before parameters: {current_params}")
		
		current_audio = self._generate_audio(current_params)
		best_params = current_params.clone()
		best_error = torch.mean((audio - current_audio) ** 2)
		
		with torch.no_grad():
			for step in range(num_steps):
				adjustment = self.policy_net(
					audio,
					self._normalize_params(current_params)
				)
				
				next_params = self._apply_adjustment(current_params, adjustment)
				
				next_audio = self._generate_audio(next_params)
				next_error = torch.mean((audio - next_audio) ** 2)
				
				current_params = next_params
				
				if next_error < best_error:
					best_error = next_error
					best_params = next_params.clone()

		return best_params
	
	def train(self, num_epochs=100, samples_per_epoch=10, steps_per_sample=10):
		"""Train the model on multiple examples"""

		dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1)
		dataloader_iter = iter(dataloader)
		
		for epoch in range(num_epochs):
			epoch_loss = 0
			
			for _ in range(samples_per_epoch):

				try:
					audio, true_params = next(dataloader_iter)
				except StopIteration:
					dataloader_iter = iter(dataloader)
					audio, true_params = next(dataloader_iter)

				audio = audio[0].to(self.device)
				true_params = true_params[0].to(self.device)
				
				initial_params = torch.tensor([
					np.random.uniform(*self.param_ranges['frequency']),
					np.random.uniform(*self.param_ranges['phase']),
					np.random.uniform(*self.param_ranges['volume'])
				], device=self.device, dtype=torch.float32)
				
				final_params = self.train_step(audio, initial_params, max_steps=steps_per_sample)
				
				final_audio = self._generate_audio(final_params)
				final_error = torch.mean((audio - final_audio) ** 2)
				epoch_loss += final_error.item()
				
				param_mse = torch.mean((final_params - true_params) ** 2).item()
				
			avg_loss = epoch_loss / samples_per_epoch
			if epoch % 10 == 0:
				print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}, Param MSE = {param_mse:.6f}")
				
		print("Training completed!")