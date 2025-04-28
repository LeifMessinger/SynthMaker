import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from synth import Wave

class SynthParameterAdjuster(nn.Module):
	def __init__(self, batch_size = 100):
		super(SynthParameterAdjuster, self).__init__()
		self.model = nn.Sequential(
			nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.LazyLinear(128),
			#nn.Linear(32 * batch_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 3)
		)

	def forward(self, predicted_audio, true_audio):

		x = torch.stack((predicted_audio, true_audio), dim=1)


		return self.model(x)

def train_with_adjustment(base_model, adjustment_model, dataset, num_epochs=10, batch_size=100, learning_rate=0.001, device="cuda"):
	dataloader = DataLoader(dataset, batch_size=batch_size)
	base_model = base_model.to(device).eval()
	adjustment_model = adjustment_model.to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(adjustment_model.parameters(), lr=learning_rate)
	scaler = torch.amp.GradScaler()

	adjustment_model.train()
	for epoch in range(num_epochs):
		audio, true_params = next(iter(dataloader))
		audio, true_params = audio.to(device), true_params.to(device)

		with torch.no_grad():
			from parameter_predictor import denormalize_batch
			base_params = denormalize_batch(base_model(audio), param_ranges=dataset.param_ranges)
			predicted_audio = torch.stack([
				Wave.from_tensor(single).sample(duration=2.0)
				for single in base_params
			])
		
		target_adjustments = true_params - base_params

		optimizer.zero_grad()
		with torch.amp.autocast(device_type=device.type):
			param_adjustments = adjustment_model(predicted_audio, audio)
			loss = criterion(param_adjustments, target_adjustments)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

	return adjustment_model

def predict_with_adjustment(audio, base_model, adjustment_model, device):
	base_model = base_model.to(device).eval()
	adjustment_model = adjustment_model.to(device).eval()
	audio = audio.to(device)

	with torch.no_grad():
		base_params = base_model(audio.unsqueeze(0))
		from parameter_predictor import denormalize_params
		param_adjustments = adjustment_model(Wave.from_tensor(denormalize_params(base_params.squeeze(0))).sample(duration=2.0).unsqueeze(0), audio.unsqueeze(0))
		refined_params = base_params + param_adjustments

	return refined_params.squeeze(0), base_params.squeeze(0)
