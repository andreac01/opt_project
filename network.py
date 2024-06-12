import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from filter_wise_normalization import get_loss_surface_3Dplot, get_loss_surface_plot, get_deltas, next_trajectory_point
from tqdm import tqdm
import numpy as np
# Set the seed for reproducibility
torch.manual_seed(42)

# Example neural network model
class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv3 = nn.Conv2d(20, 30, kernel_size=3)  # Add a new convolutional layer
		self.fc1 = nn.Linear(270, 64)  # Adjust the input size for the fully connected layer
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = torch.relu(F.max_pool2d(self.conv1(x), 2))
		x = torch.relu(F.max_pool2d(self.conv2(x), 2))
		x = torch.relu(self.conv3(x))  # Apply the new convolutional layer
		x = x.view(-1, 270)  # Adjust the reshape size
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class NN(nn.Module):
	def __init__(self):
		super(SimpleNN, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 256)
		self.bn1 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 128)
		self.bn2 = nn.BatchNorm1d(128)
		self.fc3 = nn.Linear(128, 64)
		self.bn3 = nn.BatchNorm1d(64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = torch.relu(self.fc1(x))
		x = self.bn1(x)
		x = torch.relu(self.fc2(x))
		x = self.bn2(x)
		x = torch.relu(self.fc3(x))
		x = self.bn3(x)
		x = self.fc4(x)
		return x

class SimpleNN(nn.Module):
	def __init__(self):
		super(SimpleNN, self).__init__()
		self.fc1 = nn.Linear(32*32*3, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 32)
		self.fc4 = nn.Linear(32, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = self.fc4(x)
		return x

def train_model(model, num_epochs, criterion, optimizer, inputs, labels):
	# Train the model
	for epoch in range(num_epochs):
		# Perform forward pass
		outputs = model(inputs)
		print(outputs.shape)
		loss = criterion(outputs, labels)

		# Perform backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Print the loss for monitoring
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def train_model_sgd(model, num_epochs, criterion, optimizer, inputs, labels, batch_size=64):
	losses = []
	# Divide inputs and labels into training and evaluation sets
	train_size = int(0.9 * inputs.size(0))
	train_inputs = inputs[:train_size]
	train_labels = labels[:train_size]
	eval_inputs = inputs[train_size:]
	eval_labels = labels[train_size:]
	best_loss = float('inf')
	# Train the model
	for epoch in range(num_epochs):
		# iterate over input batches
		with tqdm(total=train_inputs.size(0), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
			for i in range(0, train_inputs.size(0), batch_size):
				# Perform forward pass
				outputs = model(train_inputs[i:i+batch_size])
				loss = criterion(outputs, train_labels[i:i+batch_size])

				# Perform backward pass and optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Save the losses to repeat same training
				losses.append(loss)

				# Print the loss for monitoring
				pbar.set_postfix(loss=loss.item())
				pbar.update(batch_size)
		
		# Evaluate the model
		eval_loss = evaluate_model(model, criterion, eval_inputs, eval_labels)
		if eval_loss > 1.2 * best_loss or eval_loss > 3:
			print(f"Early stopping at epoch {epoch+1}")
			break
		best_loss = min(best_loss, eval_loss)

		# Remove the unnecessary batch count from the progress bar description
		pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
	return losses

def evaluate_model(model, criterion, inputs, labels):
	model.eval()
	# Evaluate the model
	with torch.no_grad():
		outputs = model(inputs)
		loss = criterion(outputs, labels)
	model.train()
	return loss

def test_model(model, inputs, labels):
	# Test the model
	with torch.no_grad():
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		correct = (predicted == labels).sum().item()
		print(f"Accuracy: {correct / labels.size(0) * 100:.2f}%")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_train():
	# Load the data from data_batch_i for i in range(1, 6)
	# Load all the data into a single data_dict
	data_dict = {b'data': [], b'labels': []}
	for i in range(1, 6):
		file_path = f'./data/data_batch_{i}'
		data = unpickle(file_path)
		data_dict[b'data'].extend(data[b'data'])
		data_dict[b'labels'].extend(data[b'labels'])
	data_dict[b'data'] = np.array(data_dict[b'data'])
	data_dict[b'labels'] = np.array(data_dict[b'labels'])
	# Extract inputs and labels from the data dictionary
	inputs = torch.tensor(data_dict[b'data']).view(-1, 3, 32, 32)
	# Convert inputs to float and normalize
	inputs = inputs.float() / 255.0

	# Reshape inputs to have shape (-1, 3, 32, 32)
	inputs = inputs.view(-1, 3, 32, 32)
	labels = torch.tensor(data_dict[b'labels'])

	# Move inputs and labels to the device (e.g. GPU) if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	inputs = inputs.to(device)
	labels = labels.to(device)
	return inputs, labels

def get_data_test():
	# Load the test data from test_batch
	test_data = unpickle('./data/test_batch')

	# Extract test inputs and labels from the test data dictionary
	test_inputs = torch.tensor(test_data[b'data']).view(-1, 3, 32, 32)
	# Convert test inputs to float and normalize
	test_inputs = test_inputs.float() / 255.0
	# Convert test inputs to greyscale
	# test_inputs = torch.mean(test_inputs, dim=1, keepdim=True)
	# Reshape test inputs to have shape (1, 1, 32, 32)
	test_inputs = test_inputs.view(-1, 3, 32, 32)
	test_labels = torch.tensor(test_data[b'labels'])

	# Move test inputs and labels to the device (e.g. GPU) if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	test_inputs = test_inputs.to(device)
	test_labels = test_labels.to(device)
	return test_inputs, test_labels

class settings:
	batch_size = 64
	num_epochs = 20
	learning_rate = 5e-3
	momentum = 0.9
	training_split = 0.9
	criterion = nn.CrossEntropyLoss()

######################################################
######################## SGD #########################
######################################################

for batch_size in [1, 64, 256]:
	for learning_rate in [2, 2e-3, 1e-5]:
		for momentum in [0, 0.9]:
			print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Momentum: {momentum}")
			# Create an instance of the model
			model = SimpleCNN()
			model_copy = SimpleCNN()
			model_copy.load_state_dict(model.state_dict())

			# Define the optimizer
			optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
			optimizer_copy = optim.SGD(model_copy.parameters(), lr=learning_rate, momentum=momentum)

			inputs, labels = get_data_train()
			if inputs.shape[0]/batch_size > 1000:
				inputs = inputs[:1000]
				labels = labels[:1000]
			# train data length should be divisible by 20*batch_size
			inputs_length = inputs.shape[0]-inputs.shape[0]%(10*batch_size)
			inputs = inputs[:inputs_length]
			labels = labels[:inputs_length]

			# Train the model
			losses = train_model_sgd(model, settings.num_epochs, settings.criterion, optimizer, inputs, labels, batch_size=batch_size)

			# Test the model
			model.eval()
			test_inputs, test_labels = get_data_test()
			print(len(test_inputs))
			# Test the model
			test_model(model, test_inputs, test_labels)
			model.train()

			# Plot3d mbatch loss
			data = test_inputs[:8]
			target = test_labels[:8]
			train_size = int(settings.training_split * inputs_length)
			trajectory = get_loss_surface_3Dplot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=batch_size)
			# Plot2d mbatch loss
			get_loss_surface_plot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=batch_size, trajectory=trajectory)







