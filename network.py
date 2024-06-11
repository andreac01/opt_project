import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from filter_wise_normalization import get_loss_surface_3Dplot, get_loss_surface_plot, get_deltas, next_trajectory_point
from tqdm import tqdm

# Example neural network model
class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(500, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = torch.relu(F.max_pool2d(self.conv1(x), 2))
		x = torch.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 500)
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)
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
	d1, d2, theta = get_deltas(model)
	trajectory = []
	# Train the model
	for epoch in range(num_epochs):
		# iterate over input batches
		with tqdm(total=inputs.size(0), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
			for i in range(0, inputs.size(0), batch_size):
				# Perform forward pass
				outputs = model(inputs[i:i+batch_size])
				loss = criterion(outputs, labels[i:i+batch_size])

				# Perform backward pass and optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Get the next trajectory point
				trajectory.append(next_trajectory_point(model, theta, d1, d2, loss.item()))

				# Print the loss for monitoring
				pbar.set_postfix(loss=loss.item())
				pbar.update(batch_size)
	traj_alphas, traj_betas, losses = zip(*trajectory)
	return traj_alphas, traj_betas, losses

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

# Create an instance of the model
model = SimpleNN()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the number of epochs
num_epochs = 10

# Load the data from data_batch_i for i in range(1, 6)
# Load all the data into a single data_dict
# data_dict = {b'data': [], b'labels': []}
# for i in range(1, 6):
# 	file_path = f'./data/data_batch_{i}'
# 	data = unpickle(file_path)
# 	data_dict[b'data'].extend(data[b'data'])
# 	data_dict[b'labels'].extend(data[b'labels'])
# data_dict[b'data'] = np.array(data_dict[b'data'])
# data_dict[b'labels'] = np.array(data_dict[b'labels'])
data_dict = unpickle('./data/data_batch_1')
data_dict[b'data'] = data_dict[b'data'][:1000]
data_dict[b'labels'] = data_dict[b'labels'][:1000]

# Extract inputs and labels from the data dictionary
inputs = torch.tensor(data_dict[b'data']).view(-1, 3, 32, 32)
# Convert inputs to float and normalize
inputs = inputs.float() / 255.0
# Convert inputs to greyscale
# inputs = torch.mean(inputs, dim=1, keepdim=True)
# Reshape inputs to have shape (1, 1, 32, 32)
inputs = inputs.view(-1, 3, 32, 32)
labels = torch.tensor(data_dict[b'labels'])



# Move inputs and labels to the device (e.g. GPU) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = inputs.to(device)
labels = labels.to(device)

# Print the loss untrained model
model.eval()
data = inputs[:1]
target = labels[:1]
#get_loss_surface_plot(model, criterion, data, target)
model.train()

# Train the model
trajectory = train_model_sgd(model, num_epochs, criterion, optimizer, inputs, labels)



# Plot the loss surface
alphas, betas, _ = trajectory
max_alpha, min_alpha = max(alphas), min(alphas)
max_beta, min_beta = max(betas), min(betas)
epsilon = max(max(max_alpha, max_beta), abs(min(min_alpha, min_beta))) * 1.1
model.eval()
get_loss_surface_3Dplot(model, criterion, data, target, epsilon=epsilon, trajectory=trajectory)
get_loss_surface_plot(model, criterion, data, target, epsilon=epsilon, trajectory=trajectory)
model.train()



# Test the model
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
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

# Test the model
print("Test Accuracy:")
test_model(model, test_inputs, test_labels)
print("Training Accuracy:")
test_model(model, inputs, labels)