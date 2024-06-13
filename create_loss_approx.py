from network import SimpleNN, SimpleCNN, settings, get_data_test, get_data_train, train_model_sgd, test_model
from plottings import plot_loss, plot_loss_3d
from torch import optim
import matplotlib.pyplot as plt

######################################################
######################## SGD #########################
######################################################
print('\nSGD')
batch_size = 64
learning_rate = 2e-3
momentum = 0.9

print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Momentum: {momentum}")
# Create an instance of the model
model = SimpleNN()
model_copy = SimpleNN()
model_copy.load_state_dict(model.state_dict())

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer_copy = optim.SGD(model_copy.parameters(), lr=learning_rate, momentum=momentum)

inputs, labels = get_data_train()

# train data length should be divisible by 10*batch_size
inputs_length = inputs.shape[0]-inputs.shape[0]%(10*batch_size)
inputs = inputs[:inputs_length]
labels = labels[:inputs_length]

# Train the model
losses = train_model_sgd(model, 1, settings.criterion, optimizer, inputs, labels, batch_size=batch_size)

# Test the model
model.eval()
test_inputs, test_labels = get_data_test()
print(len(test_inputs))
# Test the model
test_model(model, test_inputs, test_labels)
model.train()


# ######################################################
# ############### Loss approx. batches ####################
# ######################################################


# Plot3d mbatch loss
batch_sizes = [1, 64, 256, len(inputs)]
for batch_size in batch_sizes:
	data = inputs[:batch_size]
	target = labels[:batch_size]
	plot_loss_3d(model, settings.criterion, data, target, name=f'Batch-{batch_size}-partial', epsilon=0.1)
	plot_loss(model, settings.criterion, data, target, name=f'Batch-{batch_size}-partial', epsilon=0.1)
	plt.close('all')
# plt.show()
