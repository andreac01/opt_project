from network import SimpleNN, SimpleCNN, settings, get_data_test, get_data_train, train_model_sgd, test_model
from plottings import get_loss_surface_3Dplot, get_loss_surface_plot
from torch import optim
from matplotlib.pyplot import close


######################################################
######################## GD ##########################
######################################################
print('\nGD')
for learning_rate in [2, 0.3, 2e-2, 1e-3]:
	print(f"Learning rate: {learning_rate}")
	# Create an instance of the model
	model = SimpleNN()
	model_copy = SimpleNN()
	model_copy.load_state_dict(model.state_dict())

	# Define the optimizer
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	optimizer_copy = optim.SGD(model_copy.parameters(), lr=learning_rate)

	inputs, labels = get_data_train()

	# Train the model
	train_size = int(settings.training_split * inputs.shape[0])
	losses = train_model_sgd(model, 100, settings.criterion, optimizer, inputs, labels, batch_size=train_size)

	# Test the model
	model.eval()
	test_inputs, test_labels = get_data_test()
	test_model(model, test_inputs, test_labels)
	model.train()

	# Plot3d mbatch loss
	data = test_inputs[:64]
	target = test_labels[:64]
	train_size = int(settings.training_split * inputs.shape[0])
	trajectory = get_loss_surface_3Dplot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=train_size)
	# Plot2d mbatch loss
	get_loss_surface_plot(model, settings.criterion, data, target, losses=losses, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=train_size, trajectory=trajectory)
	close('all')