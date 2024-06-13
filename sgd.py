from network import SimpleNN, SimpleCNN, settings, get_data_test, get_data_train, train_model_sgd, test_model
from plottings import get_loss_surface_3Dplot, get_loss_surface_plot
from torch import optim
from matplotlib.pyplot import close

######################################################
######################## SGD #########################
######################################################
print('\nSGD')
for batch_size in [1, 64, 256]:
	for learning_rate in [2, 2e-3, 1e-5]:
		for momentum in [0, 0.9]:
			print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Momentum: {momentum}")
			# Create an instance of the model
			model = SimpleNN()
			model_copy = SimpleNN()
			model_copy.load_state_dict(model.state_dict())

			# Define the optimizer
			optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
			optimizer_copy = optim.SGD(model_copy.parameters(), lr=learning_rate, momentum=momentum)

			inputs, labels = get_data_train()
			if inputs.shape[0]/batch_size > 1000: # limit the training data size if the batch size is too small
				inputs = inputs[:1000]
				labels = labels[:1000]
			# train data length should be divisible by 10*batch_size
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
			data = inputs[:1024] #1024
			target = labels[:1024] #1024
			train_size = int(settings.training_split * inputs_length)
			trajectory = get_loss_surface_3Dplot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=batch_size)
			# Plot2d mbatch loss
			get_loss_surface_plot(model, settings.criterion, data, target, losses=losses, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=batch_size, trajectory=trajectory)
			close('all')
