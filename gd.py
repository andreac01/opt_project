from network import SimpleNN, SimpleCNN, settings, get_data_test, get_data_train, test_model, evaluate_model
from plottings import get_loss_surface_3Dplot, get_loss_surface_plot
from torch import optim
from matplotlib.pyplot import close

def train_model(model, num_epochs, criterion, optimizer, inputs, labels):
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
		# Perform forward pass
		outputs = model(train_inputs)
		loss = criterion(outputs, train_labels)

		# Perform backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Save the losses to repeat same training
		losses.append(loss)

		if epoch % 10 == 0:
			# Evaluate the model
			eval_loss = evaluate_model(model, criterion, eval_inputs, eval_labels)
			if eval_loss >= 1.5 * best_loss:
				break
			best_loss = min(best_loss, eval_loss)
			print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Eval loss: {eval_loss.item()}")

		# Remove the unnecessary batch count from the progress bar description
	return losses


######################################################
######################## GD ##########################
######################################################
print('\nGD')
for learning_rate in [1e-1]:#[2, 0.3, 2e-2, 1e-3]:
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
	losses = train_model(model, 1000, settings.criterion, optimizer, inputs, labels)

	# Test the model
	model.eval()
	test_inputs, test_labels = get_data_test()
	test_model(model, test_inputs, test_labels)
	model.train()

	# Plot3d mbatch loss
	data = inputs[:1024]
	target = labels[:1024]
	train_size = int(settings.training_split * inputs.shape[0])
	trajectory = get_loss_surface_3Dplot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=train_size)
	# Plot2d mbatch loss
	get_loss_surface_plot(model, settings.criterion, data, target, losses=losses, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=train_size, trajectory=trajectory)
	close('all')