from network import SimpleNN, SimpleCNN, settings, get_data_test, get_data_train, train_model_sgd, test_model
from plottings import get_loss_surface_3Dplot, get_loss_surface_plot
from torch import optim
from matplotlib.pyplot import close

######################################################
###################### ADAM ##########################
######################################################
print('\nADAM')
model = SimpleNN()
model_copy = SimpleNN()
model_copy.load_state_dict(model.state_dict())

# Define the optimizer
optimizer = optim.Adam(model.parameters())
optimizer_copy = optim.Adam(model_copy.parameters())

# Train the model
inputs, labels = get_data_train()
inputs_length = inputs.shape[0]-inputs.shape[0]%(10*settings.batch_size)
inputs = inputs[:inputs_length]
labels = labels[:inputs_length]

losses = train_model_sgd(model, settings.num_epochs, settings.criterion, optimizer, inputs, labels, batch_size=settings.batch_size)

# Test the model
model.eval()
test_inputs, test_labels = get_data_test()
test_model(model, test_inputs, test_labels)
model.train()

# Plot3d mbatch loss
data = inputs[:1024]
target = labels[:1024]
train_size = int(settings.training_split * inputs_length)
trajectory = get_loss_surface_3Dplot(model, settings.criterion, data, target, losses=losses, initial_model=model_copy, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=settings.batch_size)
# Plot2d mbatch loss
get_loss_surface_plot(model, settings.criterion, data, target, losses=losses, optimizer=optimizer_copy, training_inputs=inputs[:train_size], labels=labels[:train_size], batch_size=settings.batch_size, trajectory=trajectory)
close('all')


