import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(500, 50)
		self.fc2 = nn.Linear(50, 50)
		self.fc3 = nn.Linear(50, 10) 

	def forward(self, x):
		x = torch.relu(F.max_pool2d(self.conv1(x), 2))
		x = torch.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 500)
		x = torch.relu(self.fc1(x))
		x = torch.relu(x) + torch.relu(self.fc2(x))  
		x = self.fc3(x)  
		return x


model = SimpleCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
print(model._get_name())
def format_optim_str(optimizer):
	return str(optimizer.param_groups[0]).split('), \'')[1].replace('\'', '').replace(' ', '').replace(':','-').replace(',', '_').replace('{','').replace('}','').replace('[','').replace(']','').replace('(','').replace(')','').split('_nesterov')[0]
