import torch
import matplotlib.pyplot as plt
import numpy as np


def filter_wise_normalization(d, theta):
    for param_d, param_theta in zip(d, theta):
        if len(param_d.shape) > 1:  # Apply normalization to filters
            norm_theta = param_theta.norm(dim=1, keepdim=True)
            norm_d = param_d.norm(dim=1, keepdim=True)
            param_d.data = param_d * (norm_theta / (norm_d + 1e-10))
    return d

def get_loss_surface_plot(model, criterion, data, target, n_points=40, epsilon=1, trajectory=None):
    # Generate random Gaussian direction vectors
    theta = [param.clone() for param in model.parameters()]
    d1 = [torch.randn_like(param) for param in model.parameters()]
    d2 = [torch.randn_like(param) for param in model.parameters()]
    
    # Filter-wise normalization
    d1 = filter_wise_normalization(d1, theta)
    d2 = filter_wise_normalization(d2, theta)
    
    # Create mesh grid
    alphas = np.linspace(-epsilon, epsilon, n_points)
    betas = np.linspace(-epsilon, epsilon, n_points)
    loss_surface = np.zeros((n_points, n_points))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            for param, d1_p, d2_p in zip(model.parameters(), d1, d2):
                param.data = param.data + alpha * d1_p + beta * d2_p
            
            # Calculate loss
            output = model(data)
            loss = criterion(output, target)
            loss_surface[i, j] = loss.item()
            
            # Reset parameters
            for param, theta_p in zip(model.parameters(), theta):
                param.data = theta_p.clone()
    
    # Plot loss surface
    X, Y = np.meshgrid(alphas, betas)
    plt.contourf(X, Y, loss_surface, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Loss Surface')
    
    if trajectory is not None:
        traj_alphas, traj_betas, _ = trajectory
        plt.plot(traj_alphas, traj_betas, 'r.-')  # Plot the trajectory
        
    plt.show()

def get_loss_surface_3Dplot(model, criterion, data, target, n_points=40, epsilon=1, trajectory=None):
    model.eval()
    # Generate random Gaussian direction vectors
    theta = [param.clone() for param in model.parameters()]
    d1 = [torch.randn_like(param) for param in model.parameters()]
    d2 = [torch.randn_like(param) for param in model.parameters()]

    # Filter-wise normalization
    d1 = filter_wise_normalization(d1, theta)
    d2 = filter_wise_normalization(d2, theta)

    # Create mesh grid
    alphas = np.linspace(-epsilon, epsilon, n_points)
    betas = np.linspace(-epsilon, epsilon, n_points)
    loss_surface = np.zeros((n_points, n_points))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            for param, d1_p, d2_p in zip(model.parameters(), d1, d2):
                param.data = param.data + alpha * d1_p + beta * d2_p
            
            # Calculate loss
            output = model(data)
            loss = criterion(output, target)
            loss_surface[i, j] = loss.item()
            
            # Reset parameters
            for param, theta_p in zip(model.parameters(), theta):
                param.data = theta_p.clone()
    # Plot loss surface
    X, Y = np.meshgrid(alphas, betas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Loss')
    plt.title('Loss Surface')
    if trajectory is not None:
        traj_alphas, traj_betas, real_traj_losses = trajectory
        # calculate the z value for each point of trajectory
        traj_losses = []
        approx_alphas = []
        approx_betas = []
        for i in range(len(traj_alphas)):
            alpha_index = np.searchsorted(alphas, traj_alphas[i])
            beta_index = np.searchsorted(betas, traj_betas[i])
            traj_losses.append(loss_surface[beta_index, alpha_index])
            approx_alphas.append(alphas[alpha_index])
            approx_betas.append(betas[beta_index])
        ax.plot3D(approx_alphas, approx_betas, traj_losses, 'r.-')
        ax.plot3D(traj_alphas, traj_betas, real_traj_losses, 'g.-')

        
    plt.show()

def get_deltas(model):
    theta = [param.clone() for param in model.parameters()]
    d1 = [torch.randn_like(param) for param in model.parameters()]
    d2 = [torch.randn_like(param) for param in model.parameters()]

    d1 = filter_wise_normalization(d1, theta)
    d2 = filter_wise_normalization(d2, theta)

    return d1, d2, theta

def next_trajectory_point(model, theta, d1, d2, loss):
    alpha = sum(torch.sum((param.data - theta_p) * d1_p).item() for param, theta_p, d1_p in zip(model.parameters(), theta, d1))
    beta = sum(torch.sum((param.data - theta_p) * d2_p).item() for param, theta_p, d2_p in zip(model.parameters(), theta, d2))
    return (alpha, beta, loss)

# def track_optimization_trajectory(model, optimizer, criterion, data, target, steps=50, lr=0.01):
#     trajectory = []
#     theta = [param.clone() for param in model.parameters()]
#     d1 = [torch.randn_like(param) for param in model.parameters()]
#     d2 = [torch.randn_like(param) for param in model.parameters()]
    
#     d1 = filter_wise_normalization(d1, theta)
#     d2 = filter_wise_normalization(d2, theta)
    
#     for _ in range(steps):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
        
#         alpha = sum(torch.sum((param.data - theta_p) * d1_p).item() for param, theta_p, d1_p in zip(model.parameters(), theta, d1))
#         beta = sum(torch.sum((param.data - theta_p) * d2_p).item() for param, theta_p, d2_p in zip(model.parameters(), theta, d2))
#         trajectory.append((alpha, beta))
    
#     traj_alphas, traj_betas = zip(*trajectory)
#     return traj_alphas, traj_betas
# # Example usage
# model = SimpleCNN()
# criterion = nn.CrossEntropyLoss()
# data = torch.randn(1, 1, 32, 32)  # Example input
# print(data.shape)
# target = torch.tensor([3])  # Example target

# get_loss_surface_plot(model, criterion, data, target)
# get_loss_surface_3Dplot(model, criterion, data, target)

