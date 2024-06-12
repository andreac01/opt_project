import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

def save_plot(fig, svg_filename='plot.svg', pickle_filename='plot.html'):
    # save svg
    fig.savefig(svg_filename, format='svg')
    # save pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(fig, f)

def format_optim_str(optimizer):
	return str(optimizer.param_groups[0]).split(')], \'')[1].replace('\'', '').replace(' ', '').replace(':','-').replace(',', '_').replace('{','').replace('}','').replace('[','').replace(']','').replace('(','').replace(')','').split('_nesterov')[0]

def filter_wise_normalization(d, theta):
    for param_d, param_theta in zip(d, theta):
        if len(param_d.shape) > 1:  # Apply normalization to filters
            norm_theta = param_theta.norm(dim=1, keepdim=True)
            norm_d = param_d.norm(dim=1, keepdim=True)
            param_d.data = param_d * (norm_theta / (norm_d + 1e-10))
    return d

def get_loss_surface_plot(model, criterion, data, target, n_points=40, epsilon=1, losses=None, initial_model=None, optimizer=None, training_inputs=None, labels=None, trajectory=None, batch_size=64):
    model.eval()
    # Generate direction vectors
    theta, d1, d2 = get_deltas(model)

    if trajectory is None and losses is not None and initial_model is not None and optimizer is not None:
        trajectory = compute_whole_trajectory(initial_model, optimizer, criterion, training_inputs, labels, theta=theta, d1=d1, d2=d2, steps=len(losses), losses=losses, batch_size=batch_size)
        traj_alphas, traj_betas, real_traj_losses = zip(*trajectory)
        trajectory = (traj_alphas, traj_betas, real_traj_losses)
        print('TRAJECTORY COMPUTED')
        print(len(trajectory[0]), len(trajectory[1]), len(trajectory[2]), len(losses))
    
    if trajectory is not None:
        epsilon = 1.2*max(np.max(np.abs(trajectory[0])), np.max(np.abs(trajectory[1])))

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
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, loss_surface, levels=50, cmap='viridis')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_title('Loss Surface')
    
    if trajectory is not None:
        traj_alphas, traj_betas, real_traj_losses = trajectory
        # calculate the z value for each point of trajectory
        approx_alphas = []
        approx_betas = []
        for i in range(len(traj_alphas)):
            alpha_index = np.searchsorted(alphas, traj_alphas[i])
            beta_index = np.searchsorted(betas, traj_betas[i])
            approx_alphas.append(alphas[alpha_index])
            approx_betas.append(betas[beta_index])
        ax.plot(traj_alphas, traj_betas, 'r.-')
    name = './plots/' + model._get_name() + format_optim_str(optimizer=optimizer) + '_bs-' + str(batch_size)
    save_plot(fig, svg_filename=name+'.svg', pickle_filename=name+'.fig.pickle')

    return trajectory

def compute_whole_trajectory(model, optimizer, criterion, data, target, batch_size=64, steps=1000, theta=None, d1=None, d2=None, losses=None):
    model.train()
    trajectory = []
    for i in range(steps):
        idx = (i*batch_size)%(len(data))
        optimizer.zero_grad()
        output = model(data[idx:idx+batch_size])
        loss = criterion(output, target[idx:idx+batch_size])
        #print(loss.item(), output, target[idx:idx+batch_size])
        if losses is not None:
            if losses[i].item() != loss.item():
                print('LOSS MISMATCH', losses[i].item(), loss.item(), 'At', i, idx)
                print(len(data), len(target), len(losses))
                break
        loss.backward()
        optimizer.step()
        trajectory.append(next_trajectory_point(model, theta, d1, d2, loss.item()))
    return trajectory

def get_loss_surface_3Dplot(model, criterion, data, target, n_points=40, epsilon=1, losses=None, initial_model=None, optimizer=None, training_inputs=None, labels=None, trajectory=None, batch_size=64):
    model.eval()
    # Generate direction vectors
    theta, d1, d2 = get_deltas(model)

    if trajectory is None and losses is not None and initial_model is not None and optimizer is not None:
        trajectory = compute_whole_trajectory(initial_model, optimizer, criterion, training_inputs, labels, theta=theta, d1=d1, d2=d2, steps=len(losses), losses=losses, batch_size=batch_size)
        traj_alphas, traj_betas, real_traj_losses = zip(*trajectory)
        trajectory = (traj_alphas, traj_betas, real_traj_losses)
        print('TRAJECTORY COMPUTED')
        print(len(trajectory[0]), len(trajectory[1]), len(trajectory[2]), len(losses))
    
    if trajectory is not None:
        epsilon = 1.2*max(np.max(np.abs(trajectory[0])), np.max(np.abs(trajectory[1])))

    print('test ', next_trajectory_point(model, theta, d1, d2, 0))

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
    ax.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Loss')
    plt.title('Loss Surface')

    if trajectory is not None:
        traj_alphas, traj_betas, real_traj_losses = trajectory
        # calculate the z value for each point of trajectory
        approx_losses = []
        approx_alphas = []
        approx_betas = []
        for i in range(len(traj_alphas)):
            alpha_index = np.searchsorted(alphas, traj_alphas[i])
            beta_index = np.searchsorted(betas, traj_betas[i])
            approx_losses.append(loss_surface[beta_index, alpha_index])
            approx_alphas.append(alphas[alpha_index])
            approx_betas.append(betas[beta_index])
        ax.plot3D(approx_alphas, approx_betas, approx_losses, 'r.-')
    name = './plots/' + model._get_name() + format_optim_str(optimizer=optimizer) + '_bs-' + str(batch_size) + '_3D'
    save_plot(fig, svg_filename=name+'.svg', pickle_filename=name+'.fig.pickle')
        
    return trajectory

def get_deltas(model):
    # Generate random Gaussian direction vectors
    theta = [param.clone() for param in model.parameters()]
    d1 = [torch.randn_like(param) for param in model.parameters()]
    d2 = [torch.randn_like(param) for param in model.parameters()]
    # Filter-wise normalization
    d1 = filter_wise_normalization(d1, theta)
    d2 = filter_wise_normalization(d2, theta)

    return theta, d1, d2

def next_trajectory_point(model, theta, d1, d2, losses):
    alpha = sum(torch.sum((param.data - theta_p) * d1_p).item() for param, theta_p, d1_p in zip(model.parameters(), theta, d1))
    beta = sum(torch.sum((param.data - theta_p) * d2_p).item() for param, theta_p, d2_p in zip(model.parameters(), theta, d2))
    return (alpha, beta, losses)


