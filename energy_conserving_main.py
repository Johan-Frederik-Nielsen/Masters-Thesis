import numpy as np
import torch
from numpyro.examples.datasets import HIGGS, load_dataset
import tqdm
from pyro.distributions import Normal, MultivariateNormal
import pandas as pd
import scipy

torch.manual_seed(0)

verbose = False
debug = False


def load_data():
    n_data = 1000
    xs = []
    ys = []

    for split in ["train", "test"]:
        _, fetch = load_dataset(
            HIGGS, shuffle=False, num_datapoints=n_data, split=split
        )
        x, y = map(lambda data: torch.tensor(np.array(data)), fetch())
        xs.append(x)
        ys.append(y)
    
    return torch.concat(xs, 0), torch.concat(ys, 0)

def kernel_value_grad(particles):
    diff = particles[None, :] - particles[:, None]
    sq_dist = torch.linalg.norm(diff, dim=-1)
    h = torch.median(sq_dist) ** 2 / np.log(n_particles)
    value = torch.exp(-1.0 / h * sq_dist**2)

    grad = -2.0 / h * diff * value[..., None]
    return h, value.sum(1), grad.sum(1)


def step(particles, state_sum):
    h, k, grad_k = kernel_value_grad(particles)
    
    # Size of the following 3 tensors are all [n_data,28,n_particles]
    exp_mat = torch.exp(torch.matmul(data, particles.T))[:,None,:].repeat(1,n_dims,1)
    x_k=data[:,:,None].repeat(1,1,n_particles)
    y_k=obs[:,None,None].repeat(1,n_dims,n_particles)
    

    grad_p=-y_k*(-x_k*torch.pow(exp_mat,-1))/(1+torch.pow(exp_mat,-1))+(y_k-1)*(x_k*torch.pow(exp_mat,1))/(1+torch.pow(exp_mat,1))

    grad_p=torch.sum(grad_p,0)-particles
    grad_p=torch.transpose(grad_p,0,1)
    
    alpha=1.5

    logqp=-y_k*torch.log(1+torch.pow(exp_mat,-1))+(y_k-1)*torch.log(1+exp_mat)
    logqp=-torch.sum(logqp,0)+torch.transpose(particles,0,1)
    logqp=torch.transpose(logqp,0,1)
    #print(logqp)

    phi = 1 / n_particles * (grad_p * k[:, None] + grad_k)*torch.pow(torch.abs(logqp),alpha-1)/torch.mean(torch.pow(torch.abs(logqp),alpha-1))
    state_sum = state_sum + phi**2

    epsilon = 0.00001 * phi * state_sum**0.5

    particles = particles + epsilon
    return particles, state_sum

if __name__ == "__main__":
    n_particles = 100
    
    data, obs = load_data()
    n_data, n_feats = data.shape
    n_dims=28
    n_feats=n_dims
    data=data[:,0:n_dims]
    print("HERE: ",torch.Tensor.size(data))
    #print(torch.Tensor.size(data),torch.Tensor.size(obs))
    dist_q = Normal(0, 1.0).expand((n_feats,)).to_event(1)

    particles = dist_q.sample((n_particles,))

    log_like = -torch.log1p(torch.matmul(particles, -data.T).exp()) * obs - torch.log1p(
        torch.matmul(particles, data.T).exp()
    ) * (1 - obs)

    log_joint = torch.sum(log_like, dim=1) + dist_q.log_prob(particles)

    state_sum = torch.zeros((n_particles, 1))
    #print(particles)
    n_iterations = 250
    if debug:
        old_particles, old_state_sum = old_step(particles, state_sum)
    particles, state_sum = step(particles, state_sum)
    if debug:
        assert torch.allclose(state_sum, old_state_sum)
    #kl_init = kl(particles, data, obs, dist_q)
    for l in tqdm.tqdm(range(n_iterations - 1)):
        particles, state_sum = step(particles, state_sum)
        exp_mat = torch.exp(torch.matmul(data, particles.T))[:,None,:].repeat(1,n_dims,1)
        x_k=data[:,:,None].repeat(1,1,n_particles)
        y_k=obs[:,None,None].repeat(1,n_dims,n_particles)
        grad_p=-y_k*(-x_k*torch.pow(exp_mat,-1))/(1+torch.pow(exp_mat,-1))+(y_k-1)*(x_k*torch.pow(exp_mat,1))/(1+torch.pow(exp_mat,1))        

    #kl_final = kl(particles, data, obs, dist_q)
    kde=scipy.stats.gaussian_kde(np.array(torch.transpose(particles,0,1)))
    q=kde.pdf(torch.transpose(particles,0,1))
    print("KL estimate: ",np.sum(q*(np.log(q)-np.array(grad_p))))

