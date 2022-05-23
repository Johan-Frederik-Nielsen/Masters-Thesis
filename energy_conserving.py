import numpy as np
import torch
from numpyro.examples.datasets import HIGGS, load_dataset
import tqdm
from pyro.distributions import Normal, MultivariateNormal

torch.manual_seed(0)

verbose = False


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
    a_sqr = (particles ** 2).sum(1)
    diff = torch.matmul(particles, particles.T)
    sq_dist = a_sqr[None, :] + a_sqr[:, None] - 2 * diff
    h = torch.median(sq_dist) / np.log(n_particles)
    value = torch.exp(-1.0 / h * sq_dist)

    grad = -2.0 / h * (particles[None, :] - particles[:, None]) * value[..., None]
    return value, grad


def old_step(theta, state_sum):
    # theta_ntimes[i,j,:]=theta[j,:]
    theta_ntimes = theta[None, :, :].repeat(n_particles, 1, 1)
    # theta_difference[i,j,:]=theta[j,:]-theta[i,:]
    theta_difference = (theta_ntimes - torch.transpose(theta_ntimes, 0, 1))
    # h
    h = float(torch.median(torch.sort(
        torch.flatten(torch.flatten(torch.triu(torch.pow(torch.sum(torch.pow(theta_difference, 2), 2), 0.5)))))[0][
                           int((n_particles ** 2 + n_particles) / 2):])) ** 2 / np.log(n_particles)
    # k[i,j]=k(x_j,x_i)
    k = torch.exp(-1.0 / h * torch.sum(torch.pow(theta_difference, 2), 2))
    # grad_k[i,j,:]=\nabla_x_j k(x_j,x_i)
    grad_k = (-1.0 / h * 2.0 * theta_difference * k[:, :, None])
    # grad_p1=-theta
    grad_p1 = (data[:, :, None].repeat(1, 1, 50) * torch.transpose(
        (torch.exp(-torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0, 2).repeat(1, 28, 1)) / (
                  torch.transpose((1 + torch.exp(-torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0,
                                  2).repeat(1, 28, 1))
    grad_p0 = -(data[:, :, None].repeat(1, 1, 50) * torch.transpose(
        (torch.exp(torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0, 2).repeat(1, 28, 1)) / (
                  torch.transpose((1 + torch.exp(torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0,
                                  2).repeat(1, 28, 1))
    # print("Here: ",torch.Tensor.size(data[:,:,None].repeat(1,1,50)),torch.Tensor.size(torch.transpose((1+torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1)))
    grad_p = grad_p0 * (1 - obs[:, None, None]).repeat(1, 28, n_particles) + grad_p1 * obs[:, None, None].repeat(1, 28,
                                                                                                                 n_particles)
    grad_p = torch.sum(grad_p, 0) - torch.transpose(theta, 0, 1)
    # print("Here: ",torch.Tensor.size(grad_p),torch.Tensor.size(theta),torch.Tensor.size(k),torch.Tensor.size(grad_k))
    phi_summand = 1 / n_particles * (
            torch.transpose(grad_p, 0, 1)[:, None, :].repeat(1, n_particles, 1) * k[:, :, None].repeat(1, 1,
                                                                                                       28) + grad_k)
    phi = torch.sum(phi_summand, 1)
    # print(torch.Tensor.size(phi),torch.Tensor.size(theta))
    state_sum = state_sum + torch.pow(phi, 2)
    epsilon = 0.00001 * phi * torch.pow(state_sum, 0.5)
    test_tensor = (data[:, :, None].repeat(1, 1, 50) * torch.transpose(
        (torch.exp(torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0, 2).repeat(1, 28, 1))
    if torch.max(test_tensor != test_tensor) == 1:
        print("Error in iteration", l)
        print(torch.Tensor.size(
            torch.transpose((torch.exp(torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0, 2).repeat(1,
                                                                                                                    28,
                                                                                                                    1)),
            torch.Tensor.size(data[:, :, None].repeat(1, 1, 50)), torch.Tensor.size(-(
                    data[:, :, None].repeat(1, 1, 50) * torch.transpose(
                (torch.exp(torch.matmul(theta, torch.transpose(data, 0, 1))))[:, None, :], 0, 2).repeat(1, 28, 1))))
        # print(torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))
        print(torch.max(torch.matmul(theta, torch.transpose(data, 0, 1))))
        # print(test_tensor)
    theta += epsilon
    return theta, state_sum


def step(particles, state_sum):
    k, grad_k = kernel_value_grad(particles)

    # grad_p1=-theta
    grad_p0 = (1 / (1 + torch.exp(-torch.matmul(data, particles.T))))

    # grad_p0 + grad_p1 = 1 => obs * (grad_p0 + grad_p1) = obs
    grad_p = data[:, None, :] * (grad_p0 - obs[:, None])[..., None]
    grad_p = torch.sum(grad_p, 0) - particles

    phi_summand = (1 / n_particles * (grad_p[:, None, :] * k[:, :, None] + grad_k))
    phi = torch.sum(phi_summand, 1)

    # print(torch.Tensor.size(phi),torch.Tensor.size(theta))
    state_sum = state_sum + torch.pow(phi, 2)
    epsilon = 0.00001 * phi * torch.pow(state_sum, 0.5)

    if torch.any(torch.isnan(torch.exp(torch.matmul(particles, torch.transpose(data, 0, 1))))):
        raise ValueError('NaN in particles.')
    particles += epsilon
    return particles, state_sum


def kl(particles, data, obs, dist_q):
    p_current = torch.pow(
        1 + torch.exp(-torch.matmul(particles, torch.transpose(data, 0, 1))),
        -(obs[None, :].repeat(50, 1)),
    ) * torch.pow(
        1 + torch.exp(torch.matmul(particles, torch.transpose(data, 0, 1))),
        (obs[None, :].repeat(50, 1)) - 1,
    )

    p_current = torch.prod(p_current, 1) * torch.exp(dist_q.log_prob(particles))

    kl = torch.sum(
        torch.exp(dist_q.log_prob(particles))
        * torch.log(torch.exp(dist_q.log_prob(particles)) / p_current),
        0,
    )
    return kl


if __name__ == "__main__":
    n_particles = 50

    data, obs = load_data()
    n_data, n_feats = data.shape

    dist_q = Normal(0, 1.).expand((n_feats,)).to_event(1)

    particles = dist_q.sample((n_particles,))

    log_like = -torch.log1p(torch.matmul(particles, -data.T).exp()) * obs - torch.log1p(
        torch.matmul(particles, data.T).exp()) * (1 - obs)

    log_joint = torch.sum(log_like, dim=1) + dist_q.log_prob(particles)

    state_sum = torch.zeros((n_particles, 1))

    n_iterations = 100
    kl_init = kl(particles, data, obs, dist_q)
    for l in tqdm.tqdm(range(n_iterations)):
        old_particles, old_state_sum = old_step(particles, state_sum)
        particles, state_sum = step(particles, state_sum)
        assert torch.allclose(old_particles, particles)  # for testing

    kl_final = kl(particles, data, obs, dist_q)
    print(kl_final)
