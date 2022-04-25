import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tqdm
import pandas as pd

"""
dist_q=torch.distributions.normal.Normal(torch.tensor([-5.0]), torch.tensor([1]))

mix=torch.distributions.categorical.Categorical(torch.tensor([0.3, 0.7]))
comp=torch.distributions.normal.Normal(torch.tensor([-2.5, 1.5]), torch.tensor([1, 1]))    
dist_p=torch.distributions.mixture_same_family.MixtureSameFamily(mix,comp)
"""
dist_q = torch.distributions.normal.Normal(torch.tensor([-10.0]), torch.tensor([1]))

mix = torch.distributions.categorical.Categorical(torch.tensor([1 / 3, 0.7]))
comp = torch.distributions.normal.Normal(torch.tensor([-2, 2]), torch.tensor([1, 1]))
dist_p = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

# torch.tensor(np.linspace(-5,5,100))
# torch.exp(dist_q.log_prob(torch.tensor(np.linspace(-5,5,100))))
data = torch.vstack(
    (
        torch.tensor(np.linspace(-10, 10, 1000)),
        torch.exp(dist_q.log_prob(torch.tensor(np.linspace(-10, 10, 1000)))),
    )
)
data = torch.vstack(
    (data, torch.exp(dist_p.log_prob(torch.tensor(np.linspace(-10, 10, 1000)))))
)
data = {"x": data[0, :], "q": data[1, :], "p": data[2, :]}
data_frame = pd.DataFrame(data)

n = 100
# epsilon=0.01
alpha = 1

x = dist_q.sample(sample_shape=torch.Size([n]))

plt.plot(data_frame["x"], data_frame["p"], color="#6000FF")
plt.plot(data_frame["x"], data_frame["q"], color="#FF3000")
sns.kdeplot(data=pd.DataFrame({"particles": x[:, 0]}), color="green")
plt.legend(["p", "q", "particles"])
plt.show()

n_iterations = 500


def step_size(iteration):
    # return 0.25
    return 0.001 * 50 ** (-iteration / n_iterations + 1)


phinorms = np.zeros((n_iterations, 1))

for l in tqdm.tqdm(range(n_iterations)):
    t = time.time()
    # x_ntimes[i,j]=x_j
    x_ntimes = torch.matmul(torch.ones([n])[:, None], torch.transpose(x, 0, 1))
    # x_difference[i,j]=x_j-x_i
    x_difference = x_ntimes - torch.transpose(x_ntimes, 0, 1)
    # h=median(||x_i-x_j||_2)^2/log(n)
    h = float(
        torch.median(
            torch.sort(
                torch.flatten(torch.flatten(torch.triu(torch.abs(x_difference))))
            )[0][int((n**2 + n) / 2) :]
        )
    ) ** 2 / np.log(n)
    # k[i,j]=k(x_j,x_i)
    k = torch.exp(-1.0 / h * torch.pow(x_difference, 2))
    # grad_k[i,j]=\nabla_x_j k(x_j,x_i)
    grad_k = -1.0 / h * 2.0 * x_difference * k
    # p[i,j]=p(x_j)
    p = torch.exp(dist_p.log_prob(x_ntimes))
    # q[i,j]=q(x_j)
    q = torch.exp(dist_q.log_prob(x_ntimes))
    # grad_p[i,j]=\nabla_x p(x_j)
    grad_p = (
        0.3
        / (np.sqrt(2 * np.pi))
        * (-x_ntimes - 2.5)
        * torch.exp(-torch.pow(x_ntimes + 2.5, 2) / 2)
        + 0.7
        / (np.sqrt(2 * np.pi))
        * (-x_ntimes + 1.5)
        * torch.exp(-torch.pow(x_ntimes - 1.5, 2) / 2)
    ) * torch.pow(p, -1)
    phi_summand = (
        1
        / n
        * (grad_p * k + grad_k)
        * torch.pow(q, alpha - 1)
        * torch.pow(p, 1 - alpha)
    )
    # phi
    phi = torch.sum(phi_summand, dim=1)[:, None]
    epsilon = step_size(l)
    x = x + epsilon * phi
    # tqdm.tqdm.write("Minimum of p: "+str(float(torch.min(p))))
    # tqdm.tqdm.write("Minimum of p: "+str(dist_p.log_prob(x_ntimes)))
    tqdm.tqdm.write("Adagrad: " + str(float(torch.optim.Adagrad(torch.norm(phi)))))
    tqdm.tqdm.write("Norm of phi: " + str(float(torch.norm(phi))))
    tqdm.tqdm.write("h: " + str(h))
    phinorms[l] = float(torch.norm(phi))

plt.plot(data_frame["x"], data_frame["p"], color="#6000FF")
plt.plot(data_frame["x"], data_frame["q"], color="#FF3000")
sns.kdeplot(data=pd.DataFrame({"particles": x[:, 0]}), color="#00FFFF")
plt.legend(["p", "q", "particles"])
plt.show()

plt.plot(range(n_iterations), phinorms, color="#6000FF")
plt.xlabel("Iteration")
plt.ylabel("Norm of Phi")
plt.show()

# Example: mean 0, sd 2, alpha 1.6, epsilon 1
# Example: mean 0, sd 3, alpha 1.3, epsilon 1
# Example: mean 0, sd 3, alpha 1.4, epsilon 0.25
# Example: mean 0, sd 3, alpha 1.4, epsilon 0.001, 5000 iterations
