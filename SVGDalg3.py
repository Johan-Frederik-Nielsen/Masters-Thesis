import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tqdm

n = 31
interval_size = 3
epsilon = 0.1
alpha = 0.99

x = torch.empty((2, n * n))
for i in range(n):
    for j in range(n):
        x[0, i * n + j] = (i - np.floor(n / 2)) / n * 2 * interval_size
        x[1, i * n + j] = (j - np.floor(n / 2)) / n * 2 * interval_size

n = n**2

for l in tqdm.tqdm(range(250)):
    t = time.time()
    # x_ntimes[:,i,j]=x_i
    x_ntimes = torch.matmul(torch.ones(2, n, 1), x[:, None])
    # x_difference[:,i,j]=x_i-x_j
    x_difference = x_ntimes - torch.transpose(x_ntimes, 1, 2)
    # h=median(||x_i-x_j||_2)^2/log(n)
    h = float(
        torch.pow(
            torch.median(
                torch.sort(
                    torch.flatten(
                        torch.triu(
                            torch.pow(torch.sum(torch.pow(x_difference, 2), dim=0), 0.5)
                        )
                    )
                )[0][int((n**2 + n) / 2) :]
            ),
            2,
        )
        / np.log(n)
    )
    # k[i,j]=k(x_i,x_j)
    k = torch.exp(-1.0 / h * torch.sum(torch.pow(x_difference, 2), 0))[None, :, :]
    # grad_k=\nabla_x k
    grad_k = k * (-1.0 / h * 2.0 * x_difference)
    # grad_p=\nabla_x p
    grad_p = torch.exp(torch.sum((-0.5 * torch.pow(x_ntimes, 2)), 0)) / (2 * np.pi)
    grad_p = torch.mul(-x_ntimes, torch.stack([grad_p, grad_p], dim=0))
    # p[i,j]=p(x_i,x_j)
    p = torch.exp(torch.sum((-0.5 * torch.pow(x_ntimes, 2)), 0)) / (2 * np.pi)
    # p=torch.transpose(p,0,1)
    grad_p = torch.transpose(grad_p, 1, 2)
    # k=torch.transpose(k,1,2)
    # grad_k=torch.transpose(grad_k,1,2)
    # print(torch.Tensor.size(grad_k))
    # phi_summand[:,i,j]=j'th summand of update
    phi_summand = (
        1
        / n
        * (grad_p * k + p * grad_k)
        * torch.pow(interval_size**2 * p, alpha - 1)
        / (1 / interval_size**2)
    )
    # phi
    phi = torch.sum(phi_summand, 2)
    x = x + epsilon * phi
    # print(l,time.time()-t)
    # print("h =",h)
    # print("||grad_p|| =",float(torch.norm(grad_p)))
    # print("MINIMUM",torch.min(p))
    # print("||grad_p*k|| =",float(torch.norm(grad_p*k)))
    # print("||p*grad_k|| =",float(torch.norm(p*grad_k)))
    # print("IMPORTANT NUMBER =", float(torch.norm(grad_p*k))/float(torch.norm(p*grad_k)))
    # print("Norm of phi: ",float(torch.norm(phi)))
    # print(torch.Tensor.size(grad_p))
    tqdm.tqdm.write("Norm of phi: " + str(float(torch.norm(phi))))
x1 = x[0, :]
y1 = x[1, :]
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x1, y=y1, s=5, color="#6000FF")
# sns.histplot(x=x1, y=y1, bins=50, pthresh=.1, cmap="mako")
# sns.kdeplot(x=x1, y=y1, levels=5, color="w", linewidths=1)
plt.show()
