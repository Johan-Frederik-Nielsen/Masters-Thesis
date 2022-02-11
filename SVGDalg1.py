import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

n=111
x=torch.empty((2,n*n))
for i in range(n):
    for j in range(n):
        x[0,i*n+j]=(i-np.floor(n/2))/n*2*5
        x[1,i*n+j]=(j-np.floor(n/2))/n*2*5

h=10.0
epsilon=1.0

for l in range(5):
    t=time.time()
    x2=(torch.matmul(torch.ones(2,n**2,1),x[:,None]))
    x3=(x2-torch.transpose(x2,1,2))
    #x3[:,i,j]=xj-xi
    x4=torch.exp(-1.0/h*torch.sum(torch.pow(x3,2),0))[None,:,:]
    #x4[i,j]=kernel(xj,xi)
    x5=x4*(-1.0/h*2.0*x3)
    #x5[i,j]=k_grad(xj,xi)
    x6=-x2
    #x6[:,i,j]=logp_grad(xj)
    x7=1/n**2*x4*x6+1/n**2*x5
    #x7[:,i,j]=Phi(xi,xj)
    x8=torch.sum(x7,2)
    #x8[:,i]=Phi(xi)
    x=x+epsilon*x8
    print(l,time.time()-t)
    
x1=x[0,:]
y1=x[1,:]
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x1, y=y1, s=5, color=".15")
sns.histplot(x=x1, y=y1, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x1, y=y1, levels=5, color="w", linewidths=1)
plt.show()
