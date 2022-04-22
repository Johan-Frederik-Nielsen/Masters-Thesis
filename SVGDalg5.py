import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tqdm

n=31
interval_size=5
epsilon=1
alpha=1

x=torch.empty((2,n*n))
for i in range(n):
    for j in range(n):
        x[0,i*n+j]=(i-np.floor(n/2))/n*2*interval_size
        x[1,i*n+j]=(j-np.floor(n/2))/n*2*interval_size

n=n**2

x1=x[0,:]
y1=x[1,:]
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x1, y=y1, s=5, color="#6000FF")
#sns.histplot(x=x1, y=y1, bins=50, pthresh=.1, cmap="mako")
#sns.kdeplot(x=x1, y=y1, levels=5, color="w", linewidths=1)
plt.show()

m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
#print(torch.exp(m.log_prob(torch.tensor([3,3]))))

for l in tqdm.tqdm(range(250)):
    t=time.time()
    #x_ntimes[:,i,j]=x_j
    x_ntimes=torch.matmul(torch.ones(2,n,1),x[:,None])
    #x_difference[:,i,j]=x_j-x_i
    x_difference=(x_ntimes-torch.transpose(x_ntimes,1,2))
    #h=median(||x_i-x_j||_2)^2/log(n)
    #h=float(torch.pow(torch.median(torch.sort(torch.flatten(torch.triu(torch.pow(torch.sum(torch.pow(x_difference,2),dim=0),0.5))))[0][int((n**2+n)/2):]),2)/np.log(n))
    h=10
    #k[:,i,j]=k(x_j,x_i)
    k=torch.exp(-1.0/h*torch.sum(torch.pow(x_difference,2),0))
    #grad_k[:,i,j]=\nabla_x_j k(x_j,x_i)
    grad_k=(-1.0/h*2.0*x_difference*k)
    #p[i,j]=p(x_j)
    p=torch.exp(-0.5*torch.sum(torch.pow(x_ntimes,2),0))/(2*np.pi)
    #grad_p[:,i,j]=\nabla_x p(x_j)
    #grad_p=-x_ntimes*torch.stack([p,p],dim=0)
    #grad_p=-x_ntimes*p
    grad_p=-x_ntimes
    #p=torch.transpose(p,0,1)
    #grad_p=torch.transpose(grad_p,1,2)
    #k=torch.transpose(k,1,2)
    #grad_k=torch.transpose(grad_k,1,2)
    #phi_summand[:,i,j]=j'th summand of update
    #k=torch.transpose(k,0,1)
    #grad_k=torch.transpose(grad_k,1,2)
    #phi_summand=1/n*(grad_p*torch.stack([k,k])+torch.stack([p,p])*grad_k)*torch.pow(interval_size**2*torch.stack([p,p]),alpha-1)/(1/interval_size**2)
    #phi_summand=1/n*(grad_p*k+p*grad_k)*torch.pow(interval_size**2*p,alpha-1)/(1/interval_size**2)
    phi_summand=1/n*(grad_p*k+grad_k)*torch.pow(interval_size**(-2)*torch.pow(p,-1),alpha-1)
    #phi
    phi=torch.sum(phi_summand,dim=2)
    """
    for i in range(n):
        for j in range(n):
            #tqdm.tqdm.write(str(x[:,j]-x_ntimes[:,i,j]))
            #tqdm.tqdm.write(str(x[:,j]-x[:,i]-x_difference[:,i,j]))
            #tqdm.tqdm.write(str((p[i,j]-torch.exp(m.log_prob(x[:,j])))/torch.exp(m.log_prob(x[:,j]))))
            #tqdm.tqdm.write(str(k[:,i,j]-torch.exp(-1.0/h*torch.sum(torch.pow(x[:,j]-x[:,i],2),dim=0))))
            #tqdm.tqdm.write(str(grad_k[:,i,j]-(-1.0/h*2.0*(x[:,j]-x[:,i])*torch.exp(-1.0/h*torch.sum(torch.pow(x[:,j]-x[:,i],2),dim=0)))))
            #tqdm.tqdm.write(str(k[i,j]*x_difference[:,i,j]-(x_difference*k)[:,i,j]))
            #tqdm.tqdm.write(str(-x[:,j]*p[i,j]-grad_p[:,i,j]))
            #tqdm.tqdm.write(str(p[i,j])+str(torch.exp(m.log_prob(x[:,j])))+str(x[:,j]))
    """
    """
    h_list=np.array([])
    for i in range(n):
        for j in range(i+1,n):
            h_list=np.append(h_list,((x[0,i]-x[0,j])**2+(x[1,i]-x[1,j])**2)**0.5)
            #print(i,j)
        if(i%25==0):
            print(i)
    print(np.median(h_list)**2/np.log(n)-h)
    #tqdm.tqdm.write("HERE: "+str(np.median(h_list)**2/np.log*(n)))
    """
    x=x+epsilon*phi
    """
    print(l,time.time()-t)
    print("||grad_p|| =",float(torch.norm(grad_p)))
    print("MINIMUM",torch.min(p))
    print("||grad_p*k|| =",float(torch.norm(grad_p*k)))
    print("||p*grad_k|| =",float(torch.norm(p*grad_k)))
    print("IMPORTANT NUMBER =", float(torch.norm(grad_p*k))/float(torch.norm(p*grad_k)))
    print("Norm of phi: ",float(torch.norm(phi)))
    print(torch.Tensor.size(grad_p))
    tqdm.tqdm.write("||torch.stack([p,p])*grad_k|| ="+str(float(torch.norm(torch.stack([p,p])*grad_k))))
    tqdm.tqdm.write("||grad_p*k|| ="+str(float(torch.norm(grad_p*k))))
    tqdm.tqdm.write("Norm of phi: "+str(float(torch.norm(phi))))
    """
    tqdm.tqdm.write("Norm of phi: "+str(float(torch.norm(phi))))
    tqdm.tqdm.write("h: "+str(h))
x1=x[0,:]
y1=x[1,:]
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x1, y=y1, s=5, color="#6000FF")
#sns.histplot(x=x1, y=y1, bins=50, pthresh=.1, cmap="mako")
#sns.kdeplot(x=x1, y=y1, levels=5, color="w", linewidths=1)
plt.show()
