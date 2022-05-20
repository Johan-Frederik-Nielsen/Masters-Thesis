import numpy as np
import torch
from numpyro.examples.datasets import HIGGS, load_dataset
import tqdm

torch.manual_seed(0)
n_data=1000

_, fetch = load_dataset(
            HIGGS, shuffle=False, num_datapoints=n_data
        )
data, obs = fetch()

data=torch.tensor(np.array(data))
obs=torch.tensor(np.array(obs))


print(torch.Tensor.size(data))
print(torch.Tensor.size(obs))
#print(obs)

n_data_true=torch.Tensor.size(data)[1]
dist_q=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(28), torch.eye(28))

n_particles=50
theta=dist_q.sample(sample_shape=torch.Size([n_particles]))
log_ps=torch.log(torch.pow((1/(1+torch.exp(-torch.matmul(theta,data[:,:].T)))),obs)*torch.pow((1/(1+torch.exp(torch.matmul(theta,data[:,:].T)))),1-obs))


"""
theta=torch.tensor(torch.ones(1,28),requires_grad=True)
log_ps_grad_func=torch.tensor(torch.sum(torch.log(torch.pow((1/(1+torch.exp(-torch.matmul(theta,data[:,:].T)))),obs)*torch.pow((1/(1+torch.exp(torch.matmul(theta,data[:,:].T)))),1-obs)),1)+dist_q.log_prob(theta),requires_grad=True)


print("Here: ",torch.Tensor.size(log_ps_grad_func))
external_grad=torch.ones(1,28)
#log_ps_grad_func.backward(gradient=external_grad)
log_ps_grad_func.backward(gradient=external_grad)
print("POTENTIAL GRAD: ",log_ps_grad_func.grad)
print(torch.Tensor.size(log_ps_grad_func))

#log_ps_grad_func.backward(gradient=external_grad)

print("Check: ",torch.Tensor.size(log_ps))

"""

log_p=torch.sum(log_ps,dim=1)+dist_q.log_prob(theta)

print(torch.Tensor.size(theta))

print("Here:" ,torch.Tensor.size(dist_q.log_prob(theta)),torch.Tensor.size(torch.matmul(theta,torch.transpose(data,0,1))),torch.Tensor.size(obs[None,:].repeat(50,1)))

p_current=torch.pow(1+torch.exp(-torch.matmul(theta,torch.transpose(data,0,1))),-(obs[None,:].repeat(50,1)))*torch.pow(1+torch.exp(torch.matmul(theta,torch.transpose(data,0,1))),(obs[None,:].repeat(50,1))-1)
p_current=torch.prod(p_current,1)*torch.exp(dist_q.log_prob(theta))
KL=torch.sum(torch.exp(dist_q.log_prob(theta))*torch.log(torch.exp(dist_q.log_prob(theta))/p_current),0)
print(KL)

state_sum=torch.zeros((n_particles,1))

n_iterations=100
for l in tqdm.tqdm(range(n_iterations)):
    #theta_ntimes[i,j,:]=theta[j,:]
    theta_ntimes=theta[None,:,:].repeat(n_particles,1,1)
    #theta_difference[i,j,:]=theta[j,:]-theta[i,:]
    theta_difference=(theta_ntimes-torch.transpose(theta_ntimes,0,1))     
    #h
    h=float(torch.median(torch.sort(torch.flatten(torch.flatten(torch.triu(torch.pow(torch.sum(torch.pow(theta_difference,2),2),0.5)))))[0][int((n_particles**2+n_particles)/2):]))**2/np.log(n_particles)
    #k[i,j]=k(x_j,x_i)
    k=torch.exp(-1.0/h*torch.sum(torch.pow(theta_difference,2),2))
    #grad_k[i,j,:]=\nabla_x_j k(x_j,x_i)
    grad_k=(-1.0/h*2.0*theta_difference*k[:,:,None])
    #grad_p1=-theta
    grad_p1=(data[:,:,None].repeat(1,1,50)*torch.transpose((torch.exp(-torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))/(torch.transpose((1+torch.exp(-torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))
    grad_p0=-(data[:,:,None].repeat(1,1,50)*torch.transpose((torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))/(torch.transpose((1+torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))    
    #print("Here: ",torch.Tensor.size(data[:,:,None].repeat(1,1,50)),torch.Tensor.size(torch.transpose((1+torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1)))
    grad_p=grad_p0*(1-obs[:,None,None]).repeat(1,28,n_particles)+grad_p1*obs[:,None,None].repeat(1,28,n_particles)
    grad_p=torch.sum(grad_p,0)-torch.transpose(theta,0,1)
    #print("Here: ",torch.Tensor.size(grad_p),torch.Tensor.size(theta),torch.Tensor.size(k),torch.Tensor.size(grad_k))
    phi_summand=1/n_particles*(torch.transpose(grad_p,0,1)[:,None,:].repeat(1,n_particles,1)*k[:,:,None].repeat(1,1,28)+grad_k) 
    phi=torch.sum(phi_summand,1)
    #print(torch.Tensor.size(phi),torch.Tensor.size(theta))
    state_sum=state_sum+torch.pow(phi,2)
    epsilon=0.00001*phi*torch.pow(state_sum,0.5)
    test_tensor=(data[:,:,None].repeat(1,1,50)*torch.transpose((torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))
    if torch.max(test_tensor!=test_tensor)==1:
        print("Error in iteration",l)
        print(torch.Tensor.size(torch.transpose((torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1)),torch.Tensor.size(data[:,:,None].repeat(1,1,50)),torch.Tensor.size(-(data[:,:,None].repeat(1,1,50)*torch.transpose((torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))[:,None,:],0,2).repeat(1,28,1))))
        #print(torch.exp(torch.matmul(theta,torch.transpose(data,0,1))))
        print(torch.max(torch.matmul(theta,torch.transpose(data,0,1))))
        #print(test_tensor)
        break
    theta+=epsilon
    

print(torch.Tensor.size(theta))
p_current=torch.pow(1+torch.exp(-torch.matmul(theta,torch.transpose(data,0,1))),-(obs[None,:].repeat(50,1)))*torch.pow(1+torch.exp(torch.matmul(theta,torch.transpose(data,0,1))),(obs[None,:].repeat(50,1))-1)
p_current=torch.prod(p_current,1)*torch.exp(dist_q.log_prob(theta))
KL=torch.sum(torch.exp(dist_q.log_prob(theta))*torch.log(torch.exp(dist_q.log_prob(theta))/p_current),0)
print(KL)

