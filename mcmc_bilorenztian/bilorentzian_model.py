import pyro
import pyro.distributions as dist
import torch

def model(data):
    A = pyro.sample("A", dist.Normal(53., 1.0)) # sample a value for A uniformly between 0 and 10
    X =  pyro.sample("X", dist.Normal(8., 0.5))
    B = A + X # sample a value for B uniformly between 0 and 10
    gamma1 = pyro.sample("gamma1", dist.Normal(7.0, 1.5))
    gamma2 = gamma1#pyro.sample("gamma2", dist.Normal(6.0, 2.5)).double()
    amp = pyro.sample("amp", dist.LogNormal(3.0, 0.25))
    var = pyro.sample("var", dist.HalfNormal(scale=0.1))

    F =  (amp) * (0.5 * gamma1) / ((data[0] - A)**2 + (0.5 * gamma1)**2) \
        + (amp) * (0.5 * gamma2) / ((data[0] - B)**2 + (0.5 * gamma2)**2)
    # ensure F is a 1-D mean vector matching y_obs
    F = F.squeeze()

    #with pyro.plate("data", data[0].size(0)):
    # pyro.sample("obs", dist.MultivariateNormal(F, var * torch.eye(data[1].shape[0]).double()), obs=data[1])
    pyro.sample("obs", dist.MultivariateNormal(F, var * torch.eye(data[1].shape[0])), obs=data[1])