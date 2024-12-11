import torch
from torch.nn.functional import softmax as sm
from torch.nn.functional import mse_loss as mse
import numpy as np

a = torch.tensor([[1.0,.01], [.01,1]], requires_grad=True)

Q = torch.tensor([[1.0,0.0], [0.0,1.0]])
data = torch.tensor([[.8, .2],[.2,.8]])
lam = .5

opt = torch.optim.Adam([a], lr = .01)

for lam in np.arange(0, 1.1, .1):
    high1 = 1
    low1 = 0
    high2 = 1
    low2 = 0

    for i in range(100):
        mid1 = (high1+low1)/2
        mid2 = (high2+low2)/2
        a = torch.tensor([[mid1, 1-mid1], [mid2, 1-mid2]])
        qb = torch.matmul(Q, a)
        pol = sm(qb/lam)
        
        if pol[0,0]<=data[0,0]:
            low1 = mid1
        else:
            high1 = mid1
            
        if pol[1,0] <= data[1,0]:
            low2 = mid2
        else:
            high2 = mid2
            
    print(lam)
    print(a)
    print(pol)
    print("-"*10)

# for i in range(1000):
#     belief = sm(a, dim = 1)
#     pred = torch.matmul(Q, belief)
#     pred = sm(pred, dim = 1)
#     err = mse(pred, data)
    
#     opt.zero_grad()
#     err.backward()
#     opt.step()
    
    

# print(belief)
# print(pred)

# pred = torch.matmul(Q, belief)
    
# print(pred)    
