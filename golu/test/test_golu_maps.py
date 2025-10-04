"The following file prints the output and gradients of GoLU Activation"

import torch
from golu.golu_activation import GoLU

x = torch.linspace(-10, 10, 50, requires_grad=True, dtype=torch.float32, device='cuda')
activation_function = GoLU().to(dtype=torch.float32, device='cuda')
y = activation_function(x)
loss = y.sum()
loss.backward()
print('\n\nGoLU #################################################################')
print(f'\nInput - \n{x}\n\n')
print(f'Activation Map - \n{y}\n\n')
print(f'Gradients - \n{x.grad}')
