import torch
from torch import from_numpy
import numpy as np
import matplotlib.pyplot as plt

from tasks.utils.utils import create_directory
from visualizations.golu_plots.gates import golu_act, gompertz, gompertz_backward_x, \
    golu_backward_x, golu_backward_alpha, golu_backward_beta, golu_backward_gamma
from visualizations.golu_plots.gates import relu, relu_backward_x, leakyrelu, leakyrelu_backward_x, \
    elu, elu_backward_x, gelu, gelu_backward_x, swish, swish_backward_x, gaussian_cdf, sigmoid, mish, mish_backward_x, \
        tanhsoftplus, relu_gate, tanh


root_path = './visualizations/golu_plots/plots'
create_directory(root_path)

# Plot the GoLU Activation and Gompertz Function ------------------------------------------------------------
x_golu = np.linspace(-3, 3, 1000)
x_gompertz = np.linspace(-5, 8, 1000)
x_yintercept = np.array([0])
golu_output = golu_act(x_golu)
gompertz_output = gompertz(x_gompertz)
gompertz_output_y_intercept = gompertz(x_yintercept)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_golu, golu_output, color='teal')
# plt.title(r'GoLU Activation with $\alpha$, $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axhline(gompertz_output_y_intercept, linestyle='dotted', color='#919191', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.scatter(x_yintercept, gompertz_output_y_intercept, color='red')
plt.plot(x_gompertz, gompertz_output, color='red')
# plt.title(r'Gompertz Function with $\alpha$, $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)

plt.savefig(f'{root_path}/visualize_golu.png')

# Plot the first derivatives of GoLU wrt x, alpha, beta and gamma -------------------------------------------
x = np.linspace(-5, 5, 1000)
alpha = np.linspace(-5, 5, 1000)
beta = np.linspace(-5, 5, 1000)
gamma = np.linspace(-5, 5, 1000)

x_backward = golu_backward_x(x)
alpha_backward = golu_backward_alpha(alpha)
beta_backward = golu_backward_beta(beta)
gamma_backward = golu_backward_gamma(gamma)

plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_backward, color='teal')
# plt.title(r'First Derivative of GoLU with respect to x and fixed $\alpha$, $\beta$, $\gamma$ to 1', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('Gradient', fontsize=15)

plt.subplot(2, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(alpha, alpha_backward, color='teal')
# plt.title(r'First Derivative of GoLU with respect to $\alpha$ and fixed x, $\beta$, $\gamma$ to 1', fontsize=15)
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel('Gradient', fontsize=15)

plt.subplot(2, 2, 3)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(beta, beta_backward, color='teal')
# plt.title(r'First Derivative of GoLU with respect to $\beta$ and fixed x, $\alpha$, $\gamma$ to 1', fontsize=15)
plt.xlabel(r'$\beta$', fontsize=15)
plt.ylabel('Gradient', fontsize=15)

plt.subplot(2, 2, 4)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(gamma, gamma_backward, color='teal')
# plt.title(r'First Derivative of GoLU with respect to $\gamma$ and fixed x, $\alpha$, $\beta$ to 1', fontsize=15)
plt.xlabel(r'$\gamma$', fontsize=15)
plt.ylabel('Gradient', fontsize=15)

plt.savefig(f'{root_path}/visualize_golu_grads.png')


# Plot the effect of changing alpha --------------------------------------------------------------------------
alpha_1 = 0.5
alpha_2 = 1
alpha_3 = 1.5

beta = 1
gamma = 1

x = np.linspace(-5, 5, 1000)
x_golu_alpha_1 = golu_act(x, alpha=alpha_1)
x_golu_alpha_2 = golu_act(x, alpha=alpha_2)
x_golu_alpha_3 = golu_act(x, alpha=alpha_3)
x_gompertz_alpha_1 = gompertz(x, alpha=alpha_1)
x_gompertz_alpha_2 = gompertz(x, alpha=alpha_2)
x_gompertz_alpha_3 = gompertz(x, alpha=alpha_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_alpha_1, color='red', label=r'$\alpha$' + f'={alpha_1}')
plt.plot(x, x_golu_alpha_2, color='green', label=r'$\alpha$' + f'={alpha_2}')
plt.plot(x, x_golu_alpha_3, color='orange', label=r'$\alpha$' + f'={alpha_3}')
# plt.title(r'GoLU Activation with different $\alpha$. $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_alpha_1, color='red', label=r'$\alpha$' + f'={alpha_1}')
plt.plot(x, x_gompertz_alpha_2, color='green', label=r'$\alpha$' + f'={alpha_2}')
plt.plot(x, x_gompertz_alpha_3, color='orange', label=r'$\alpha$' + f'={alpha_3}')
# plt.title(r'Gompertz Function with different $\alpha$. $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_alpha.png')


# Plot the effect of changing beta ---------------------------------------------------------------------------
beta_1 = 0.1
beta_2 = 1
beta_3 = 10

alpha = 1
gamma = 1

x = np.linspace(-7, 7, 1000)
x_golu_beta_1 = golu_act(x, beta=beta_1)
x_golu_beta_2 = golu_act(x, beta=beta_2)
x_golu_beta_3 = golu_act(x, beta=beta_3)
x_gompertz_beta_1 = gompertz(x, beta=beta_1)
x_gompertz_beta_2 = gompertz(x, beta=beta_2)
x_gompertz_beta_3 = gompertz(x, beta=beta_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_beta_1, color='red', label=r'$\beta$' + f'={beta_1}')
plt.plot(x, x_golu_beta_2, color='green', label=r'$\beta$' + f'={beta_2}')
plt.plot(x, x_golu_beta_3, color='orange', label=r'$\beta$' + f'={beta_3}')
# plt.title(r'GoLU Activation with different $\beta$. $\alpha$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_beta_1, color='red', label=r'$\beta$' + f'={beta_1}')
plt.plot(x, x_gompertz_beta_2, color='green', label=r'$\beta$' + f'={beta_2}')
plt.plot(x, x_gompertz_beta_3, color='orange', label=r'$\beta$' + f'={beta_3}')
# plt.title(r'Gompertz Function with different $\beta$. $\alpha$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_beta.png')

# Plot the effect of changing gamma --------------------------------------------------------------------------
gamma_1 = 0.5
gamma_2 = 1
gamma_3 = 2

alpha = 1
beta = 1

x = np.linspace(-9, 9, 1000)
x_golu_gamma_1 = golu_act(x, gamma=gamma_1)
x_golu_gamma_2 = golu_act(x, gamma=gamma_2)
x_golu_gamma_3 = golu_act(x, gamma=gamma_3)
x_gompertz_gamma_1 = gompertz(x, gamma=gamma_1)
x_gompertz_gamma_2 = gompertz(x, gamma=gamma_2)
x_gompertz_gamma_3 = gompertz(x, gamma=gamma_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_gamma_1, color='red', label=r'$\gamma$' + f'={gamma_1}')
plt.plot(x, x_golu_gamma_2, color='green', label=r'$\gamma$' + f'={gamma_2}')
plt.plot(x, x_golu_gamma_3, color='orange', label=r'$\gamma$' + f'={gamma_3}')
# plt.title(r'GoLU Activation with different $\gamma$. $\alpha$, and $\beta$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_gamma_1, color='red', label=r'$\gamma$' + f'={gamma_1}')
plt.plot(x, x_gompertz_gamma_2, color='green', label=r'$\gamma$' + f'={gamma_2}')
plt.plot(x, x_gompertz_gamma_3, color='orange', label=r'$\gamma$' + f'={gamma_3}')
# plt.title(r'Gompertz Function with different $\gamma$. $\alpha$, and $\beta$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_gamma.png')

# Plot effect of alpha on gradient -------------------------------------------------------------------
alpha_1 = 0.5
alpha_2 = 1
alpha_3 = 1.5

beta = 1
gamma = 1

x = np.linspace(-5, 5, 1000)
x_golu_alpha_1 = golu_backward_x(x, alpha=alpha_1)
x_golu_alpha_2 = golu_backward_x(x, alpha=alpha_2)
x_golu_alpha_3 = golu_backward_x(x, alpha=alpha_3)
x_gompertz_alpha_1 = gompertz_backward_x(x, alpha=alpha_1)
x_gompertz_alpha_2 = gompertz_backward_x(x, alpha=alpha_2)
x_gompertz_alpha_3 = gompertz_backward_x(x, alpha=alpha_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_alpha_1, color='red', label=r'$\alpha$' + f'={alpha_1}')
plt.plot(x, x_golu_alpha_2, color='green', label=r'$\alpha$' + f'={alpha_2}')
plt.plot(x, x_golu_alpha_3, color='orange', label=r'$\alpha$' + f'={alpha_3}')
# plt.title(r'1st Derivative of GoLU Activation with different $\alpha$. $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gradient', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_alpha_1, color='red', label=r'$\alpha$' + f'={alpha_1}')
plt.plot(x, x_gompertz_alpha_2, color='green', label=r'$\alpha$' + f'={alpha_2}')
plt.plot(x, x_gompertz_alpha_3, color='orange', label=r'$\alpha$' + f'={alpha_3}')
# plt.title(r'1st Derivative of Gompertz Function with different $\alpha$. $\beta$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Gradient', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_gradient_alpha.png')

# Plot effect of beta on gradient ----------------------------------------------------------------------------
beta_1 = 0.1
beta_2 = 1
beta_3 = 10

alpha = 1
gamma = 1

x = np.linspace(-7, 7, 1000)
x_golu_beta_1 = golu_backward_x(x, beta=beta_1)
x_golu_beta_2 = golu_backward_x(x, beta=beta_2)
x_golu_beta_3 = golu_backward_x(x, beta=beta_3)
x_gompertz_beta_1 = gompertz_backward_x(x, beta=beta_1)
x_gompertz_beta_2 = gompertz_backward_x(x, beta=beta_2)
x_gompertz_beta_3 = gompertz_backward_x(x, beta=beta_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_beta_1, color='red', label=r'$\beta$' + f'={beta_1}')
plt.plot(x, x_golu_beta_2, color='green', label=r'$\beta$' + f'={beta_2}')
plt.plot(x, x_golu_beta_3, color='orange', label=r'$\beta$' + f'={beta_3}')
# plt.title(r'1st Derivative of GoLU Activation with different $\beta$. $\alpha$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gradient', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_beta_1, color='red', label=r'$\beta$' + f'={beta_1}')
plt.plot(x, x_gompertz_beta_2, color='green', label=r'$\beta$' + f'={beta_2}')
plt.plot(x, x_gompertz_beta_3, color='orange', label=r'$\beta$' + f'={beta_3}')
# plt.title(r'1st Derivative of Gompertz Function with different $\beta$. $\alpha$, and $\gamma$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Gradient', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_gradient_beta.png')

# Plot effect of gamma on gradient ------------------------------------------------------------------------
gamma_1 = 0.5
gamma_2 = 1
gamma_3 = 2

alpha = 1
beta = 1

x = np.linspace(-9, 9, 1000)
x_golu_gamma_1 = golu_backward_x(x, gamma=gamma_1)
x_golu_gamma_2 = golu_backward_x(x, gamma=gamma_2)
x_golu_gamma_3 = golu_backward_x(x, gamma=gamma_3)
x_gompertz_gamma_1 = gompertz_backward_x(x, gamma=gamma_1)
x_gompertz_gamma_2 = gompertz_backward_x(x, gamma=gamma_2)
x_gompertz_gamma_3 = gompertz_backward_x(x, gamma=gamma_3)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_golu_gamma_1, color='red', label=r'$\gamma$' + f'={gamma_1}')
plt.plot(x, x_golu_gamma_2, color='green', label=r'$\gamma$' + f'={gamma_2}')
plt.plot(x, x_golu_gamma_3, color='orange', label=r'$\gamma$' + f'={gamma_3}')
# plt.title(r'1st Derivative of GoLU Activation with different $\gamma$. $\alpha$, and $\beta$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gradient', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gompertz_gamma_1, color='red', label=r'$\gamma$' + f'={gamma_1}')
plt.plot(x, x_gompertz_gamma_2, color='green', label=r'$\gamma$' + f'={gamma_2}')
plt.plot(x, x_gompertz_gamma_3, color='orange', label=r'$\gamma$' + f'={gamma_3}')
# plt.title(r'1st Derivative of Gompertz Function with different $\gamma$. $\alpha$, and $\beta$ equals 1')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Gradient', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/changing_gradient_gamma.png')

# Plot Activations and Derivatives of GoLU, GELU, Swish, ReLU, LeakyReLU, ELU -------------------------------
x = np.linspace(-3, 3, 1000)

x_relu = relu(x)
x_leakyrelu = leakyrelu(x)
x_elu = elu(x)
x_gelu = gelu(x)
x_swish = swish(x)
x_golu = golu_act(x)

x_relu_backward = relu_backward_x(x)
x_leakyrelu_backward = leakyrelu_backward_x(x)
x_elu_backward = elu_backward_x(x)
x_gelu_backward = gelu_backward_x(x)
x_swish_backward = swish_backward_x(x)
x_golu_backward = golu_backward_x(x)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_relu, color='red', linestyle='dotted', label='ReLU')
plt.plot(x, x_leakyrelu, color='green', linestyle='dotted', label='LeakyReLU')
plt.plot(x, x_elu, color='orange', linestyle='dotted', label='ELU')
plt.plot(x, x_gelu, color='red', label='GELU')
plt.plot(x, x_swish, color='green', label='Swish')
plt.plot(x, x_golu, color='orange', label='GoLU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_relu_backward, color='red', linestyle='dotted', label='ReLU')
plt.plot(x, x_leakyrelu_backward, color='green', linestyle='dotted', label='LeakyReLU')
plt.plot(x, x_elu_backward, color='orange', linestyle='dotted', label='ELU')
plt.plot(x, x_gelu_backward, color='red', label='GELU')
plt.plot(x, x_swish_backward, color='green', label='Swish')
plt.plot(x, x_golu_backward, color='orange', label='GoLU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gradient Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/activations.png')

# Plot Gates and Derivatives of GoLU, GELU, Swish, Mish -------------------------------
x = np.linspace(-4, 4, 1000)

x_gompertz = gompertz(x)
x_gcdf = gaussian_cdf(x)
x_sigmoid = sigmoid(x)
x_tanhsoftplus = tanhsoftplus(x)

x_golu_derivative = golu_backward_x(x)
x_gelu_derivative = gelu_backward_x(x)
x_swish_derivative = swish_backward_x(x)
x_mish_derivative = mish_backward_x(x)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gcdf, color='#8BD41E', label='Gaussian CDF')
plt.plot(x, x_sigmoid, color='#3AA5D6', label='Sigmoid')
plt.plot(x, x_tanhsoftplus, color='#7921B8', label='tanh(softplus(x))')
plt.plot(x, x_gompertz, color='#FF5733', label='Gompertz')
# plt.title('Gaussian CDF, Sigmoid, TanhSoftplus and Gompertz Gates')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, x_gelu_derivative, color='#8BD41E', label='GELU')
plt.plot(x, x_swish_derivative, color='#3AA5D6', label='Swish')
plt.plot(x, x_mish_derivative, color='#7921B8', label='Mish')
plt.plot(x, x_golu_derivative, color='#FF5733', label='GoLU')
# plt.title('First Derivatives of GELU, Swish, Mish and GoLU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gradient Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/gates.png')


# Plot Gates and Derivatives of GoLU, GELU, Swish -------------------------------
x = np.linspace(-4, 4, 1000)

x_gompertz = gompertz(x)
x_gcdf = gaussian_cdf(x)
x_sigmoid = sigmoid(x)
x_tanhsoftplus = tanhsoftplus(x)
x_relu_gate = relu_gate(x)

x_marker = 0.5
sample_x_gompertz = gompertz(np.array([x_marker]))
sample_x_gaussian_cdf = gaussian_cdf(np.array([x_marker]))
sample_x_sigmoid = sigmoid(np.array([x_marker]))
sample_x_tanhsoftplus = tanhsoftplus(np.array([x_marker]))
sample_x_relu_gate = relu_gate(np.array([x_marker]))

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Comparing Activation Gates with x_marker at 0.5", fontsize=16)

# Row 1, Col 1: Gaussian CDF and Gompertz
axs[0, 0].grid(True, color='#afbab2', linewidth=1.5)
axs[0, 0].axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[0, 0].axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[0, 0].axvline(x_marker, color='blue', linestyle='dashed', linewidth=1.5)
axs[0, 0].plot(x, x_gcdf, color='red', label='Gaussian CDF')
axs[0, 0].plot(x, x_gompertz, color='green', label='Gompertz')
axs[0, 0].plot(x_marker, sample_x_gaussian_cdf, 'ro')
axs[0, 0].plot(x_marker, sample_x_gompertz, 'go')
axs[0, 0].text(x_marker - 0.8, sample_x_gaussian_cdf, f'{sample_x_gaussian_cdf[0]:.2f}', color='red', fontweight='bold')
axs[0, 0].text(x_marker + 0.1, sample_x_gompertz, f'{sample_x_gompertz[0]:.2f}', color='green', fontweight='bold')
axs[0, 0].set_title('Gaussian CDF and Gompertz Gates')
axs[0, 0].set_ylabel('Output Neuron Value', fontsize=13)
axs[0, 0].legend()

# Row 1, Col 2: Sigmoid and Gompertz
axs[0, 1].grid(True, color='#afbab2', linewidth=1.5)
axs[0, 1].axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[0, 1].axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[0, 1].axvline(x_marker, color='blue', linestyle='dashed', linewidth=1.5)
axs[0, 1].plot(x, x_sigmoid, color='red', label='Sigmoid')
axs[0, 1].plot(x, x_gompertz, color='green', label='Gompertz')
axs[0, 1].plot(x_marker, sample_x_sigmoid, 'ro')
axs[0, 1].plot(x_marker, sample_x_gompertz, 'go')
axs[0, 1].text(x_marker - 0.8, sample_x_sigmoid, f'{sample_x_sigmoid[0]:.2f}', color='red', fontweight='bold')
axs[0, 1].text(x_marker + 0.1, sample_x_gompertz, f'{sample_x_gompertz[0]:.2f}', color='green', fontweight='bold')
axs[0, 1].set_title('Sigmoid and Gompertz Gates')
axs[0, 1].legend()

# Row 2, Col 1: tanh(softplus(x)) and Gompertz
axs[1, 0].grid(True, color='#afbab2', linewidth=1.5)
axs[1, 0].axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[1, 0].axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[1, 0].axvline(x_marker, color='blue', linestyle='dashed', linewidth=1.5)
axs[1, 0].plot(x, x_tanhsoftplus, color='red', label='tanh(softplus(x))')
axs[1, 0].plot(x, x_gompertz, color='green', label='Gompertz')
axs[1, 0].plot(x_marker, sample_x_tanhsoftplus, 'ro')
axs[1, 0].plot(x_marker, sample_x_gompertz, 'go')
axs[1, 0].text(x_marker - 0.8, sample_x_tanhsoftplus, f'{sample_x_tanhsoftplus[0]:.2f}', color='red', fontweight='bold')
axs[1, 0].text(x_marker + 0.1, sample_x_gompertz, f'{sample_x_gompertz[0]:.2f}', color='green', fontweight='bold')
axs[1, 0].set_title('tanh(softplus(x)) and Gompertz Gates')
axs[1, 0].set_xlabel('Input Neuron Value', fontsize=13)
axs[1, 0].set_ylabel('Output Neuron Value', fontsize=13)
axs[1, 0].legend()

# Row 2, Col 2: Step function (ReLU gate) and Gompertz
axs[1, 1].grid(True, color='#afbab2', linewidth=1.5)
axs[1, 1].axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[1, 1].axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
axs[1, 1].axvline(x_marker, color='blue', linestyle='dashed', linewidth=1.5)
axs[1, 1].plot(x, x_relu_gate, color='red', label='ReLU Gate')
axs[1, 1].plot(x, x_gompertz, color='green', label='Gompertz')
axs[1, 1].plot(x_marker, sample_x_relu_gate, 'ro')
axs[1, 1].plot(x_marker, sample_x_gompertz, 'go')
axs[1, 1].text(x_marker - 1.3, sample_x_relu_gate, f'{sample_x_relu_gate[0]:.2f}', color='red', fontweight='bold')
axs[1, 1].text(x_marker + 0.1, sample_x_gompertz, f'{sample_x_gompertz[0]:.2f}', color='green', fontweight='bold')
axs[1, 1].set_title('Step Function (ReLU Gate) and Gompertz Gates')
axs[1, 1].set_xlabel('Input Neuron Value', fontsize=13)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{root_path}/gates_with_x_marks.png')


# Visualize instability of beta and gamma ----------------------------------------------

alpha_1 = 1
beta_1 = -0.1
gamma_1 = 1

alpha_2 = 1
beta_2 = 1
gamma_2 = -0.9


x_1 = np.linspace(-4, 4, 1000)
x_2 = np.linspace(-10, 4, 1000)
x_beta = gompertz(x_1, alpha=alpha_1, beta=beta_1, gamma=gamma_1)
x_gamma = gompertz(x_2, alpha=alpha_2, beta=beta_2, gamma=gamma_2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_1, x_beta, color='red', label=r'$\beta$ = -0.1')
# plt.title('Effect of negative beta on Gompertz Function')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
# plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_2, x_gamma, color='green', label=r'$\gamma$ = -0.9')
# plt.title('Effect of negative gamma on Gompertz Function')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Gate Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/negative_beta_gamma.png')

# Visualize Sigmoid and TanH gradients ------------------------------------------------------------------------------

# Define the input range
x = np.linspace(-4, 4, 1000)

# Compute gradients
sigmoid_gradient = sigmoid(x) * (1 - sigmoid(x))
tanh_gradient = (1 - tanh(x) ** 2)

# Plot settings
plt.figure(figsize=(15, 5))

# Plot Sigmoid gradient
plt.subplot(1, 2, 1)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, sigmoid_gradient, color='red', label='Sigmoid Gradient')
# plt.title('Gradient of Sigmoid Function')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Gradient Value', fontsize=13)
plt.legend()

# Plot Tanh gradient
plt.subplot(1, 2, 2)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, tanh_gradient, color='green', label='Tanh Gradient')
# plt.title('Gradient of Tanh Function')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Gradient Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/sigmoid_tanh_grads.png')

# Visualize negative parts of Gated Activations and ELU ---------------------------------------------------------

# Define the input range
x_neg = np.linspace(-4, 1, 1000)
x_pos = np.linspace(-1, 4, 1000)

y_gelu_neg = gelu(x_neg)
y_swish_neg = swish(x_neg)
y_mish_neg = mish(x_neg)
y_golu_neg = golu_act(x_neg)
y_elu_neg = elu(x_neg)

y_gelu_pos = gelu(x_pos)
y_swish_pos = swish(x_pos)
y_mish_pos = mish(x_pos)
y_golu_pos = golu_act(x_pos)
y_elu_pos = elu(x_pos)

# Plot settings
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_neg, y_gelu_neg, color='red', label='GELU')
plt.plot(x_neg, y_swish_neg, color='green', label='Swish')
plt.plot(x_neg, y_mish_neg, color='blue', label='Mish')
plt.plot(x_neg, y_golu_neg, color='orange', label='GoLU')
plt.plot(x_neg, y_elu_neg, color='black', label='ELU')
# plt.title('Negative Neuron Space of GELU, Swish, Mish, GoLU and ELU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_pos, y_gelu_pos, color='red', label='GELU')
plt.plot(x_pos, y_swish_pos, color='green', label='Swish')
plt.plot(x_pos, y_mish_pos, color='blue', label='Mish')
plt.plot(x_pos, y_golu_pos, color='orange', label='GoLU')
plt.plot(x_pos, y_elu_pos, color='black', label='ELU')
# plt.title('Positive Neuron Space of GELU, Swish, Mish, GoLU and ELU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/negative_positive_neuron_space.png')



# Visualize negative parts of Gated Activations and ELU ---------------------------------------------------------

# Define the input range
x_neg = np.linspace(-3, 3, 1000)
x_pos = np.linspace(-1, 4, 1000)

y_gelu_neg = gelu(x_neg)
y_swish_neg = swish(x_neg)
y_mish_neg = mish(x_neg)
y_golu_neg = golu_act(x_neg)
y_elu_neg = elu(x_neg)

y_gelu_pos = gelu(x_pos)
y_swish_pos = swish(x_pos)
y_mish_pos = mish(x_pos)
y_golu_pos = golu_act(x_pos)
y_elu_pos = elu(x_pos)

# Plot settings
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_neg, y_gelu_neg, color='#8BD41E', label='GELU')
plt.plot(x_neg, y_swish_neg, color='#3AA5D6', label='Swish')
plt.plot(x_neg, y_mish_neg, color='#7921B8', label='Mish')
plt.plot(x_neg, y_golu_neg, color='#FF5733', label='GoLU')
plt.plot(x_neg, y_elu_neg, color='black', label='ELU')
# plt.title('Negative Neuron Space of GELU, Swish, Mish, GoLU and ELU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.subplot(1, 2, 2)
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x_pos, y_gelu_pos, color='#8BD41E', label='GELU')
plt.plot(x_pos, y_swish_pos, color='#3AA5D6', label='Swish')
plt.plot(x_pos, y_mish_pos, color='#7921B8', label='Mish')
plt.plot(x_pos, y_golu_pos, color='#FF5733', label='GoLU')
plt.plot(x_pos, y_elu_pos, color='black', label='ELU')
# plt.title('Positive Neuron Space of GELU, Swish, Mish, GoLU and ELU')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Output Neuron Value', fontsize=13)
plt.legend()

plt.savefig(f'{root_path}/full_scale_neuron_space.png')
