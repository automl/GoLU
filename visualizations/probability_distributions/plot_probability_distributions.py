"""
This file plots the assumed probability distributions of the respective gated activations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, norm

# Generate x values from -5 to 5
x = np.linspace(-5, 5, 1000)

# Parameters for the Gumbel distribution
mu = 0  # location
beta = 1  # scale

# Step 1: Plot the Gumbel distribution between -5 and 5
pdf_gumbel = gumbel_r.pdf(x, loc=mu, scale=beta)

plt.figure()
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, pdf_gumbel, color='#FF5733')
plt.xlabel('Input Neuron Value')
plt.ylabel('Probability Density')
plt.savefig('./visualizations/probability_distributions/gumbel_distribution.png')  # Save the plot as a PNG file


# Step 2: Plot Gumbel, Gaussian, sigma(x)(1-sigma(x)), and first derivative of tanh(softplus(x))

# Gaussian Distribution
pdf_gaussian = norm.pdf(x, loc=0, scale=1)

# Sigma(x) = 1 / (1 + exp(-x))
sigma_x = 1 / (1 + np.exp(-x))
sigma_derivative = sigma_x * (1 - sigma_x)

# First derivative of tanh(softplus(x))
# Softplus function: softplus(x) = ln(1 + exp(x))
softplus_x = np.log1p(np.exp(x))
tanh_softplus = np.tanh(softplus_x)

# Compute derivative of tanh(softplus(x))
# Derivative: [1 - tanh^2(softplus(x))] * sigma(x)
derivative_tanh_softplus = (1 - tanh_softplus**2) * sigma_x

plt.figure(figsize=(8, 5))
plt.grid(True, color='#afbab2', linewidth=1.5)
plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(x, pdf_gumbel, label='Gumbel', color='#FF5733')
plt.plot(x, pdf_gaussian, label='Gaussian', color='#8BD41E')
plt.plot(x, sigma_derivative, label=r'$\sigma(x)(1 - \sigma(x))$', color='#3AA5D6')
plt.plot(x, derivative_tanh_softplus, label=r'$(1 - \tanh^2(\text{softplus}(x)))\sigma(x)$', color='#7921B8')
plt.xlabel('Input Neuron Value', fontsize=13)
plt.ylabel('Probability Density', fontsize=13)
plt.legend()
plt.savefig('./visualizations/probability_distributions/comparison_plot.png', dpi=300)  # Save the plot as a PNG file
