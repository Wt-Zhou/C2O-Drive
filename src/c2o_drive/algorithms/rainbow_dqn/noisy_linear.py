"""Noisy Linear Layer for Rainbow DQN

Implements factorized Gaussian noise for parameter-space exploration,
replacing epsilon-greedy exploration. This is one of the six Rainbow components.

Reference:
    Fortunato et al. "Noisy Networks for Exploration" (2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with learnable noise parameters.

    This layer adds parametric noise to the weights and biases,
    enabling learned exploration in parameter space rather than
    action space (as in epsilon-greedy).

    The forward pass computes:
        y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)

    where:
        μ: mean parameters (learnable)
        σ: noise scale parameters (learnable)
        ε: noise samples (sampled each forward pass)
        ⊙: element-wise product

    Attributes:
        in_features: Size of input features
        out_features: Size of output features
        sigma_init: Initial value for noise scale parameters
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """Initialize Noisy Linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            sigma_init: Initial standard deviation for noise parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: mean values
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Learnable parameters: noise scale
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Non-learnable buffers: noise samples
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        # Initialize parameters and noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize learnable parameters.

        Uses initialization scheme from the Noisy Networks paper:
        - μ sampled from uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        - σ initialized to sigma_init / sqrt(in_features)
        """
        mu_range = 1.0 / math.sqrt(self.in_features)

        # Initialize mean parameters
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # Initialize noise scale parameters
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Sample new noise values.

        Uses factorized Gaussian noise for efficiency:
        - Sample ε_in ~ f(ε) for inputs
        - Sample ε_out ~ f(ε) for outputs
        - Weight noise = ε_out ⊗ ε_in (outer product)
        - Bias noise = ε_out

        where f(ε) = sgn(ε) * sqrt(|ε|) for reduced correlation.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product for factorized noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise.

        Uses the factorization trick: f(ε) = sgn(ε) * sqrt(|ε|)
        This reduces correlation between noise samples.

        Args:
            size: Number of noise samples to generate

        Returns:
            Scaled noise tensor of shape (size,)
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters.

        During training, adds noise to weights and biases.
        During evaluation, uses only mean parameters (no noise).

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        if self.training:
            # Training mode: use noisy parameters
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Evaluation mode: use mean parameters only
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, sigma_init={self.sigma_init}'
