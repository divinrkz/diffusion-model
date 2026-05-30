"""
diffusion/vp.py  —  Variance-Preserving (VP) SDE
=================================================
Part 5 of EE/CS 148B HW4.

Reference: Song et al. (2021) "Score-Based Generative Modeling through
Stochastic Differential Equations" (Song21), Appendix B & D.

Students implement every method marked TODO.  Methods marked PROVIDED
are complete and should not be modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VPSDE:
    """Variance-Preserving SDE forward process and samplers.

    The VP-SDE is:
        dx = -½ β(t) x dt + √β(t) dB_t

    with β(t) = β_min + (β_max - β_min) * t  (linear schedule).

    Args:
        beta_min: Minimum noise schedule value β_min.
        beta_max: Maximum noise schedule value β_max.
        T:        Number of discrete time steps (used by the EM/PC samplers).
    """

    def __init__(self, beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    # ------------------------------------------------------------------
    # 5.A  Defining the VP SDE
    # ------------------------------------------------------------------

    def beta(self, t: Tensor) -> Tensor:
        """β(t) — the linear noise schedule.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            β(t), same shape as t.

        Reference: Eq. (32) of Song21.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def c(self, t: Tensor) -> Tensor:
        """c(t) = exp(-½ ∫_0^t β(s) ds) — the signal decay factor.

        For a linear β schedule:
            ∫_0^t β(s) ds = β_min * t + ½ (β_max - β_min) * t²

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            c(t), same shape as t.

        Reference: Eq. (33) of Song21.
        """
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def sigma(self, t: Tensor) -> Tensor:
        """σ(t) = √(1 - c(t)²) — the noise standard deviation.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            σ(t), same shape as t.
        """
        return torch.sqrt(1.0 - self.c(t) ** 2)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient  f(x, t) = -½ β(t) x.

        Args:
            x: State tensor, shape (B, *).
            t: Time tensor, shape (B,) broadcast-compatible with x.

        Returns:
            Drift f(x, t), same shape as x.
        """
        beta_t = self.beta(t).view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Diffusion coefficient  g(t) = √β(t).

        Args:
            t: Time tensor, shape (*).

        Returns:
            g(t), same shape as t.
        """
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample from the forward marginal  q(x_t | x_0).

        The marginal satisfies:
            x_t = c(t) * x_0 + σ(t) * ε,   ε ~ N(0, I)

        Args:
            x0: Clean data, shape (B, *).
            t:  Continuous time in [0, 1], shape (B,).

        Returns:
            (x_t, eps): noised sample and the noise used, both shape (B, *).
        """
        eps = torch.randn_like(x0)
        c_t = self.c(t).view(-1, *([1] * (x0.dim() - 1)))
        sigma_t = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        x_t = c_t * x0 + sigma_t * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # 5.B  Samplers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_maruyama(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Euler-Maruyama reverse-SDE sampler (Problem 5.B.i).

        Starting from x(T=1) ~ N(0, σ(1)² I), integrates the reverse VP-SDE:
            dx = [-½ β(t) x - β(t) ∇_x log p_t(x)] dt + √β(t) dB̄_t

        Args:
            score_model: Trained score network s_θ(x, t).
                         Called as `score_model(x, t)` where t is a float
                         tensor of shape (B,) with values in [0, 1].
            shape:       Output shape (B, C, H, W).
            num_steps:   Number of discretisation steps (default: self.T).
            device:      Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        eps = 1e-3
        batch_size = shape[0]
        dev = torch.device(device)

        t1 = torch.ones(batch_size, device=dev)
        std_1 = self.sigma(t1).view(-1, *([1] * (len(shape) - 1)))
        if initial_noise is not None:
            x = initial_noise.to(dev) * std_1.view(-1, *([1] * (len(shape) - 1)))
        else:
            x = torch.randn(shape, device=dev) * std_1.view(-1, *([1] * (len(shape) - 1)))

        step_size = torch.tensor((1.0 - eps) / num_steps, device=dev)
        time_steps = torch.linspace(1.0, eps + step_size, num_steps, device=dev)

        for time_step in time_steps:
            batch_t = torch.full((batch_size,), time_step, device=dev)
            score = score_model(x, batch_t)
            drift = self.drift(x, batch_t)
            g = self.diffusion(batch_t).view(-1, *([1] * (len(shape) - 1)))
            x_mean = x - drift * step_size + g ** 2 * score * step_size
            if time_step > time_steps[-1]:
                z = torch.randn_like(x)
                x = x_mean + g * torch.sqrt(step_size) * z
            else:
                x = x_mean
        return x

    @torch.no_grad()
    def predictor_corrector(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        n_corrector: int = 1,
        snr: float = 0.16,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Predictor-Corrector sampler with EM predictor (Problem 5.B.ii).

        Follows Algorithm 5 of Song21.  Each predictor step is an EM step;
        each corrector step is one step of annealed Langevin dynamics.

        Args:
            score_model:  Trained score network s_θ(x, t).
            shape:        Output shape (B, C, H, W).
            num_steps:    Number of predictor steps (default: self.T).
            n_corrector:  Number of Langevin corrector steps per predictor step.
            snr:          Signal-to-noise ratio for the corrector step size.
            device:       Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        eps = 1e-3
        batch_size = shape[0]
        dev = torch.device(device)

        t1 = torch.ones(batch_size, device=dev)
        std_1 = self.sigma(t1).view(-1, *([1] * (len(shape) - 1)))
        x = torch.randn(shape, device=dev) * std_1.view(-1, *([1] * (len(shape) - 1)))

        step_size = torch.tensor((1.0 - eps) / num_steps, device=dev)
        time_steps = torch.linspace(1.0, eps + step_size, num_steps, device=dev)

        for time_step in time_steps:
            batch_t = torch.full((batch_size,), time_step, device=dev)
            alpha_t = self.c(batch_t).mean().item()

            for _ in range(n_corrector):
                z = torch.randn_like(x)
                score = score_model(x, batch_t)
                grad_norm = torch.norm(score.reshape(batch_size, -1), dim=-1).mean()
                noise_norm = torch.norm(z.reshape(batch_size, -1), dim=-1).mean()
                langevin_step = 2 * alpha_t * (snr * noise_norm / (grad_norm + 1e-8)) ** 2
                x = x + langevin_step * score + torch.sqrt(2 * langevin_step) * z

            score = score_model(x, batch_t)
            drift = self.drift(x, batch_t)
            g = self.diffusion(batch_t).view(-1, *([1] * (len(shape) - 1)))
            x_mean = x - drift * step_size + g ** 2 * score * step_size
            if time_step > time_steps[-1]:
                z = torch.randn_like(x)
                x = x_mean + g * torch.sqrt(step_size) * z
            else:
                x = x_mean
        return x

    @torch.no_grad()
    def ddim(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Deterministic DDIM sampler (η = 0) for the VP process.

        Uses the score network to predict noise ε̂ = −σ(t) s_θ(x_t, t), then
        applies the standard DDIM update with α(t) = c(t).
        """
        num_steps = num_steps or self.T
        eps = 1e-3
        batch_size = shape[0]
        dev = torch.device(device)

        t1 = torch.ones(batch_size, device=dev)
        std_1 = self.sigma(t1).view(-1, *([1] * (len(shape) - 1)))
        if initial_noise is not None:
            x = initial_noise.to(dev) * std_1.view(-1, *([1] * (len(shape) - 1)))
        else:
            x = torch.randn(shape, device=dev) * std_1.view(-1, *([1] * (len(shape) - 1)))

        step_size = torch.tensor((1.0 - eps) / num_steps, device=dev)
        time_steps = torch.linspace(1.0, eps + step_size, num_steps, device=dev)
        for i, t_cur in enumerate(time_steps):
            t_next = max(eps, t_cur.item() - step_size)
            batch_t = torch.full((batch_size,), t_cur, device=dev)
            alpha_t = self.c(batch_t).view(-1, *([1] * (len(shape) - 1)))
            sigma_t = self.sigma(batch_t).view(-1, *([1] * (len(shape) - 1)))
            score = score_model(x, batch_t)
            eps_pred = -sigma_t * score

            alpha_next = self.c(torch.full((batch_size,), t_next, device=dev)).view(
                -1, *([1] * (len(shape) - 1))
            )
            sigma_next = self.sigma(torch.full((batch_size,), t_next, device=dev)).view(
                -1, *([1] * (len(shape) - 1))
            )
            x0_pred = (x - sigma_t * eps_pred) / alpha_t.clamp(min=1e-8)
            x = alpha_next * x0_pred + sigma_next * eps_pred
        return x

    # ------------------------------------------------------------------
    # 5.D  Inverse problems (EC)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        score_model: nn.Module,
        corrupted: Tensor,
        mask: Tensor,
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Conditional reverse diffusion for inpainting (EC Problem 5.D).

        At each reverse step, replaces the known pixels with their
        forward-diffused ground-truth values, conditioning the reverse
        process on the observed measurements.

        Reference: Song et al. (2022) "Solving Inverse Problems in Medical
        Imaging with Score-Based Generative Models".

        Args:
            score_model: Trained score network s_θ(x, t).
            corrupted:   Observed (corrupted) image, shape (B, C, H, W).
                         Unknown pixels are set to 0.
            mask:        Binary mask, shape (B, 1, H, W).
                         1 = observed pixel, 0 = missing pixel.
            num_steps:   Reverse steps (default: self.T).
            device:      Target device.

        Returns:
            Reconstructed images, shape (B, C, H, W).
        """
        num_steps = num_steps or self.T
        # TODO (EC 5.D)
        raise NotImplementedError
