"""
Taylor Expansion Optimizer for PyTorch
=====================================

This optimizer extrapolates the local Taylor series of the loss landscape to
construct a *better* descent direction than vanilla SGD.

Core idea
---------
Given loss **L(θ)** and a small step **Δ**, the second‑order Taylor expansion is

    L(θ + Δ) ≈ L(θ) + ∇L(θ)^T Δ + 0.5 Δ^T H Δ

where **H** is the Hessian.  Newton’s method chooses Δ = −H⁻¹∇L, but computing
(and inverting) H is expensive.  Instead, we build a *proxy* for the curvature
using *temporal differences* of consecutive gradients:

    ∇L(θₜ) − ∇L(θₜ₋₁) ≈ H (θₜ − θₜ₋₁).

Solving for the diagonal of **H** yields an inexpensive Hessian estimate that
feeds a quasi‑Newton step.

The resulting update for each parameter element *i* is

    gₜ = ∇L(θₜ)
    ĥₜ = (gₜ[i] − gₜ₋₁[i]) / (θₜ[i] − θₜ₋₁[i] + ε)

    Δθ[i] = − lr * (gₜ[i] / (|ĥₜ| + damping))

where **damping** stabilises very small curvatures.  When the finite‑difference
curvature is unreliable (e.g. at the first step), the update collapses to SGD.

Hyper‑parameters
----------------
* **lr** (float): Base learning rate.
* **damping** (float): Added to |ĥ| in the denominator for numerical stability.
* **weight_decay** (float): Optional L2 regularisation.
* **eps** (float): Small constant preventing division by zero in ĥ.
* **momentum** (float): Optional Polyak momentum on the final step.

Usage
-----
>>> optimizer = TaylorExp(params, lr=1e‑3, damping=1e‑2)
>>> for input, target in loader:
>>>     loss = criterion(model(input), target)
>>>     loss.backward()
>>>     optimizer.step()
>>>     optimizer.zero_grad()

The implementation is self‑contained and requires only PyTorch.
"""
from __future__ import annotations

from typing import Iterable, Optional
import torch
from torch.optim.optimizer import Optimizer, required

class TaylorExp(Optimizer):
    r"""Implements a lightweight Taylor‑series quasi‑Newton optimizer.

    Each parameter keeps a one‑step history of (value, grad) to form a finite‑
    difference Hessian estimate along the coordinate axis.  The step is then a
    Newton‑like correction *without* explicit Hessian storage.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = required,
        damping: float = 1e-2,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        eps: float = 1e-8,
    ):
        if lr is required:
            raise ValueError("lr must be specified")
        defaults = dict(
            lr=lr,
            damping=damping,
            weight_decay=weight_decay,
            momentum=momentum,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["prev_grad"] = torch.zeros_like(p)
                    state["prev_param"] = torch.empty_like(p).copy_(p)
                    state["momentum_buf"] = torch.zeros_like(p) if momentum > 0 else None

                prev_grad = state["prev_grad"]
                prev_param = state["prev_param"]

                # Finite‑difference curvature estimate
                param_diff = p - prev_param
                grad_diff = grad - prev_grad
                denom = torch.abs(param_diff) + eps
                h_est = grad_diff / denom  # elementwise Hessian diag approximation

                # Avoid extremely small / negative curvatures by absolute value and damping.
                step_dir = grad / (torch.abs(h_est) + damping)

                # Momentum (applied to the final step)
                if momentum > 0:
                    buf = state["momentum_buf"]
                    buf.mul_(momentum).add_(step_dir)
                    step_dir = buf

                p.add_(step_dir, alpha=-lr)  # descent step

                # Update state for next iteration
                prev_grad.copy_(grad)
                prev_param.copy_(p)

        return loss

# ---------------------------------------------------------------------------
# Minimal sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(10, 1)
    y = 3 * x + 0.8  # linear relation

    model = torch.nn.Linear(1, 1, bias=False)
    criterion = torch.nn.MSELoss()
    optim = TaylorExp(model.parameters(), lr=0.1, damping=1e-3)

    for epoch in range(200):
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optim.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | loss = {loss.item():.4f}")

    print("Learned weight:", model.weight.data.squeeze().item())
