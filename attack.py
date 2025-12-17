from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("mean/std must be 1D tensors with shape [C]")
    if x.ndim != 4:
        raise ValueError("x must be a 4D tensor with shape [N, C, H, W]")
    mean = mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / std


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x_in = x.detach().clone().requires_grad_(True)
    logits = model(normalize_batch(x_in, mean, std))
    loss = F.cross_entropy(logits, y_true)
    grad = torch.autograd.grad(loss, x_in, only_inputs=True)[0]
    x_adv = x_in + eps * grad.sign()
    return x_adv.clamp(0.0, 1.0).detach()


def pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    eps: float,
    alpha: float,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    random_start: bool = True,
) -> torch.Tensor:
    x_orig = x.detach()
    if random_start:
        x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x_orig.clone()

    for _ in range(int(steps)):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = model(normalize_batch(x_adv, mean, std))
        loss = F.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass(frozen=True)
class CWResult:
    x_adv: torch.Tensor
    success: torch.Tensor


def cw_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 1000,
    lr: float = 1e-2,
    targeted: bool = False,
    y_target: Optional[torch.Tensor] = None,
) -> CWResult:
    if targeted and y_target is None:
        raise ValueError("y_target must be provided when targeted=True")

    x0 = x.detach().clamp(0.0, 1.0)
    eps = 1e-6
    x0_tanh = x0 * (1.0 - 2.0 * eps) + eps
    w0 = _atanh(x0_tanh * 2.0 - 1.0).detach()
    w = w0.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x0.clone()
    best_l2 = torch.full((x0.shape[0],), float("inf"), device=x0.device, dtype=x0.dtype)
    best_success = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.bool)

    y_cmp = y_target if targeted else y_true

    for _ in range(int(steps)):
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(normalize_batch(x_adv, mean, std))

        num_classes = logits.shape[1]
        y_onehot = F.one_hot(y_cmp, num_classes=num_classes).to(dtype=logits.dtype)

        real = (logits * y_onehot).sum(dim=1)
        other = (logits - 1e4 * y_onehot).amax(dim=1)

        if targeted:
            f = torch.clamp(other - real + kappa, min=0.0)
            success = (logits.argmax(dim=1) == y_cmp)
        else:
            f = torch.clamp(real - other + kappa, min=0.0)
            success = (logits.argmax(dim=1) != y_cmp)

        l2 = (x_adv - x0).view(x0.shape[0], -1).pow(2).sum(dim=1)
        loss = (l2 + c * f).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        improved = success & (l2 < best_l2)
        if improved.any():
            best_l2 = torch.where(improved, l2, best_l2)
            best_success = best_success | improved
            best_adv = torch.where(improved.view(-1, 1, 1, 1), x_adv.detach(), best_adv)

    final_adv = torch.where(best_success.view(-1, 1, 1, 1), best_adv, (0.5 * (torch.tanh(w) + 1.0)).detach())
    return CWResult(x_adv=final_adv, success=best_success)
