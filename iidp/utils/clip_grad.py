import torch

from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm_for_overlap(gradients: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of gradients.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        gradients (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the gradients (viewed as a single vector).
    """
    if isinstance(gradients, torch.Tensor):
        gradients = [gradients]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(gradients) == 0:
        return torch.tensor(0.)
    device = gradients[0].device
    if norm_type == inf:
        norms = [grad.detach().abs().max().to(device) for grad in gradients]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), norm_type).to(device) for grad in gradients]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    # https://github.com/pytorch/pytorch/issues/60691
    # To avoid `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in gradients:
        grad.detach().mul_(clip_coef_clamped.to(grad.device))
    return total_norm