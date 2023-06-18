from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Tuple, Optional

import drjit as dr
import torch
import torch.distributed as dist

from emitters.util import affine_left
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.rotater import Rotater
from nerfstudio.utils import comms
from nerfstudio.utils.comms import ASYNC_OP
from nerfstudio.utils.mi_gl_conversion import torch_point_scale_shifted_mi2gl


# Return whether the input argument is a PyTorch tensor
def is_torch_tensor(a):
    return getattr(a, '__module__', None) == 'torch' \
        and type(a).__name__ == 'Tensor'


# Casting routing from Dr.Jit tensors to PyTorch tensors
def drjit_to_torch(a, enable_grad=False):
    if isinstance(a, Sequence) and not isinstance(a, str):
        return tuple(drjit_to_torch(b, enable_grad) for b in a)
    elif isinstance(a, Mapping):
        return {k: drjit_to_torch(v, enable_grad) for k, v in a.items()}
    elif dr.is_array_v(a) and dr.is_tensor_v(a):
        b = a.torch()
        b.requires_grad = dr.grad_enabled(a) or (enable_grad and dr.is_diff_v(a))
        return b
    elif dr.is_diff_v(a) and a.IsFloat:
        raise TypeError("wrap_ad(): differential input arguments "
                        "should be Dr.Jit tensor!")
    else:
        return a


# Casting routing from PyTorch tensors to Dr.Jit tensors
def torch_to_drjit(a):
    if isinstance(a, Sequence) and not isinstance(a, str):
        return tuple(torch_to_drjit(b) for b in a)
    elif isinstance(a, Mapping):
        return {k: torch_to_drjit(v) for k, v in a.items()}
    elif is_torch_tensor(a):
        dtype_str = {
            torch.float: 'TensorXf',
            torch.float32: 'TensorXf',
            torch.float64: 'TensorXf64',
            torch.int32: 'TensorXi',
            torch.int: 'TensorXi',
            torch.int64: 'TensorXi64',
            torch.long: 'TensorXi64',
        }[a.dtype]
        device = 'cuda' if a.is_cuda else 'llvm'
        m = getattr(getattr(dr, device), 'ad')
        return getattr(m, dtype_str)(a)
    else:
        return a


# Ensure tensors in `a` have same shape as tensors in `b` (handles dim==0 case)
def torch_ensure_shape(a, b):
    if isinstance(a, Sequence) and not isinstance(a, str):
        return tuple(torch_ensure_shape(a[i], b[i]) for i in range(len(a)))
    elif isinstance(a, Mapping):
        return {k: torch_ensure_shape(v, b[k]) for k, v in a.items()}
    elif is_torch_tensor(a):
        return a.reshape(b.shape)
    else:
        return a


def get_ray_bundle(o: torch.Tensor, v: torch.Tensor, torch_mi2gl_left: torch.Tensor, scene_scale: float,
                   camera_index: int, rotater: Optional[Rotater] = None) -> RayBundle:
    o = torch_point_scale_shifted_mi2gl(o, scene_scale)
    o = affine_left(torch_mi2gl_left, o)
    v = affine_left(torch_mi2gl_left, v, is_pos=False)
    ray_bundle = RayBundle(
        origins=o,
        directions=v,
        pixel_area=torch.ones_like(o[..., :1]),
    )
    ray_bundle.set_camera_indices(camera_index)
    if rotater is not None:
        ray_bundle.rotater = rotater.apply_frustums
    return ray_bundle


def get_target_sizes(total_size):
    cnt, rem = divmod(total_size, comms.get_world_size())
    res = [cnt + 1 if i < rem else cnt for i in range(comms.get_world_size())]
    return res


def pad_gather(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    pad_n = (n + comms.get_world_size() - 1) // comms.get_world_size()

    if x.shape[dim] < pad_n:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_n - x.shape[dim]
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)

    if comms.is_main_process():
        gather_shape = list(x.shape)
        gather_shape[dim] = pad_n
        gather_list = [torch.empty(gather_shape, device=x.device) for _ in range(comms.get_world_size())]
        dist.gather(x, gather_list, async_op=ASYNC_OP)

        target_sizes = get_target_sizes(n)
        for i in range(len(gather_list)):
            if target_sizes[i] != pad_n:
                gather_list[i], _ = torch.split(
                    gather_list[i], [target_sizes[i], pad_n - target_sizes[i]], dim=dim)
        x = torch.cat(gather_list, dim=dim)
    else:
        dist.gather(x, async_op=ASYNC_OP)

    return x


def pad_scatter(send_shape: Tuple, device: torch.device, dim: int,
                send_payload: torch.Tensor | None = None) -> torch.Tensor:
    pad_n = (send_shape[dim] + comms.get_world_size() - 1) // comms.get_world_size()

    recv_shape = list(send_shape)
    recv_shape[dim] = pad_n
    recv_payload = torch.empty(recv_shape, device=device)

    if comms.is_main_process():
        target_sizes = get_target_sizes(send_shape[dim])
        scatter_list = list(torch.split(send_payload, target_sizes, dim=dim))
        for i in range(len(scatter_list)):
            if scatter_list[i].shape[dim] != pad_n:
                pad_shape = list(scatter_list[i].shape)
                pad_shape[dim] = pad_n - scatter_list[i].shape[dim]
                scatter_list[i] = torch.cat([scatter_list[i], scatter_list[i].new_zeros(pad_shape)], dim=dim)
        dist.scatter(recv_payload, scatter_list, async_op=ASYNC_OP)
    else:
        dist.scatter(recv_payload, async_op=ASYNC_OP)

    target_size = get_target_size_by_rank(send_shape[dim])
    if target_size < pad_n:
        recv_payload, _ = torch.split(recv_payload, [target_size, pad_n - target_size], dim=dim)
    return recv_payload


def get_target_size_by_rank(total_size):
    cnt, rem = divmod(total_size, comms.get_world_size())
    cnt += 1 if comms.get_local_rank() < rem else 0
    return cnt


def scatter_camera_idx(camera_idx: int = -1) -> int:
    recv_shape = (1,)
    device = f'cuda:{comms.get_local_rank()}'
    recv_payload = torch.empty(recv_shape, device=device, dtype=torch.int64)
    if comms.is_main_process():
        scatter_list = [torch.full(size=recv_shape, fill_value=camera_idx, device=device, dtype=torch.int64)
                        for _ in range(comms.get_world_size())]
        dist.scatter(recv_payload, scatter_list, async_op=ASYNC_OP)
    else:
        dist.scatter(recv_payload, async_op=ASYNC_OP)
    return recv_payload.item()
