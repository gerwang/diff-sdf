from __future__ import annotations

from typing import Sequence, Mapping, Optional

import drjit as dr
import mitsuba as mi
import torch
from torch.autograd import forward_ad

from emitters.nerf_op import drjit_to_torch, torch_to_drjit, pad_scatter, get_ray_bundle, pad_gather, \
    torch_ensure_shape, is_torch_tensor, scatter_camera_idx
from nerfstudio.field_components.rotater import Rotater
from nerfstudio.path_guiding import PathGuiding
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import comms


def nerf_emitter_op(_o: mi.TensorXf, _v: mi.TensorXf, _near: mi.Float | float,
                    _dummy_nerf_parameter: mi.TensorXf, _guiding_weight: mi.Float, camera_idx: int,
                    pipeline: Pipeline, torch_mi2gl_left: torch.Tensor, scene_scale: float, path_guiding: PathGuiding,
                    rotater: Optional[Rotater] = None):
    if isinstance(_near, mi.Float):
        near = _near.torch().unsqueeze(-1)
    else:
        near = _near
    guiding_weight = _guiding_weight.torch().unsqueeze(-1)

    # change of variable, o' = o + near * v, do'/dx = do/dx + near * dv/dx

    class NeRFEmitterOp(dr.CustomOp):
        def eval(self, *args):
            self.args = args
            self.args_torch = drjit_to_torch(self.args)

            self.res_torch = self.ns_render_ray(*self.args_torch)
            return torch_to_drjit(self.res_torch.flatten().contiguous())

        @staticmethod
        @torch.no_grad()
        def ns_render_ray(o, v, _):
            guide_o = o.t().contiguous()
            guide_v = v.t().contiguous()
            __o = guide_o + near * guide_v
            __v = guide_v
            if comms.get_world_size() > 1:
                # dispatch o and v
                send_payload = torch.cat([__o, __v], dim=-1)
                recv_payload = pad_scatter((send_payload.shape[0], 6), o.device, 0, send_payload)
                o, v = torch.split(recv_payload, [3, 3], dim=-1)
                scatter_camera_idx(camera_idx)
            else:
                o, v = __o, __v
            ray_bundle = get_ray_bundle(o, v, torch_mi2gl_left, scene_scale, camera_idx, rotater)
            rgb = pipeline.model.get_rgb_for_camera_ray_bundle(ray_bundle)
            if comms.get_world_size() > 1:
                rgb = pad_gather(rgb, send_payload.shape[0], dim=0)
            path_guiding.train_primal(guide_o, guide_v, rgb, guiding_weight)
            return rgb

        @staticmethod
        @torch.no_grad()
        def ns_forward_ray(o, v, _, grad_o, grad_v, __):
            guide_o = o.t().contiguous()
            guide_v = v.t().contiguous()
            __o = guide_o + near * guide_v
            __v = guide_v
            __grad_o = grad_o.t().contiguous()
            __grad_v = grad_v.t().contiguous()
            __grad_o += near * __grad_v
            if comms.get_world_size() > 1:
                # dispatch o and v
                send_payload = torch.cat([__o, __v, __grad_o, __grad_v], dim=-1)
                recv_payload = pad_scatter((send_payload.shape[0], 12), o.device, 0, send_payload)
                o, v, grad_o, grad_v = torch.split(recv_payload, [3, 3, 3, 3], dim=-1)
                scatter_camera_idx(camera_idx)
            else:
                o, v, grad_o, grad_v = __o, __v, __grad_o, __grad_v
            with forward_ad.dual_level():
                o = forward_ad.make_dual(o, grad_o)
                v = forward_ad.make_dual(v, grad_v)
                ray_bundle = get_ray_bundle(o, v, torch_mi2gl_left, scene_scale, camera_idx, rotater)
                ray_bundle.origins, grad_o = forward_ad.unpack_dual(ray_bundle.origins)
                ray_bundle.directions, grad_v = forward_ad.unpack_dual(ray_bundle.directions)
            grad_rgb = pipeline.model.forward_grad_for_camera_ray_bundle(ray_bundle, grad_o, grad_v)
            if comms.get_world_size() > 1:
                grad_rgb = pad_gather(grad_rgb, send_payload.shape[0], dim=0)
            grad_rgb[torch.isnan(grad_rgb)] = 0.
            path_guiding.train_forward(guide_o, guide_v, grad_rgb, guiding_weight)
            return grad_rgb

        @staticmethod
        def ns_backward_ray(o, v, dummy, grad_out):
            guide_o = o.t().contiguous().detach()
            guide_v = v.t().contiguous().detach()
            __o = guide_o + near * guide_v
            __v = guide_v
            if comms.get_world_size() > 1:
                send_payload = torch.cat([__o, __v, grad_out], dim=-1)
                recv_payload = pad_scatter((send_payload.shape[0], 9), o.device, 0, send_payload)
                __o, __v, grad_out = torch.split(recv_payload, [3, 3, 3], dim=-1)
                scatter_camera_idx(camera_idx)
            __o.requires_grad = True
            __v.requires_grad = True
            ray_bundle = get_ray_bundle(__o, __v, torch_mi2gl_left, scene_scale, camera_idx, rotater)
            pipeline.backward_for_camera_ray_bundle(ray_bundle, grad_out)
            if comms.get_world_size() > 1:
                grad = torch.cat([__o.grad, __v.grad], dim=-1)
                grad = pad_gather(grad, send_payload.shape[0], dim=0)
                __o_grad, __v_grad = torch.split(grad, [3, 3], dim=-1)
            else:
                __o_grad, __v_grad = __o.grad, __v.grad
            __o_grad -= near * __v_grad

            o.grad = __o_grad.t().contiguous()
            v.grad = __v_grad.t().contiguous()
            dummy.grad = torch.zeros_like(dummy)
            path_guiding.train_backward(guide_o.detach(), guide_v.detach(), __o_grad, guiding_weight)

        def forward(self):
            grad_in_torch = drjit_to_torch(self.grad_in('args'))
            grad_in_torch = torch_ensure_shape(grad_in_torch, self.args_torch)

            grad_out_torch = self.ns_forward_ray(*self.args_torch, *grad_in_torch)

            grad_out = torch_to_drjit(grad_out_torch)
            self.set_grad_out(grad_out)

        def backward(self):
            grad_out_torch = drjit_to_torch(self.grad_out())
            grad_out_torch = torch_ensure_shape(grad_out_torch, self.res_torch)

            self.ns_backward_ray(*self.args_torch, grad_out_torch)

            def get_grads(args):
                if isinstance(args, Sequence) and not isinstance(args, str):
                    return tuple(get_grads(b) for b in args)
                elif isinstance(args, Mapping):
                    return {k: get_grads(v) for k, v in args.items()}
                elif is_torch_tensor(args):
                    return getattr(args, 'grad', None)
                else:
                    return None

            args_grad_torch = get_grads(self.args_torch)
            args_grad = torch_to_drjit(args_grad_torch)
            self.set_grad_in('args', args_grad)

    return dr.custom(NeRFEmitterOp, (_o, _v, _dummy_nerf_parameter))
