import io
from typing import Optional

import drjit as dr
import mitsuba as mi
import numpy as np

from emitters.envmap import MyEnvironmentMapEmitter, indent
from util import dr_no_jit


def vector3f_2_tensor(o):
    o_tensor = dr.empty(mi.TensorXf, shape=dr.shape(o))
    o_tensor[0] = o.x
    o_tensor[1] = o.y
    o_tensor[2] = o.z
    return o_tensor


class NeRFEmitter(MyEnvironmentMapEmitter):
    def __init__(self, props: mi.Properties):
        """
        What do I need:
            load_config: str, file path of nerf model
            bounded_radius: float, the bounding sphere of bounded scene, centered at (0, 0, 0)
            ..., some options for experiments
        Args:
            props:
        """
        super().__init__(props)

        self.scene: Optional[mi.Scene] = None
        padding_size = float(props.get('padding_size', 0.0))
        self.bounding_box = mi.BoundingBox3f(mi.Point3f(0.0 - padding_size), mi.Point3f(1.0 + padding_size))
        self.skip_distance = float(props.get('skip_distance', 0.0))
        self.outside_only = bool(props.get('outside_only', False))
        self.dummy_nerf_parameter = mi.TensorXf(mi.Float(np.nan))  # dummy value
        self.nerf_emitter_op = None
        self.detach_op = bool(props.get('detach_op', False))
        self.guiding_weight = None
        self.camera_idx = 0

    def set_guiding_weight(self, guiding_weight):
        self.guiding_weight = guiding_weight

    def set_scene(self, scene: mi.Scene) -> None:
        super().set_scene(scene)
        self.scene = scene

    def set_op(self, op):
        self.nerf_emitter_op = op

    def set_camera_idx(self, camera_idx):
        self.camera_idx = camera_idx

    def eval(self, si: mi.SurfaceInteraction3f, active: bool = True) -> mi.Color3f:
        return self.eval_nerf(si.p, -si.wi, active)

    def eval_nerf_ray(self, o: mi.Point3f, v: mi.Vector3f, near: mi.Float, active: bool = True) -> mi.Color3f:
        o_tensor = vector3f_2_tensor(o)
        v_tensor = vector3f_2_tensor(v)
        res_tensor = self.nerf_emitter_op(o_tensor, v_tensor, near, self.dummy_nerf_parameter, self.guiding_weight,
                                          self.camera_idx)
        if self.detach_op:
            res_tensor = dr.detach(res_tensor)
        res = dr.unravel(mi.Color3f, res_tensor, order='F')
        return res * self.m_scale

    def eval_nerf(self, o: mi.Point3f, v: mi.Vector3f, active: bool = True) -> mi.Color3f:
        world2cam = self.world_transform().inverse()
        tmp_ray = mi.Ray3f(o, v)
        tmp_ray = world2cam @ tmp_ray
        o, v = tmp_ray.o, tmp_ray.d
        ray = mi.Ray3f(o, v)
        if self.outside_only:
            inter_mask, sol_l, sol_h = self.bounding_box.ray_intersect(ray)
            sph_t = dr.select(inter_mask, sol_h, self.skip_distance)
        else:
            sph_t = self.skip_distance
        with dr_no_jit():
            dr.schedule(o, v, sph_t, self.guiding_weight)
            return self.eval_nerf_ray(o, v, sph_t, active)

    def sample_direction(self, it: mi.SurfaceInteraction3f, sample: mi.Point2f, active: bool = True):
        ds, _ = super().sample_direction(it, sample, active)
        return ds, mi.Spectrum(0.) & active

    def to_string(self):
        res = mi.ScalarVector2u(self.m_data.shape[1], self.m_data.shape[0])
        oss = io.StringIO()
        oss.write(f'NeRFEmitter[\n')
        if self.m_filename != '':
            oss.write(f'  filename = "{self.m_filename}",\n')
        oss.write(f'  res = "{res}",\n'
                  f'  bsphere = {indent(str(self.m_bsphere))},\n')
        oss.write(f']')
        return oss.getvalue()

    def traverse(self, callback: mi.TraversalCallback) -> None:
        super().traverse(callback)
        callback.put_parameter('dummy_nerf_parameter', self.dummy_nerf_parameter, flags=mi.ParamFlags.Differentiable)
