import drjit as dr
import mitsuba as mi
import numpy as np

from constants import SDF_DEFAULT_KEY_P, SDF_DEFAULT_KEY
from nerfstudio.utils.mi_util import render_aggregate, forward_grad_aggregate


def mi_create_sphere_sdf(res, radius: mi.Float, center=[0.5, 0.5, 0.5]):
    z, y, x = np.meshgrid(np.linspace(0, 1, res[0]), np.linspace(
        0, 1, res[1]), np.linspace(0, 1, res[2]), indexing='ij')
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    dist2origin = np.linalg.norm(pts - center, axis=-1)
    sdf = np.reshape(dist2origin, res).astype(np.float32)
    return mi.TensorXf(sdf) - radius


def mi_create_cube_sdf(res, side_length, center=[0.5, 0.5, 0.5]):
    side_length = np.array(side_length)
    center = np.array(center)
    z, y, x = np.meshgrid(np.linspace(0, 1, res[0]), np.linspace(
        0, 1, res[1]), np.linspace(0, 1, res[2]), indexing='ij')
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) - center
    q = np.abs(pts) - side_length * 0.5
    sdf = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.)
    sdf = np.reshape(sdf, res).astype(np.float32)
    return mi.TensorXf(sdf)


def eval_forward_gradient(scene, config, axis='x', spp=1024, fd_spp=8192, fd_eps=1e-3, sensor=0, spp_per_batch=1024,
                          target_value=0.3, seed=0, power_of_two=False):
    """Evalutes a forward gradient image for a given axis"""
    sdf = scene.integrator().sdf
    ROUGHNESS_KEY = 'principled-bsdf.roughness.volume.data'
    params = mi.traverse(scene)
    if axis in ['x', 'y', 'z']:
        params.keep([SDF_DEFAULT_KEY_P])
    elif axis in ['r']:
        params.keep([SDF_DEFAULT_KEY])
    elif axis in ['rho']:
        params.keep([ROUGHNESS_KEY])

    def forward_xyz():
        dr.eval(params[SDF_DEFAULT_KEY_P])

    forward_dict = forward_xyz
    del forward_xyz

    if axis == 'x':
        param = params[SDF_DEFAULT_KEY_P].x
    elif axis == 'y':
        param = params[SDF_DEFAULT_KEY_P].y
    elif axis == 'z':
        param = params[SDF_DEFAULT_KEY_P].z
    elif axis == 'r':
        param = mi.Float(target_value)

        def forward_r():
            params[SDF_DEFAULT_KEY] = mi_create_sphere_sdf(dr.shape(params[SDF_DEFAULT_KEY]), param)
            params.update()

        forward_dict = forward_r
        del forward_r
    elif axis == 'rho':
        param = mi.Float(target_value)

        def forward_rho():
            params[ROUGHNESS_KEY] = dr.full(mi.TensorXf, 0., dr.shape(params[ROUGHNESS_KEY])) + param
            params.update()

        forward_dict = forward_rho
        del forward_rho
    else:
        raise ValueError(f'Unknown axis {axis}')

    dr.enable_grad(param)
    dr.set_grad(param, 0.0)
    dr.kernel_history_clear()
    if config.use_finite_differences:
        with dr.suspend_grad():
            forward_dict()
            img = render_aggregate(scene, integrator=scene.integrator(), sensor=sensor, spp=fd_spp,
                                   spp_per_batch=spp_per_batch, seed=seed, power_of_two=power_of_two)
            param += fd_eps
            forward_dict()
            img2 = render_aggregate(scene, integrator=scene.integrator(), sensor=sensor, spp=fd_spp,
                                    spp_per_batch=spp_per_batch, seed=seed + 1, power_of_two=power_of_two)
            grad = (img2 - img) / fd_eps
            dr.eval(grad)
    else:
        scene.integrator().warp_field = config.get_warpfield(sdf)
        img, grad = forward_grad_aggregate(scene, param, spp, spp_per_batch, set_value=forward_dict,
                                           params=params, sensor=sensor, integrator=scene.integrator(), seed=seed,
                                           power_of_two=power_of_two)

    history = dr.kernel_history()
    total_time = sum(h['execution_time'] for h in history)
    stats = {'total_time': total_time}
    return img, grad, stats
