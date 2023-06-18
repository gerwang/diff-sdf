"""Contains logic to initialize, update and read/write optimized variables.

The code base currently only supports volumetric variables, but the interface
could also be used for other types of variables.
"""

import os
from typing import Literal

import drjit as dr
import mitsuba as mi
import numpy as np
from scipy.ndimage import zoom

import redistancing
from largesteps import CholeskySolver
from math_util import safe_exp
from shapes import BoxSDF, Grid3d, create_sphere_sdf
from util import atleast_4d


def upsample_sdf(sdf_data):
    new_res = 2 * mi.ScalarVector3i(sdf_data.shape[:3])
    return resample_sdf(sdf_data, new_res)


def resample_sdf(sdf_data, new_res):
    sdf = Grid3d(sdf_data)
    z, y, x = dr.meshgrid(*[(dr.arange(mi.Int32, new_res[i]) + 0.5) / new_res[i] for i in range(3)], indexing='ij')
    sdf = type(sdf_data)(dr.detach(sdf.eval(mi.Point3f(x, y, z))), new_res)
    return atleast_4d(sdf)


def upsample_grid(data):
    return dr.upsample(mi.Texture3f(data, migrate=False), scale_factor=[2, 2, 2, 1]).tensor()


def resample_grid_complex(data, new_res):
    """
    A complex way to resample grid, deprecated
    """
    grid = mi.Texture3f(data, migrate=False)
    z, y, x = dr.meshgrid(*[(dr.arange(mi.Int32, new_res[i]) + 0.5) / new_res[i] for i in range(3)], indexing='ij')
    eval_res = dr.detach(grid.eval(mi.Point3f(x, y, z)))
    res = dr.empty(mi.TensorXf, (*new_res, len(eval_res)))
    for i in range(len(eval_res)):
        res[..., i] = eval_res[i]
    data = type(data)(res)
    return atleast_4d(data)


def resample_grid(data, new_res):
    target_shape = (*new_res, *data.shape[3:])
    data = dr.upsample(data, shape=target_shape)
    return atleast_4d(data)


def simple_lr_decay(initial_lrate, decay, i):
    lr = initial_lrate / (1 + decay * i)

    # Hardcoded for now: further decay LR as target (512 iterations) is reached
    if i > 480:
        lr = lr / 2
    if i > 500:
        lr = lr / 2
    return lr


class Variable:
    """Represents a variable in an optimization that can be initialized, updated and saved"""

    def __init__(self, k, beta=None, regularizer_weight=0.0, regularizer=None, lr=None, adaptive_learning_rate=True,
                 upsample_regularizer_weight=False):
        self.k = k
        self.mean = None
        self.beta = beta
        self.regularizer_weight = regularizer_weight
        self.regularizer = regularizer
        self.upsample_regularizer_weight = upsample_regularizer_weight
        self.lr = lr
        self.adaptive_learning_rate = adaptive_learning_rate
        self.lr_decay_rate = 0.02
        self.warm_up_end = 10
        self.neus_alpha = 0.05

    def initialize(self, opt):
        return

    def save(self, opt, output_dir, suffix):
        return

    def restore(self, opt, output_dir, suffix):
        return

    def load(self, opt, mi_value, i):
        return

    def export(self, opt):
        return opt[self.k]

    def validate_gradient(self, opt, i):
        return

    def validate(self, opt, i):
        return False

    def update_mean(self, opt, i):
        return

    def load_mean(self, opt):
        if self.mean is not None:
            opt[self.k] = self.mean
            dr.enable_grad(opt[self.k])

    def eval_regularizer(self, opt, sdf_object, i):
        if self.regularizer is not None:
            return self.regularizer_weight * self.regularizer(opt[self.k])
        else:
            return mi.Float(0.0)

    def update_loss_dict(self, opt, loss_dict):
        return

    def load_data(self, opt, data, adaptive_resolution=True):
        return

    def get_param(self, opt):
        return opt[self.k]


class VolumeVariable(Variable):
    def __init__(self, k, shape, init_value=0.5, upsample_iter=[64, 128], step_lrs=[], step_regs=[], clamp_min=None,
                 **kwargs):
        super().__init__(k, **kwargs)
        self.shape = np.array(shape)
        self.init_value = init_value
        self.upsample_iter = upsample_iter
        if self.upsample_iter is not None:
            self.upsample_iter = list(self.upsample_iter)
            for i in range(3):
                self.shape[i] = self.shape[i] // 2 ** len(self.upsample_iter)
        len_upsample_iter = len(self.upsample_iter) if self.upsample_iter is not None else 0
        self.step_lrs = step_lrs[:len_upsample_iter]
        self.step_regs = step_regs[:len_upsample_iter]
        self.use_step_lr = len(self.step_lrs) > 0
        self.use_step_reg = len(self.step_regs) > 0
        if self.use_step_lr:
            self.step_lrs = [self.lr] + self.step_lrs
        if self.use_step_reg:
            self.step_regs = [self.regularizer_weight] + self.step_regs
        self.clamp_min = clamp_min

    def initialize(self, opt):
        self.initial_lr = opt.lr[self.k]
        opt[self.k] = dr.full(mi.TensorXf, self.init_value, self.shape)

        if self.lr is not None:
            opt.set_learning_rate({self.k: self.lr})

    def get_variable_path(self, output_dir, suffix, suffix2=''):
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{self.k.replace(".", "-")}-{suffix_str}{suffix2}.vol')

    def save(self, opt, output_dir, suffix):
        mi.VolumeGrid(np.array(opt[self.k])).write(self.get_variable_path(output_dir, suffix))

    def restore(self, opt, output_dir, suffix):
        loaded_data = np.array(mi.VolumeGrid(self.get_variable_path(output_dir, suffix)))
        if loaded_data.shape != opt[self.k].shape:
            loaded_data = zoom(loaded_data, (x / y for x, y in zip(opt[self.k].shape, loaded_data.shape)))
        # Make sure the number of dimensions matches up
        if self.k in opt and (opt[self.k].ndim == 4 and loaded_data.ndim == 3):
            loaded_data = loaded_data[..., None]
        opt[self.k] = mi.TensorXf(loaded_data)
        dr.enable_grad(opt[self.k])

    def load(self, opt, mi_value, i):
        k = self.k
        opt[k] = atleast_4d(mi_value)
        self.shape = opt[k].shape
        dr.enable_grad(opt[k])

    def validate(self, opt, i):
        k = self.k
        upsampled = False
        if self.upsample_iter is not None and i is not None and i + 1 in self.upsample_iter:
            opt[k] = mi.TensorXf(upsample_grid(opt[k]))
            upsampled = True
            if self.upsample_regularizer_weight:
                self.regularizer_weight *= 2

        if self.use_step_reg and i is not None:
            reg_idx = 0
            while reg_idx + 1 < len(self.step_regs) and self.upsample_iter[reg_idx] <= i + 1:
                reg_idx += 1
            self.regularizer_weight = self.step_regs[reg_idx]

        clamp_min = 0.
        if self.clamp_min is not None:
            clamp_min = self.clamp_min
        elif k.endswith('reflectance.volume.data') or k.endswith('base_color.volume.data'):
            clamp_min = 1e-5
        elif k.endswith('roughness.volume.data'):
            clamp_min = 0.1

        if k.endswith('reflectance.volume.data') or k.endswith('base_color.volume.data'):
            opt[k] = dr.clamp(opt[k], clamp_min, 1.0)
        if k.endswith('roughness.volume.data'):
            opt[k] = dr.clamp(opt[k], clamp_min, 0.8)

        if self.adaptive_learning_rate and i is not None:
            if self.use_step_lr:
                lr_idx = 0
                while lr_idx + 1 < len(self.step_lrs) and self.upsample_iter[lr_idx] <= i + 1:
                    lr_idx += 1
                lr = self.step_lrs[lr_idx]
            else:
                lr = simple_lr_decay(self.initial_lr, self.lr_decay_rate, i)
            opt.set_learning_rate({k: lr})

        dr.enable_grad(opt[k])
        return upsampled

    def update_mean(self, opt, i):
        if self.beta is None:
            return

        if self.mean is None or (opt[self.k].shape != self.mean.shape):
            self.mean = dr.detach(mi.TensorXf(opt[self.k]), True)
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * dr.detach(opt[self.k], True)

        # This is crucial to prevent enoki from building a
        # graph across iterations, leading to growth in memory use
        dr.schedule(self.mean)

    def load_data(self, opt, data, adaptive_resolution=True):
        k = self.k
        data = atleast_4d(mi.TensorXf(data))
        if adaptive_resolution and data.shape != opt[k].shape:
            data = resample_grid(data, opt[k].shape[:3])
        opt[k] = atleast_4d(mi.TensorXf(data))
        self.validate(opt, i=None)


class SdfVariable(VolumeVariable):

    def __init__(self, k, resolution,
                 sdf_init_fn=create_sphere_sdf,
                 redistance_freq=1,
                 redistance_by_scale=False,
                 bbox_constraint=True,
                 **kwargs,
                 ):
        super().__init__(k, shape=(resolution,) * 3, **kwargs)
        self.bbox_constraint = bbox_constraint
        self.sdf_init_fn = sdf_init_fn
        self.redistance_freq = redistance_freq
        self.redistance_by_scale = redistance_by_scale
        if self.bbox_constraint:
            self.update_box_sdf(self.shape)

    def initialize(self, opt):
        self.initial_lr = opt.lr[self.k]
        self.initial_shape = self.shape
        opt[self.k] = atleast_4d(mi.TensorXf(self.sdf_init_fn(self.shape)))
        if self.lr is not None:
            opt.set_learning_rate({self.k: self.lr})

    # Overload variable path to not use "SamplingIntegrator" as a prefix
    def get_variable_path(self, output_dir, suffix, suffix2=''):
        k = self.k.replace("SamplingIntegrator.", "")
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{k.replace(".", "-")}-{suffix_str}{suffix2}.vol')

    def update_box_sdf(self, res):
        bbox = BoxSDF(mi.Point3f(0), mi.Vector3f(0.49), smoothing=0.01)
        z, y, x = dr.meshgrid(dr.linspace(mi.Float, -0.5, 0.5, res[0]),
                              dr.linspace(mi.Float, -0.5, 0.5, res[1]),
                              dr.linspace(mi.Float, -0.5, 0.5, res[2]), indexing='ij')
        self.bbox_sdf = atleast_4d(mi.TensorXf(bbox.eval(mi.Point3f(x, y, z)), res))

    def validate(self, opt, i):
        k = self.k
        if self.upsample_iter is not None and i is not None and i + 1 in self.upsample_iter:
            sdf = upsample_sdf(opt[k])
            self.shape = sdf.shape
            if self.bbox_constraint:
                self.update_box_sdf(self.shape)
            if self.upsample_regularizer_weight:
                self.regularizer_weight *= 2
            just_upsampled = True
        else:
            self.shape = opt[k].shape
            sdf = opt[k]
            just_upsampled = False

        if self.use_step_reg and i is not None:
            reg_idx = 0
            while reg_idx + 1 < len(self.step_regs) and self.upsample_iter[reg_idx] <= i + 1:
                reg_idx += 1
            self.regularizer_weight = self.step_regs[reg_idx]

        if self.adaptive_learning_rate and i is not None:
            if self.use_step_lr:
                lr_idx = 0
                while lr_idx + 1 < len(self.step_lrs) and self.upsample_iter[lr_idx] <= i + 1:
                    lr_idx += 1
                lr = self.step_lrs[lr_idx]
            else:
                # Scale LR according to SDF res compared to a "32x32x32" "reference" case
                lr_scale = 32 / self.shape[0]
                lr = lr_scale * simple_lr_decay(self.initial_lr, self.lr_decay_rate, i)
            opt.set_learning_rate({k: lr})

        if self.bbox_constraint:
            assert sdf.shape == self.bbox_sdf.shape
            sdf = dr.maximum(sdf, self.bbox_sdf)

        redistance_freq = self.redistance_freq if not self.redistance_by_scale \
            else self.redistance_freq * self.shape[0] // 16
        if i is None or just_upsampled or i % redistance_freq == 0:
            sdf = redistancing.redistance(sdf)
        opt[k] = atleast_4d(mi.TensorXf(sdf))
        dr.enable_grad(opt[k])
        return just_upsampled

    def validate_gradient(self, opt, i):
        k = self.k
        grad = dr.grad(opt[k])

        # Clamp gradients and suppress NaNs just in case
        r = 1e-1
        dr.set_grad(opt[k], dr.select(dr.isnan(grad), 0.0, dr.clamp(grad, -r, r)))

    def eval_regularizer(self, opt, sdf_object, i):
        if self.regularizer is not None and self.regularizer_weight > 0.0:
            return self.regularizer_weight * self.regularizer(opt[self.k], sdf_object)
        else:
            return mi.Float(0.0)

    def load(self, opt, mi_value, i):
        super().load(opt, mi_value, i)
        k = self.k

        if self.bbox_constraint:
            self.update_box_sdf(self.shape)

        if self.adaptive_learning_rate and i is not None:
            # Scale LR according to SDF res compared to a "32x32x32" "reference" case
            lr_scale = 32 / self.shape[0]
            lr = lr_scale * simple_lr_decay(self.initial_lr, self.lr_decay_rate, i)
            opt.set_learning_rate({k: lr})

    def load_data(self, opt, data, adaptive_resolution=True):
        k = self.k
        data = atleast_4d(mi.TensorXf(data))
        if adaptive_resolution and data.shape != opt[k].shape:
            data = resample_sdf(data, opt[k].shape[:3])
        opt[k] = atleast_4d(mi.TensorXf(data))
        if not adaptive_resolution and self.bbox_constraint:
            self.update_box_sdf(data.shape)
        self.validate(opt, i=None)


class HeightVariable(Variable):
    def __init__(self, k, axis: Literal['x', 'y', 'z'] = 'y', init_value=0.0, **kwargs):
        super().__init__(k, **kwargs)
        self.axis = axis
        self.init_value = init_value

    def initialize(self, opt):
        opt[self.k] = mi.Vector3f(self.init_value)
        if self.lr is not None:
            opt.set_learning_rate({self.k: self.lr})

    def get_variable_path(self, output_dir, suffix, suffix2=''):
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{self.k.replace(".", "-")}-{suffix_str}{suffix2}.txt')

    def save(self, opt, output_dir, suffix):
        np.savetxt(self.get_variable_path(output_dir, suffix), np.array(dr.ravel(opt[self.k])))

    def restore(self, opt, output_dir, suffix):
        loaded_data = np.loadtxt(self.get_variable_path(output_dir, suffix))
        opt[self.k] = dr.unravel(mi.Vector3f, mi.TensorXf(loaded_data))
        dr.enable_grad(opt[self.k])

    def validate_gradient(self, opt, i):
        k = self.k
        grad = dr.grad(opt[k])

        for i, axis in enumerate('xyz'):
            if axis not in self.axis:
                grad[i] = 0.

        dr.set_grad(opt[k], dr.select(dr.isnan(grad), 0.0, grad))

    def validate(self, opt, i):
        k = self.k

        # opt[k] = dr.clamp(opt[k], -0.5, 0.5)  # hard coded here to prevent y go out of camera's view

        dr.enable_grad(opt[k])
        return False

    def update_loss_dict(self, opt, loss_dict):
        loss_dict['height'] = float(opt[self.k][['x', 'y', 'z'].index(self.axis), 0])


class EnvmapVariable(Variable):
    def __init__(self, k, height=256, init_value=2.5, lambda_=29.0, **kwargs):
        super().__init__(k, **kwargs)
        self.height = height
        self.width = self.height * 2
        self.init_value = init_value
        self.lambda_ = lambda_
        self.largesteps = CholeskySolver(dr.full(mi.TensorXf, self.init_value, (self.height, self.width, 3)),
                                         self.lambda_, is_envmap=True)
        row_indices, col_indices = np.indices((self.height, self.width))
        self.original_indices = mi.UInt32((row_indices * (self.width + 1) + col_indices).flatten())
        row_indices, col_indices = np.indices((self.height, 1))
        self.first_column_indices = mi.UInt32((row_indices * (self.width + 1) + col_indices).flatten())
        self.last_column_indices = mi.UInt32((row_indices * (self.width + 1) + col_indices).flatten() + self.width)

    def initialize(self, opt):
        opt[self.k] = dr.full(mi.TensorXf, self.init_value, (self.height, self.width, 3))
        if self.lr is not None:
            opt.set_learning_rate({self.k: self.lr})

    def get_variable_path(self, output_dir, suffix, suffix2=''):
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{self.k.replace(".", "-")}-{suffix_str}{suffix2}.exr')

    def save(self, opt, output_dir, suffix):
        mi.Bitmap(opt[self.k]).write(self.get_variable_path(output_dir, suffix))

    def restore(self, opt, output_dir, suffix):
        loaded_data = mi.Bitmap(self.get_variable_path(output_dir, suffix))
        opt[self.k] = mi.TensorXf(loaded_data)
        dr.enable_grad(opt[self.k])

    def validate_gradient(self, opt, i):
        k = self.k
        if self.largesteps is not None:
            dr.set_grad(opt[k], self.largesteps.precondition(dr.grad(opt[k])))

    def get_param(self, opt):
        k = self.k
        result = dr.empty(mi.TensorXf, (self.height, self.width + 1, 3))
        hdr_radiance = safe_exp(opt[k])

        v0 = dr.gather(dr.cuda.ad.Array3f, hdr_radiance.array, self.first_column_indices)
        v = dr.unravel(dr.cuda.ad.Array3f, hdr_radiance.array)
        dr.scatter(result.array, v, self.original_indices)
        dr.scatter(result.array, v0, self.last_column_indices)

        return result
