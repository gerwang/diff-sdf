import gc

import drjit as dr
import mitsuba as mi

from shapes import Grid3d
from warp import DummyWarpField


class ReparamSplitLightIntegrator(mi.SamplingIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 4)
        self.weight_by_spp = props.get('weight_by_spp', False)
        assert not self.weight_by_spp, "Not supported"
        self.use_mis = props.get('use_mis', False)
        self.force_optix = props.get('force_optix', False)
        self.antithetic_sampling = props.get('antithetic_sampling', False)
        self.use_bbox_sdf = props.get('use_bbox_sdf', False)
        props.mark_queried('query_emitter_index')
        props.mark_queried('curvature_epsilon')

        sdf_transform = None
        if props.has_property('sdf_to_world'):
            sdf_transform = props.get('sdf_to_world', mi.ScalarTransform4f())
            sdf_transform = mi.ScalarTransform4f(sdf_transform.matrix)

        sdf_filename = props.get('sdf_filename', '')
        if sdf_filename != '':
            self.sdf = Grid3d(sdf_filename, transform=sdf_transform, use_bbox=self.use_bbox_sdf)
        else:
            self.sdf = None

        self.warp_field = None
        self.sdf_shape = None
        self.is_prepared = False
        self.use_optix = True
        self.use_aovs = False
        self.emitter_scene = None
        self.adjoint_emitters = []

    def get_emitter_scene(self, scene):
        if self.emitter_scene is None:
            return scene
        return self.emitter_scene

    def prepare(self, sensor, seed, spp, aovs=[]):
        film = sensor.film()
        sampler = sensor.sampler().clone()
        if spp != 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()
        wavefront_size = dr.prod(film_size) * spp
        wavefront_size_limit = 0xffffffff if dr.is_llvm_v(mi.Float) else 0x40000000
        if wavefront_size > wavefront_size_limit:
            raise Exception(f"Wavefront {wavefront_size} exceeds {wavefront_size_limit}")
        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)
        return sampler, spp

    def prepare_sdf(self, scene):
        if self.is_prepared:
            return

        if self.sdf is None:
            self.is_prepared = True
            return

        # Disable optix if there is only a single shape that is an SDF
        # self.use_optix = self.force_optix or any(not s.is_sdf() for s in scene.shapes())

        # Enable optix if we detect that there is more than just one shape
        self.use_optix = self.force_optix or len(scene.shapes()) > 1
        # extract the dummy shape that holds the SDFs BSDF
        for idx, s in enumerate(scene.shapes()):
            if '_sdf_' in s.id():
                self.sdf_shape = s
                shape_idx = idx
                break
        if self.sdf_shape is None:
            raise ValueError("Scene is missing a dummy SDF shape that holds the BSDF")

        self.sdf_shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx)
        dr.eval(self.sdf_shape)
        self.is_prepared = True

    def eval_sample_one(self, mode, scene, sensor, sampler, _aovs, position_sample, diff_scale_factor,
                        active=mi.Bool(True)):
        aperture_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d(active)
        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d(active) * sensor.shutter_open_time()

        wavelength_sample = sampler.next_1d(active)
        crop_offset = sensor.film().crop_offset()
        crop_size = mi.Vector2f(sensor.film().crop_size())
        adjusted_position = (position_sample - crop_offset) / crop_size
        ray, ray_weight = sensor.sample_ray_differential(time, wavelength_sample, adjusted_position, aperture_sample)
        ray.scale_differential(diff_scale_factor)
        rgb, valid_ray, det, aovs_ = self.sample(mode, scene, sampler, ray,
                                                 None, None, None, active)
        p = mi.Point3f(aovs_[0], aovs_[1], aovs_[2])
        wi = mi.Point3f(aovs_[3], aovs_[4], aovs_[5])
        guiding_weight = aovs_[6]
        active_s = aovs_[7]
        aovs_ = aovs_[8:]
        return {
            'rgb': rgb,
            'valid_ray': valid_ray,
            'det': det,
            'aovs_': aovs_,
            'p': p,
            'wi': wi,
            'guiding_weight': guiding_weight,
            'active_s': active_s,
            'aperture_sample': aperture_sample,
            'ray': ray,
        }

    def eval_sample_two(self, sensor, block, _aovs, rgb, valid_ray, det, aovs_, aperture_sample, ray,
                        active=mi.Bool(True)):
        # Re-evaluate sample's sensor position and sensor importance
        it = dr.zeros(mi.Interaction3f)
        it.p = ray.o + ray.d
        ds, ray_weight = sensor.sample_direction(it, aperture_sample)
        ray_weight = dr.select(ray_weight > 0.0, ray_weight / dr.detach(ray_weight), 1.0)
        ray_weight = dr.replace_grad(type(ray_weight)(1.0), ray_weight)
        position_sample = ds.uv
        rgb = ray_weight * rgb

        has_alpha_channel = block.channel_count() == 5 + len(aovs_)
        aovs = [None] * 3
        aovs[0] = rgb.x
        aovs[1] = rgb.y
        aovs[2] = rgb.z
        if has_alpha_channel:
            aovs.append(valid_ray)
        aovs.append(dr.replace_grad(mi.Float(1.0), det * ray_weight[0]))

        aovs = aovs + aovs_
        block.put(position_sample, aovs, active)

    def render(self, scene, sensor=0, seed=0,
               spp=0, develop=True, evaluate=True, mode=dr.ADMode.Primal):
        var_dict = self.render_one(scene, sensor, seed, spp, develop, evaluate, mode)
        # dr.schedule(var_dict)  # evaluate all the variables produced by render_one to avoid rendering again
        self.evaluate_light(scene, var_dict)
        return self.render_two(var_dict)

    def render_one(self, scene, sensor=0, seed=0,
                   spp=0, develop=True, evaluate=True, mode=dr.ADMode.Primal):
        if not develop:
            raise Exception("Must use develop=True for this AD integrator")
        self.prepare_sdf(scene)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # prepare film and sampler
        sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names())

        # Logic to sample pixel indices
        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()
        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Avoid division if spp is a power of 2
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)
        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)
        if film.sample_border():
            pos -= border_size
        pos += mi.Vector2i(film.crop_offset())
        # Done generating pixel indices

        # Sample film positions and subsequently accumulate actual light paths
        block = sensor.film().create_block()
        _aovs = [None] * len(self.aov_names())
        diff_scale_factor = dr.rsqrt(mi.ScalarFloat(spp))

        # If we ask for AOVs, reparameterize also in forward pass to get values
        if self.warp_field is not None and self.warp_field.return_aovs:
            mode = dr.ADMode.Forward

        # Generate light paths, potentially apply antithetic sampling at the pixel itself
        # (disabled by default, but used in evaluation)
        active = mi.Bool(True)
        r = sampler.next_2d(active)
        position_sample = pos + r
        # TODO antithetic_sampling
        var_dict = self.eval_sample_one(mode, scene, sensor, sampler, _aovs, position_sample, diff_scale_factor, active)
        var_dict.update({
            'sensor': sensor,
            'block': block,
            '_aovs': _aovs,
            'active': active,
        })
        return var_dict

    def evaluate_light(self, scene, var_dict):
        p = var_dict['p']
        wi = var_dict['wi']
        guiding_weight = var_dict['guiding_weight']
        active_s = var_dict['active_s']
        del var_dict['p']
        del var_dict['wi']
        del var_dict['guiding_weight']
        del var_dict['active_s']
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.p = p
        si.wi = wi
        if hasattr(scene.environment(), 'set_guiding_weight'):
            scene.environment().set_guiding_weight(guiding_weight)
        emitter_val = scene.environment().eval(si)
        var_dict['rgb'] *= dr.select(active_s, emitter_val, mi.Color3f(0.))

    def render_two(self, var_dict):
        sensor = var_dict['sensor']
        del var_dict['sensor']
        block = var_dict['block']
        del var_dict['block']
        _aovs = var_dict['_aovs']
        del var_dict['_aovs']
        active = var_dict['active']
        del var_dict['active']
        self.eval_sample_two(sensor, block, _aovs, **var_dict, active=active)
        gc.collect()

        # Develop the film given the block
        sensor.film().put_block(block)
        primal_image = sensor.film().develop()
        return primal_image

    def render_backward(self, scene, params, grad_in, sensor=0, seed=0, spp=0):
        def backward(i, spp):
            image = self.render(scene=scene, sensor=sensor, seed=seed + i,
                                spp=spp, develop=True, evaluate=False, mode=dr.ADMode.Backward)
            dr.backward_from(image * grad_in)

        backward(0, spp)

    def render_forward(self, scene, params, sensor=0, seed=0, spp=0):
        def forward(i, spp, last):
            image = self.render(scene=scene, sensor=sensor, seed=seed + i, spp=spp,
                                develop=True, evaluate=False, mode=dr.ADMode.Forward)
            if last:
                flags = dr.ADFlag.Default
            else:
                flags = dr.ADFlag.ClearEdges | dr.ADFlag.ClearInterior
            dr.forward_to(image, flags=flags)
            return dr.grad(image)

        return forward(0, spp, last=True)

    def ray_test(self, scene, sampler, ray, depth=0, reparam=True, active=True):
        return self.ray_intersect(scene, sampler, ray, depth=depth, reparam=reparam, ray_test=True, active=active)

    def ray_intersect(self, scene, sampler, ray, depth=0, ray_test=False, reparam=True, active=True):
        """Intersects both SDFs and other scene objects if necessary"""

        si = dr.zeros(mi.SurfaceInteraction3f)
        si_d = dr.zeros(mi.SurfaceInteraction3f)
        div = mi.Float(1.0)
        extra_output = {}
        its_found = mi.Mask(False)
        if self.use_optix:
            if ray_test:
                its_found |= scene.ray_test(ray, active)
            else:
                si = scene.ray_intersect(ray, active)
                si_d = dr.detach(si)

        if self.sdf is not None:
            wf = self.warp_field if self.warp_field is not None else DummyWarpField(self.sdf)
            if ray_test:
                its_found2, div, extra_output = wf.ray_intersect(
                    self.sdf_shape, sampler, ray, depth=depth, ray_test=ray_test, reparam=reparam, active=active)
                its_found |= its_found2
            else:
                si2, si_d2, div, extra_output = wf.ray_intersect(
                    self.sdf_shape, sampler, ray, depth=depth, ray_test=ray_test, reparam=reparam, active=active)
                valid = (not self.use_optix) | (si2.t < si.t)
                si[valid] = si2
                si_d[valid] = si_d2

        if ray_test:
            return its_found, div, extra_output
        else:
            si.p = dr.select(si.is_valid(), si.p, ray.o)
            si_d.p = dr.select(si_d.is_valid(), si_d.p, dr.detach(ray.o))
            return si, si_d, div, extra_output

    def ray_intersect_preliminary(self, scene, ray, active=True):
        """Computes a preliminary shape intersection relatively efficiently"""
        pi = dr.zeros(mi.PreliminaryIntersection3f)
        if self.use_optix:
            pi = scene.ray_intersect_preliminary(ray, active)
        if self.sdf is not None:
            pi2 = self.sdf.ray_intersect_preliminary(ray, active)
            pi2.shape = self.sdf_shape
            valid = (not self.use_optix) | (pi2.t < pi.t)
            pi[valid] = pi2
        return pi

    def compute_surface_interaction(self, pi, ray, flags=mi.RayFlags.All):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = -ray.d

        if self.use_optix:
            # TODO: this needs to not invoke SDF?
            si = pi.compute_surface_interaction(ray, flags)

        if self.sdf is not None:
            si2 = self.sdf.compute_surface_interaction(ray, pi.t)
            si2.shape = self.sdf_shape
            if self.use_optix:
                si[dr.eq(self.sdf_shape, pi.shape)] = si2
            else:
                si[si2.is_valid()] = si2
        return si

    def aov_names(self):
        if self.use_aovs:
            return ['sdf_value', 'warp_t', 'vx', 'vy', 'div', 'i', 'weight_sum', 'weight', 'warp_t_dx', 'warp_t_dy',
                    'warp_t_dz']
        else:
            return []

    def traverse(self, cb):
        if self.sdf is not None:
            self.sdf.traverse(cb)
        super().traverse(cb)

    def parameters_changed(self, keys):
        if self.sdf is not None:
            self.sdf.parameters_changed(keys)
        super().parameters_changed(keys)
