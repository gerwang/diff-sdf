import drjit as dr
import mitsuba as mi

from .reparam_split_light import ReparamSplitLightIntegrator


def mis_weight_balanced(pdf_a, pdfs):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of multiple sampling strategies according to the power heuristic.
    """
    w = pdf_a / sum(pdfs)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))


class SdfDirectReparamOneSampleMisIntegrator(ReparamSplitLightIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.hide_emitters = props.get('hide_emitters', False)
        self.detach_indirect_si = props.get('detach_indirect_si', False)
        self.decouple_reparam = props.get('decouple_reparam', False)
        self.guiding_mis_compensation = props.get('guiding_mis_compensation', False)
        self.use_visibility = props.get('use_visibility', True)
        self._use_adjoint = (False, True)
        self._use_bsdf = (True, True)
        self._use_emitter = (True, True)

        self.m_aov_names = []
        for term in ['si_p', 'si_wi']:
            for component in ['X', 'Y', 'Z']:
                self.m_aov_names.append(f'{term}.{component}')
        self.m_aov_names.extend(['guiding_weight', 'active_s'])

    @staticmethod
    def split_to_aovs(si: mi.SurfaceInteraction3f):
        return [si.p.x, si.p.y, si.p.z,
                si.wi.x, si.wi.y, si.wi.z, ]

    def use_emitter(self, mode):
        return self._use_emitter[0 if mode == dr.ADMode.Primal else 1]

    def use_bsdf(self, mode):
        return self._use_bsdf[0 if mode == dr.ADMode.Primal else 1]

    def use_adjoint(self, mode):
        return self._use_adjoint[0 if mode == dr.ADMode.Primal else 1]

    def sample(self, mode, scene, sampler, ray,
               δL, state_in, reparam, active, **kwargs):

        active = mi.Mask(active)
        # Reparameterize only if we are not rendering in primal mode
        reparametrize = True and mode != dr.ADMode.Primal
        reparam_primary_ray = True and reparametrize
        si, si_d0, det, extra_output = self.ray_intersect(scene, sampler, ray, depth=0, reparam=reparam_primary_ray)
        valid_ray = (not self.hide_emitters) and scene.environment() is not None
        valid_ray |= si.is_valid()
        valid_foreground_ray = dr.select(si.is_valid(), mi.Float(1.0), mi.Float(0.0))
        valid_foreground_ray *= det  # differentiable re-parameterized valid_ray

        throughput = mi.Spectrum(1.0)
        if self.hide_emitters:
            throughput *= dr.select(valid_ray, 1.0, 0.0)
        result = mi.Spectrum(0.0)
        throughput *= det
        primary_det = det

        ctx = mi.BSDFContext()
        bsdf = si.shape.bsdf()

        # ---------------------- Count strategy numbers ----------------------
        n_strategy = 0
        if self.use_emitter(mode):
            n_strategy += 1
        if self.use_bsdf(mode):
            n_strategy += 1
        adjoint_emitters = self.adjoint_emitters
        if self.use_adjoint(mode):
            n_strategy += len(adjoint_emitters)
        inv_strategy = 1. / n_strategy
        # ---------------------- Randomly sample one strategy, create active mask ----------------------
        strategy_sample = sampler.next_1d(active)

        active_shading = active & si.is_valid()
        # ---------------------- Every sampling strategy outputs a direction ----------------------
        si_d = dr.detach(si)
        pdf_a = mi.Float(1.)
        d_sample = mi.Vector3f(0., 1., 0.)  # avoid zero direction vector
        active_s = mi.Mask(False)
        delta = mi.Bool(False)
        pdfs = []
        with dr.suspend_grad():

            cache_emitters = []
            if self.use_adjoint(mode):
                cache_emitters.extend(adjoint_emitters)
            if self.use_emitter(mode) and scene.environment() not in cache_emitters:
                cache_emitters.append(scene.environment())
            for emitter in cache_emitters:
                if hasattr(emitter, 'cache_interaction'):
                    emitter.cache_interaction(si_d.p)

            if self.use_emitter(mode):
                choose = (0 <= strategy_sample) & (strategy_sample < inv_strategy)
                strategy_sample -= inv_strategy
                active_e = active_shading & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & choose
                ds, _ = scene.environment().sample_direction(si_d, sampler.next_2d(active_e), active_e)
                d_sample[active_e] = ds.d  # dr.normalize(ds.p - si_d.p)
                pdf_a[active_e] = ds.pdf
                active_s[active_e] = active_e & dr.neq(ds.pdf, 0.0)
            if self.use_adjoint(mode):
                for adjoint_emitter in adjoint_emitters:
                    choose = (0 <= strategy_sample) & (strategy_sample < inv_strategy)
                    strategy_sample -= inv_strategy
                    active_a = active_shading & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & choose
                    ds, _ = adjoint_emitter.sample_direction(si_d, sampler.next_2d(active_a), active_a)
                    d_sample[active_a] = ds.d
                    pdf_a[active_a] = ds.pdf
                    active_s[active_a] = active_a & dr.neq(ds.pdf, 0.0)
            if self.use_bsdf(mode):
                choose = (0 <= strategy_sample) & (strategy_sample < inv_strategy)
                strategy_sample -= inv_strategy
                active_b = active_shading & choose
                bs, _ = bsdf.sample(ctx, si_d, sampler.next_1d(), sampler.next_2d(active_b), active_b)
                delta[active_b] = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                d_sample[active_b] = si_d.to_world(bs.wo)
                pdf_a[active_b] = bs.pdf
                active_s[active_b] = active_b & dr.neq(bs.pdf, 0.0)
            # ---------------------- Get pdf of every direction ----------------------
            ds = mi.DirectionSample3f()
            ds.d = d_sample
            ds.n = -d_sample
            ds.pdf = pdf_a
            if self.use_emitter(mode):
                pdfs.append(dr.select(delta, 0.0, scene.environment().pdf_direction(si, ds, active_s)))
            if self.use_bsdf(mode):
                wo = si_d.to_local(ds.d)
                pdfs.append(bsdf.pdf(ctx, si, wo, active_s))
            if self.use_adjoint(mode):
                for adjoint_emitter in adjoint_emitters:
                    pdfs.append(dr.select(delta, 0.0, adjoint_emitter.pdf_direction(si, ds, active_s)))
            mis_weight = mis_weight_balanced(ds.pdf, pdfs)
            if self.guiding_mis_compensation:
                mis_weight_guiding = mis_weight_balanced(pdfs[0], pdfs)
            else:
                mis_weight_guiding = 1.
        # ---------------------- Compute visibility ----------------------
        if self.detach_indirect_si:
            shadow_ray = si_d.spawn_ray(d_sample)
        elif self.decouple_reparam:
            shadow_ray = si_d0.spawn_ray(d_sample)
        else:
            shadow_ray = si.spawn_ray(d_sample)

        shadow_ray.d = dr.detach(shadow_ray.d)  # TODO: delta bsdf sample should not detach
        occluded, det_s, extra_output_ = self.ray_test(scene, sampler, shadow_ray, depth=1,
                                                       active=active_s, reparam=reparametrize)
        if not reparam_primary_ray:
            extra_output = extra_output_
        else:
            if self.warp_field is not None:
                if 'warp_t' in extra_output_:
                    extra_output['weight_sum'] = extra_output_['warp_t']

        wo = si.to_local(shadow_ray.d)
        si_e = dr.zeros(mi.SurfaceInteraction3f)
        si_e.sh_frame.n = ds.n
        si_e.initialize_sh_frame()
        si_e.n = si_e.sh_frame.n
        si_e.wi = -shadow_ray.d
        si_e.p = shadow_ray.o
        si_e.wavelengths = ray.wavelengths

        emitter_val = dr.select(active_s, 1.0, 0.0)
        emitter_val = dr.select(ds.pdf > 0, emitter_val / ds.pdf, 0.0)
        visiblity = dr.select(~occluded, 1.0, 0.0)

        bsdf_val = bsdf.eval(ctx, si, wo, active_s)
        contrib = emitter_val * mis_weight
        if self.use_visibility:
            contrib *= visiblity * det_s

        prob_mis = inv_strategy

        visible_mask = active & ~si.is_valid()
        si_e = dr.select(visible_mask, si, si_e)
        active_s |= visible_mask
        contrib = dr.select(visible_mask, 1.0, contrib)
        prob_mis = dr.select(visible_mask, 1.0, prob_mis)
        bsdf_val = dr.select(visible_mask, mi.Color3f(1.), bsdf_val)

        contrib /= prob_mis

        result[active_s] += throughput * contrib * bsdf_val
        aovs = self.split_to_aovs(si_e)
        aovs.append(mis_weight_guiding * contrib)
        aovs.append(active_s)
        aovs.extend([extra_output[k] if (extra_output is not None) and (k in extra_output)
                     else mi.Float(0.0) for k in self.aov_names()])
        return result, valid_foreground_ray, primary_det, aovs


mi.register_integrator("sdf_direct_reparam_onesamplemis", lambda props: SdfDirectReparamOneSampleMisIntegrator(props))
