import drjit as dr
import mitsuba as mi

from .reparam_split_light import ReparamSplitLightIntegrator


class SdfEmissionReparamIntegrator(ReparamSplitLightIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        props.mark_queried('hide_emitters')
        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')
        props.mark_queried('guiding_mis_compensation')

        self.m_aov_names = []
        for term in ['si_p', 'si_wi']:
            for component in ['X', 'Y', 'Z']:
                self.m_aov_names.append(f'{term}.{component}')
        self.m_aov_names.append('guiding_weight')

    @staticmethod
    def split_to_aovs(ray: mi.Ray3f):
        return [ray.o.x, ray.o.y, ray.o.z,
                -ray.d.x, -ray.d.y, -ray.d.z, ]

    def sample(self, mode, scene, sampler, ray,
               δL, state_in, reparam, active, **kwargs):
        # Reparameterize only if we are not rendering in primal mode
        reparametrize = True and mode != dr.ADMode.Primal
        reparam_primary_ray = True and reparametrize
        occluded, det, extra_output = self.ray_test(scene, sampler, ray, depth=0, reparam=reparam_primary_ray)
        valid_ray = scene.environment() is not None
        valid_ray &= ~occluded
        valid_foreground_ray = dr.select(occluded, 1.0, 0.0)
        valid_foreground_ray *= det  # differentiable re-parameterized valid_ray

        result = mi.Spectrum(1.0)
        result *= dr.select(valid_ray, 1.0, 0.0)
        result *= det
        primary_det = det

        aovs = self.split_to_aovs(ray)
        aovs.append(mi.Float(1.))
        aovs.extend([extra_output[k] if (extra_output is not None) and (k in extra_output)
                     else mi.Float(0.0) for k in self.aov_names()])
        return result, valid_foreground_ray, primary_det, aovs

    def to_string(self):
        return 'SdfEmissionReparamIntegrator'


mi.register_integrator("sdf_emission_reparam", lambda props: SdfEmissionReparamIntegrator(props))
