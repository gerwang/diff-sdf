import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfDepthIntegrator(ReparamIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.query_emitter_index = int(props.get('query_emitter_index', -1))
        props.mark_queried('hide_emitters')
        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')
        props.mark_queried('guiding_mis_compensation')

        self.adjoint_emitters = []

    def sample(self, mode, scene, sampler, ray,
               δL, state_in, reparam, active, **kwargs):
        active = mi.Mask(active)
        pi: mi.PreliminaryIntersection3f = self.ray_intersect_preliminary(scene, ray, active=active)
        valid_ray = pi.is_valid()
        active &= valid_ray
        primary_det = mi.Float(1.0)
        aovs = []

        result = mi.Spectrum(pi.t)

        return result, valid_ray, primary_det, aovs

    def to_string(self):
        return 'SdfDepthIntegrator'


mi.register_integrator("sdf_depth", lambda props: SdfDepthIntegrator(props))
