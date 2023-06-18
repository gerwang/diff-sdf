import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfNormalDepthIntegrator(ReparamIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.query_emitter_index = int(props.get('query_emitter_index', -1))
        props.mark_queried('hide_emitters')
        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')
        props.mark_queried('guiding_mis_compensation')

        self.adjoint_emitters = []

    def aov_names(self):
        return ['depth']

    def sample(self, mode, scene, sampler, ray,
               δL, state_in, reparam, active, **kwargs):
        active = mi.Mask(active)
        si: mi.SurfaceInteraction3f
        si, si_d0, det, extra_output = self.ray_intersect(scene, sampler, ray, depth=0)
        valid_ray = si.is_valid()
        active &= valid_ray
        primary_det = det
        aovs = [si.t]

        result = mi.Color3f(si.n.x, si.n.y, si.n.z)

        return result, valid_ray, primary_det, aovs

    def to_string(self):
        return 'SdfNormalDepthIntegrator'


mi.register_integrator("sdf_normal_depth", lambda props: SdfNormalDepthIntegrator(props))
