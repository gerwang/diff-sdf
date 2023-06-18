import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfCurvatureIntegrator(ReparamIntegrator):

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.query_emitter_index = int(props.get('query_emitter_index', -1))
        props.mark_queried('hide_emitters')
        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')
        props.mark_queried('guiding_mis_compensation')
        self.curvature_epsilon = float(props.get('curvature_epsilon', 1e-4))

        self.adjoint_emitters = []

    def sample(self, mode, scene, sampler, ray,
               δL, state_in, reparam, active, **kwargs):
        active = mi.Mask(active)
        with dr.suspend_grad():
            pi: mi.PreliminaryIntersection3f = self.ray_intersect_preliminary(scene, ray, active=active)
            valid_ray = pi.is_valid()
        active &= valid_ray
        primary_det = mi.Float(1.0)
        aovs = []

        with dr.suspend_grad():
            point = dr.select(valid_ray, ray.o + ray.d * pi.t, mi.Point3f(0.))
        normal = dr.normalize(self.sdf.eval_grad(point))

        with dr.suspend_grad():
            random_d = mi.warp.square_to_uniform_sphere(sampler.next_2d(active=active))
            tangent = dr.cross(normal, random_d)
            point_shifted = point + tangent * self.curvature_epsilon * sampler.next_1d(active=active)
        normal_shifted = dr.normalize(self.sdf.eval_grad(point_shifted))

        curvature = dr.select(valid_ray,
                              dr.acos(dr.clamp(dr.dot(normal, normal_shifted), -1.0 + 1e-6, 1.0 - 1e-6)) / dr.pi,
                              mi.Float(0.))
        result = mi.Color3f(curvature, 0., 0.)

        return result, valid_ray, primary_det, aovs

    def to_string(self):
        return 'SdfCurvatureIntegrator'


mi.register_integrator("sdf_curvature", lambda props: SdfCurvatureIntegrator(props))
