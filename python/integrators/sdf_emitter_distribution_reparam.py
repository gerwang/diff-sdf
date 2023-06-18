import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfEmitterDistributionReparamIntegrator(ReparamIntegrator):

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
        # Reparameterize only if we are not rendering in primal mode
        reparametrize = True and mode != dr.ADMode.Primal
        reparam_primary_ray = True and reparametrize
        occluded, det, extra_output = self.ray_test(scene, sampler, ray, depth=0, reparam=reparam_primary_ray)
        valid_ray = scene.environment() is not None

        ds = mi.DirectionSample3f()
        ds.d = ray.d
        ds.n = -ray.d

        si = mi.SurfaceInteraction3f()
        si.p = ray.o
        si.n = mi.Vector3f(0, 1, 0)

        with dr.suspend_grad():
            index = self.query_emitter_index + 1
            emitters = ([scene.environment()] + self.adjoint_emitters)[index:index + 3]
            pdfs = [emitter.pdf_direction(si, ds) for emitter in emitters]
            while len(pdfs) < 3:
                pdfs.append(pdfs[-1])

        result = mi.Spectrum(1.0)
        with dr.suspend_grad():
            result *= mi.Color3f(pdfs)
        result *= det
        primary_det = det

        aovs = [extra_output[k] if (extra_output is not None) and (k in extra_output)
                else mi.Float(0.0) for k in self.aov_names()]
        return result, valid_ray, primary_det, aovs

    def to_string(self):
        return 'SdfEmitterDistributionReparamIntegrator'


mi.register_integrator("sdf_emitter_distribution_reparam", lambda props: SdfEmitterDistributionReparamIntegrator(props))
