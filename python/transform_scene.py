import drjit as dr
import mitsuba as mi


def get_id(x, fallback, prefix):
    key = x.id()
    if key == '':
        key = fallback
    return f'{prefix}{key}'


def copy_scene_dict(scene, T=None, use_shape=True, use_emitter=True, use_sensor=True, use_integrator=True, prefix=''):
    res = {
        'type': 'scene',
    }
    if use_shape:
        for i, x in enumerate(scene.shapes()):
            x_id = get_id(x, f'shape_{i}', prefix)
            if T is not None:
                params = mi.traverse(x)
                if 'to_world' in params:
                    params['to_world'] = T @ params['to_world']
                    params.update()
                elif 'vertex_positions' in params and 'vertex_normals' in params:
                    assert x.emitter() is None
                    params['vertex_positions'] = dr.ravel(T @ dr.unravel(mi.Point3f, params['vertex_positions']))
                    params['faces'] = params['faces']
                    if x.has_vertex_normals():
                        params['vertex_normals'] = dr.ravel(T @ dr.unravel(mi.Normal3f, params['vertex_normals']))
                    params.update()
            res.update({
                x_id: x,
            })
    if use_emitter:
        for i, x in enumerate(scene.emitters()):
            if T is not None:
                params = mi.traverse(x)
                if 'to_world' in params:
                    params['to_world'] = T @ params['to_world']
                params.update()
            res.update({
                get_id(x, f'emitter_{i}', prefix): x,
            })
    if use_sensor:
        for i, x in enumerate(scene.sensors()):
            if T is not None:
                params = mi.traverse(x)
                params['to_world'] = T @ params['to_world']
                params.update()
            res.update({
                get_id(x, f'sensor_{i}', prefix): x,
            })
    if use_integrator:
        if scene.integrator() is not None:
            res.update({
                get_id(scene.integrator(), 'integrator', prefix): scene.integrator(),
            })
    return res


def transform_scene(scene, T):
    scene_dict = copy_scene_dict(scene, T)
    new_scene = mi.load_dict(scene_dict)
    return new_scene
