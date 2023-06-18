from __future__ import annotations

from typing import Optional

import drjit as dr
import mitsuba as mi
import torch

from nerfstudio.field_components.rotater import Rotater
from nerfstudio.path_guiding import PathGuiding
from nerfstudio.pipelines.base_pipeline import Pipeline


def env_emitter_op(_o: mi.TensorXf, _v: mi.TensorXf, _rgb: mi.Color3f, _guiding_weight: mi.Float,
                   pipeline: Pipeline, torch_mi2gl_left: torch.Tensor, scene_scale: float, path_guiding: PathGuiding,
                   rotater: Optional[Rotater] = None):
    guiding_weight = _guiding_weight.torch().unsqueeze(-1)

    guide_o = _o.torch()
    guide_v = _v.torch()
    rgb = dr.ravel(_rgb).torch().view(-1, 3)
    path_guiding.train_primal(guide_o, guide_v, rgb, guiding_weight)
