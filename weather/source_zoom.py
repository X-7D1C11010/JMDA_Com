"""Opt-in paired center zoom; disabled unless a separate augmentation trial selects it."""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def augment_source_zoom(source, min_area=1., probability=.5, generator=None):
    if not .1 <= min_area <= 1 or not 0 <= probability <= 1:
        raise ValueError("Source zoom requires min area in [.1,1] and probability in [0,1]")
    record = {"source_zoom_min_area": min_area, "source_zoom_probability": probability,
              "source_zoom_applied_samples": 0}
    if min_area == 1 or probability == 0:
        return source, record
    if generator is None:
        raise ValueError("Use a dedicated zoom generator to preserve training RNG")
    rgb, ir = source["rgb"], source["nir"]
    if rgb.shape != ir.shape or rgb.ndim != 4:
        raise ValueError("Paired source zoom requires equal NCHW VIS and IR tensors")
    count, _, height, width = rgb.shape
    chosen = torch.rand(count, device=rgb.device, generator=generator) < probability
    areas = min_area+(1-min_area)*torch.rand(count, device=rgb.device, generator=generator)
    indices = chosen.nonzero().flatten().tolist()
    record["source_zoom_applied_samples"] = len(indices)
    if not indices:
        return source, record
    areas = areas.tolist()
    applied = [areas[index] for index in indices]
    record["source_zoom_applied_area_mean"] = sum(applied)/len(applied)
    record["source_zoom_applied_area_min"] = min(applied)
    record["source_zoom_applied_area_max"] = max(applied)
    result = dict(source)
    for key in ("rgb", "nir"):
        images = source[key].clone()
        for index in indices:
            side = math.sqrt(areas[index])
            ch, cw = max(1, round(height*side)), max(1, round(width*side))
            y, x = (height-ch)//2, (width-cw)//2
            crop = source[key][index:index+1, :, y:y+ch, x:x+cw]
            images[index] = F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)[0]
        result[key] = images
    return result, record
