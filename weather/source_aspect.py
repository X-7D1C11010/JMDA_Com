"""Paired source-only aspect distortion; keep the complete image in frame."""
import math
import torch
from torch.nn import functional as F


def augment_source_aspect(batch, minimum=1., maximum=1., probability=.75, generator=None):
    if not 0 < minimum <= maximum or not 0 <= probability <= 1:
        raise ValueError("Invalid source aspect bounds or probability")
    record={"source_aspect_min":minimum,"source_aspect_max":maximum,
            "source_aspect_augmented":0,"target_aspect_augmented":False}
    if not probability or (minimum==1 and maximum==1): return batch,record
    x=batch["rgb"];n=len(x)
    active=torch.rand((n,1,1,1),device=x.device,generator=generator)<probability
    ratio=(math.log(minimum)+(math.log(maximum)-math.log(minimum))*
           torch.rand((n,),device=x.device,generator=generator)).exp()
    # Squeeze one axis, then pad. Unlike a zoomed affine transform this never
    # cuts a bow/stern off the already cropped classification observation.
    sx=ratio.clamp(max=1);sy=(1/ratio).clamp(max=1)
    theta=x.new_zeros(n,2,3);theta[:,0,0]=1/sx;theta[:,1,1]=1/sy
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    result=dict(batch)
    for key in ("rgb","nir"):
        image=batch[key]
        border=torch.cat((image[:,:,0,:],image[:,:,-1,:],image[:,:,:,0],image[:,:,:,-1]),2)
        center=border.mean(2).unsqueeze(-1).unsqueeze(-1)
        transformed=F.grid_sample(image-center,grid,mode="bilinear",padding_mode="zeros",align_corners=False)+center
        result[key]=torch.where(active,transformed,image)
    record.update(source_aspect_augmented=int(active.sum()),
                  source_aspect_ratio_mean=float(ratio.mean()),
                  source_aspect_ratio_min_observed=float(ratio.min()),source_aspect_ratio_max_observed=float(ratio.max()))
    return result,record
