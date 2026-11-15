"""Opt-in source-only weather synthesis; geometry and validation stay fixed."""
import torch
from torchvision.transforms.functional import gaussian_blur


def augment_source_weather(batch, weather, probability=0., generator=None):
    if not 0 <= probability <= 1:
        raise ValueError("Weather augmentation probability must lie in [0,1]")
    record = {"source_weather_probability": probability, "source_weather": weather,
              "source_weather_augmented": 0, "target_weather_augmented": False}
    if not probability:
        return batch, record
    if weather not in {"dark", "foggy", "rainy", "backlight"}:
        raise ValueError("Unknown weather synthesis")
    x, ir = batch["rgb"], batch["nir"]
    n=len(x)
    def uniform(low, high):
        return low+(high-low)*torch.rand((n,1,1,1),device=x.device,generator=generator)
    active=uniform(0,1)<probability
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    raw=(x*std+mean).clamp(0,1)
    gray=(raw*raw.new_tensor([.2989,.5870,.1140]).view(1,3,1,1)).sum(1,keepdim=True)
    gp={"dark":.6,"rainy":.4,"foggy":.2,"backlight":.2}[weather]
    raw=torch.where(uniform(0,1)<gp,gray.expand_as(raw),raw)
    if weather=="dark":
        gain,gamma,contrast=uniform(.15,.6),uniform(1.1,1.8),uniform(.7,1.1)
        transformed=(raw.pow(gamma)*gain)
        blur_probability=.25
    elif weather=="foggy":
        gain,gamma,contrast=uniform(.8,1.1),uniform(.8,1.2),uniform(.5,.9)
        alpha,airlight=uniform(.15,.55),uniform(.75,1.)
        transformed=(1-alpha)*raw.pow(gamma)*gain+alpha*airlight
        blur_probability=.35
    else:
        gain,gamma,contrast=uniform(.5,1.1),uniform(.8,1.3),uniform(.5,1.)
        transformed=raw.pow(gamma)*gain
        blur_probability=.7 if weather=="rainy" else .2
    center=transformed.mean((2,3),keepdim=True)
    transformed=(center+contrast*(transformed-center)).clamp(0,1)
    blur_active=(uniform(0,1)<blur_probability)&active
    if bool(blur_active.any()):
        # A training-only blur; the two modalities' spatial registration is
        # unchanged. The IR input receives radiometric contrast only.
        sigma=float(uniform(.6,1.8)[0,0,0,0])
        transformed=torch.where(blur_active,gaussian_blur(transformed,[7,7],[sigma,sigma]),transformed)
    transformed=torch.where(active,transformed,raw)
    ir_raw=(ir*.5+.5).clamp(0,1)
    ir_mean=ir_raw.mean((2,3),keepdim=True)
    ir_contrast=uniform(.6,1.5)
    ir_aug=(ir_mean+ir_contrast*(ir_raw-ir_mean)).clamp(0,1)
    result=dict(batch,rgb=torch.where(active,(transformed-mean)/std,x),
                nir=torch.where(active,(ir_aug-.5)/.5,ir))
    record.update(source_weather_augmented=int(active.sum()),
        source_weather_gain_mean=float(gain.mean()),source_weather_gamma_mean=float(gamma.mean()),
        source_weather_blurred=int(blur_active.sum()),source_IR_contrast_mean=float(ir_contrast.mean()))
    return result,record
