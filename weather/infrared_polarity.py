"""Source-only IR polarity augmentation; paired geometry and labels stay fixed."""
import math
import torch


def augment_source_ir_polarity(batch,probability=0.,generator=None):
    if not math.isfinite(probability) or not 0<=probability<=1:raise ValueError("Invalid IR inversion probability")
    record={"source_ir_inversion_probability":probability,"source_ir_inverted":0,"target_ir_inverted":False}
    if not probability:return batch,record
    infrared=batch["nir"]
    active=torch.rand((len(infrared),1,1,1),device=infrared.device,generator=generator)<probability
    result=dict(batch)
    # JMDA IR normalization is (x-.5)/.5: raw inversion 1-x is exactly -z.
    result["nir"]=torch.where(active,-infrared,infrared)
    record["source_ir_inverted"]=int(active.sum())
    return result,record
