"""Detached target-batch observability; never reads labels or alters gradients."""
import torch


@torch.no_grad()
def target_batch_diagnostics(logits, paths, mcc_temperature=None):
    logits=logits.detach()
    if len(logits)!=len(paths):raise ValueError("Target batch paths/logits differ")
    probabilities=logits.softmax(1)
    entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(1)
    result={"samples":len(logits),"unique_samples":len(set(paths)),
            "predicted_class_counts":torch.bincount(probabilities.argmax(1),minlength=logits.shape[1]).cpu().tolist(),
            "mean_entropy":float(entropy.mean()),"mean_max_probability":float(probabilities.max(1).values.mean()),
            "real_labels_used":False}
    if mcc_temperature is not None:
        p=(logits/mcc_temperature).softmax(1)
        weights=1+torch.exp((p*(p+1e-5).log()).sum(1))
        weights=len(p)*weights/weights.sum()
        confusion=(p*weights[:,None]).T@p
        # Match the vendored official implementation's orientation exactly.
        normalized=confusion/confusion.sum(1)
        result["MCC"]={"temperature":mcc_temperature,"soft_class_mass":p.sum(0).cpu().tolist(),
            "entropy_weight_effective_sample_count":float(weights.sum().square()/weights.square().sum()),
            "normalized_confusion":normalized.cpu().tolist(),
            "loss_reconstructed":float((normalized.sum()-normalized.trace())/p.shape[1])}
    return result
