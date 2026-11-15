"""Opt-in population BN calibration on training partitions only.

This is a normalization enhancement. Frozen convolution/affine weights are
unchanged; no optimizer, target labels, validation images, or test images fit it.
"""
import hashlib
import math
import torch
from torch import nn
from torch.utils.data import DataLoader


class _ObservedLayer(Exception):
    pass


def refit_population_bn(encoder, view, input_key, device, batch_size=16, strength=1., only_layer=None):
    partition = getattr(view, "partition", None)
    if partition not in {"source_train", "target_train"}:
        raise ValueError("BN population fitting accepts training partitions only")
    if partition == "target_train" and getattr(view, "labelled", True):
        raise ValueError("Target labels are forbidden during BN fitting")
    if input_key not in {"rgb", "nir"} or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Invalid population BN settings")
    layers = {name: module for name, module in encoder.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    if not layers or any(not layer.track_running_stats for layer in layers.values()):
        raise ValueError("Encoder must have tracked BN buffers")
    if any(type(module).__name__ == "DomainRunningStats" for module in encoder.modules()):
        raise ValueError("Fit standalone domain copies, not online domain-stat wrappers")
    if only_layer is not None:
        if only_layer not in layers: raise ValueError("Unknown BN layer")
        layers = {only_layer: layers[only_layer]}
    before = {name: (layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone()) for name, layer in layers.items()}
    training = {module: module.training for module in encoder.modules()}
    moments, handles = {}, []

    def observe(name, _module, inputs):
        values = inputs[0].detach().float()
        axes = tuple(index for index in range(values.ndim) if index != 1)
        variance, mean = torch.var_mean(values, dim=axes, correction=0)
        count = values.numel() // values.shape[1]
        mean, m2 = mean.double(), variance.double() * count
        if name not in moments:
            moments[name] = [count, mean, m2]
        else:
            previous_count, previous_mean, previous_m2 = moments[name]
            total = previous_count + count
            delta = mean - previous_mean
            moments[name] = [total, previous_mean + delta * (count / total),
                             previous_m2 + m2 + delta.square() * (previous_count * count / total)]
        if only_layer is not None:
            # Later layers do not affect this layer's input distribution.
            raise _ObservedLayer()

    completed, samples, batches = False, 0, 0
    try:
        encoder.eval()
        for name, layer in layers.items():
            handles.append(layer.register_forward_pre_hook(lambda module, inputs, key=name: observe(key, module, inputs)))
        with torch.no_grad():
            for batch in DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(42)):
                if partition == "target_train" and "label" in batch:
                    raise RuntimeError("Target labels entered the statistics fitter")
                try:
                    encoder(batch[input_key].to(device))
                except _ObservedLayer:
                    if only_layer is None: raise
                samples += len(batch[input_key])
                batches += 1
        if samples != len(view) or set(moments) != set(layers):
            raise RuntimeError("Incomplete population BN coverage")
        with torch.no_grad():
            for name, layer in layers.items():
                count, mean, m2 = moments[name]
                layer.running_mean.lerp_(mean.to(layer.running_mean.dtype), strength)
                layer.running_var.lerp_((m2 / count).to(layer.running_var.dtype), strength)
                layer.num_batches_tracked.fill_(batches)
        completed = True
    finally:
        for handle in handles:
            handle.remove()
        if not completed:
            for name, layer in layers.items():
                for buffer, value in zip((layer.running_mean, layer.running_var, layer.num_batches_tracked), before[name]):
                    buffer.copy_(value)
        for module, mode in training.items():
            module.training = mode
    digest = hashlib.sha256()
    for layer in layers.values():
        digest.update(layer.running_mean.detach().cpu().numpy().tobytes())
        digest.update(layer.running_var.detach().cpu().numpy().tobytes())
    return {"fit_partition": partition, "fit_samples": samples, "batches": batches, "BN_layers": len(layers),
            "strength": strength, "input_modality": input_key, "target_training_labels_used": False,
            "population_moments": "all training pixels, Welford merge; validation never passed to fitter",
            "buffer_sha256": digest.hexdigest(), "only_layer": only_layer,
            "algorithm": "one-layer prefix" if only_layer is not None else "simultaneous iterative update"}


def refit_population_bn_layerwise(encoder, view, input_key, device, batch_size=16, strength=1.):
    """Fit in execution order, so downstream moments use fitted upstream BN.

    One simultaneous pass observes downstream inputs before upstream statistics
    are replaced. Prefix fitting avoids that stale-statistics approximation.
    """
    partition = getattr(view, "partition", None)
    if partition not in {"source_train", "target_train"} or (partition == "target_train" and getattr(view, "labelled", True)):
        raise ValueError("Only source training and label-free target training may fit BN")
    if input_key not in {"rgb", "nir"} or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Invalid layerwise population BN settings")
    layers = {name: module for name, module in encoder.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    order, hooks = [], []
    modes = {module: module.training for module in encoder.modules()}
    try:
        encoder.eval()
        for name, layer in layers.items():
            hooks.append(layer.register_forward_pre_hook(lambda _module, _inputs, key=name: order.append(key)))
        batch = next(iter(DataLoader(view, batch_size=1, num_workers=0, shuffle=False,
                                     generator=torch.Generator().manual_seed(42))))
        if partition == "target_train" and "label" in batch: raise RuntimeError("Target labels entered BN order inspection")
        with torch.no_grad(): encoder(batch[input_key].to(device))
    finally:
        for hook in hooks: hook.remove()
        for module, mode in modes.items(): module.training = mode
    if len(order) != len(set(order)) or set(order) != set(layers):
        raise ValueError("Layerwise calibration requires uniquely executed BN layers")
    records = [refit_population_bn(encoder, view, input_key, device, batch_size, strength, only_layer=name) for name in order]
    return {"algorithm": "execution-order prefix population fitting", "fit_partition": partition,
            "fit_samples": len(view), "BN_layers": len(order), "full_cohort_prefix_passes": len(order),
            "target_training_labels_used": False, "input_modality": input_key, "strength": strength,
            "layer_records": records}
