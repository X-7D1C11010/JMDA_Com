"""Optimizer helpers for the parameter-group contract used by official CDAN."""


def materialize_cdan_parameter_groups(parameter_groups, base_lr, weight_decay):
    """Translate CDAN's ``lr_mult``/``decay_mult`` metadata for PyTorch SGD.

    PyTorch optimizers ignore those two custom keys unless the caller turns
    them into the standard per-group ``lr`` and ``weight_decay`` fields.  The
    official CDAN scheduler performs exactly this multiplication every step.
    """
    materialized = []
    for parameter_group in parameter_groups:
        group = dict(parameter_group)
        group["lr"] = float(base_lr) * float(group.get("lr_mult", 1.0))
        group["weight_decay"] = float(weight_decay) * float(
            group.get("decay_mult", 1.0)
        )
        materialized.append(group)
    return materialized


__all__ = ["materialize_cdan_parameter_groups"]
