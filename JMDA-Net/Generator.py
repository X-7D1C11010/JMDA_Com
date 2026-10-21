"""Neural entropic optimal transport and legacy row-Softmax ablations."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSPORT_MODES = ("legacy_row_softmax", "row_softmax", "sinkhorn")


class CostNet(nn.Module):
    """Learn a metric space and construct the squared Euclidean cost matrix."""

    def __init__(self, feature_dim: int):
        super().__init__()
        hidden_dim = max(feature_dim // 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        source_mapped = self.proj(source_features)
        target_mapped = self.proj(target_features)
        return (
            source_mapped.unsqueeze(1) - target_mapped.unsqueeze(0)
        ).square().sum(dim=2)


class TransmissionNetwork(nn.Module):
    """Predict batch-dependent logits used as transport-cost corrections."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        if cost_matrix.ndim != 2:
            raise ValueError("The transport cost matrix must have shape [B_s, B_t].")
        source_count, target_count = cost_matrix.shape
        logits = self.net(cost_matrix.reshape(1, 1, source_count, target_count))
        return logits.reshape(source_count, target_count)


class NeuralOptimalTransportGenerator(nn.Module):
    """Learn a cost and construct an intermediate domain through transport.

    Modes:
      - ``legacy_row_softmax`` reproduces the original TransNet + row-Softmax
        implementation and is retained only as the historical ablation.
      - ``row_softmax`` uses the revised geometric cost plus neural correction,
        but omits the target-marginal Sinkhorn projection.
      - ``sinkhorn`` is the main method: TransNet corrects the cost, Softmax
        creates a positive row-normalized kernel, and differentiable Sinkhorn
        scaling enforces both source and target marginals.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        transport_mode: str = "sinkhorn",
        epsilon: float = 0.1,
        sinkhorn_iterations: int = 50,
        sinkhorn_tolerance: float = 1e-4,
        adaptive_sinkhorn: bool = False,
        correction_scale: float = 0.1,
        numerical_eps: float = 1e-8,
    ):
        super().__init__()
        if transport_mode not in TRANSPORT_MODES:
            raise ValueError(
                f"Unknown transport_mode={transport_mode!r}; "
                f"expected one of {TRANSPORT_MODES}."
            )
        if epsilon <= 0 or numerical_eps <= 0 or sinkhorn_tolerance <= 0:
            raise ValueError(
                "epsilon, sinkhorn_tolerance, and numerical_eps must be positive."
            )
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be at least 1.")
        if correction_scale < 0:
            raise ValueError("correction_scale cannot be negative.")

        self.transport_mode = transport_mode
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_tolerance = float(sinkhorn_tolerance)
        self.adaptive_sinkhorn = bool(adaptive_sinkhorn)
        self.sinkhorn_max_iterations = (
            max(
                self.sinkhorn_iterations + 100,
                self.sinkhorn_iterations * 4,
            )
            if self.adaptive_sinkhorn
            else self.sinkhorn_iterations
        )
        self.correction_scale = float(correction_scale)
        self.numerical_eps = float(numerical_eps)

        self.cost_net = CostNet(feature_dim)
        self.transmission_net = TransmissionNetwork()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )
        self.last_transport_info: Dict[str, float] = {}

    def _sinkhorn_from_log_kernel(
        self,
        log_kernel: torch.Tensor,
        source_marginal: torch.Tensor,
        target_marginal: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Log-domain scaling with bounded convergence refinement.

        By default ``sinkhorn_iterations`` is also the fixed maximum, matching
        the supervised reference.  ``adaptive_sinkhorn=True`` instead treats
        it as the minimum and continues in deterministic blocks of ten until
        both residuals meet ``sinkhorn_tolerance`` or the bounded maximum.
        """
        log_source = torch.log(source_marginal.clamp_min(self.numerical_eps))
        log_target = torch.log(target_marginal.clamp_min(self.numerical_eps))
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)

        maximum_iterations = self.sinkhorn_max_iterations
        iterations_used = 0
        for iteration_index in range(maximum_iterations):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0), dim=1
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1), dim=0
            )
            iterations_used = iteration_index + 1

            should_check = (
                iterations_used >= self.sinkhorn_iterations
                and (
                    iterations_used % 10 == 0
                    or iterations_used == maximum_iterations
                )
            )
            if should_check:
                with torch.no_grad():
                    log_plan = (
                        log_u.unsqueeze(1)
                        + log_kernel
                        + log_v.unsqueeze(0)
                    )
                    row_marginal = torch.logsumexp(log_plan, dim=1).exp()
                    column_marginal = torch.logsumexp(log_plan, dim=0).exp()
                    residual = torch.maximum(
                        (row_marginal - source_marginal).abs().max(),
                        (column_marginal - target_marginal).abs().max(),
                    )
                if float(residual) <= self.sinkhorn_tolerance:
                    break

        mass_plan = torch.exp(
            log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
        )
        return mass_plan, iterations_used

    def compute_transport(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        source_marginal: Optional[torch.Tensor] = None,
        target_marginal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a row-conditional map and differentiable diagnostics."""
        if source_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError("Transport features must have shape [batch, feature_dim].")
        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError("Source and target transport feature dimensions must match.")

        source_count = source_features.shape[0]
        target_count = target_features.shape[0]
        if source_count == 0 or target_count == 0:
            raise ValueError("Transport batches cannot be empty.")

        geometric_cost = self.cost_net(source_features, target_features)
        transnet_logits = self.transmission_net(geometric_cost)
        # Squared distances grow with feature dimension. Normalize their batch
        # scale in the revised modes so epsilon has a stable meaning across
        # feature widths and training stages. Detaching the scale prevents the
        # network from manipulating the normalization denominator.
        geometric_cost_scale = geometric_cost.detach().mean().clamp_min(
            self.numerical_eps
        )
        normalized_geometric_cost = geometric_cost / geometric_cost_scale
        def normalized_marginal(values, count, name):
            if values is None:
                return geometric_cost.new_full((count,), 1.0 / count)
            values = values.to(
                device=geometric_cost.device,
                dtype=geometric_cost.dtype,
            ).reshape(-1)
            if values.numel() != count:
                raise ValueError(f"{name} must contain {count} values.")
            if not torch.isfinite(values).all() or bool((values < 0).any()):
                raise ValueError(f"{name} must contain finite non-negative values.")
            values = values.clamp_min(self.numerical_eps)
            return values / values.sum()

        source_marginal = normalized_marginal(
            source_marginal, source_count, "source_marginal"
        )
        target_marginal = normalized_marginal(
            target_marginal, target_count, "target_marginal"
        )

        class_support = None
        if source_labels is not None or target_labels is not None:
            if source_labels is None or target_labels is None:
                raise ValueError(
                    "source_labels and target_labels must be provided together."
                )
            source_labels = source_labels.reshape(-1)
            target_labels = target_labels.reshape(-1)
            if source_labels.numel() != source_count or target_labels.numel() != target_count:
                raise ValueError("Transport labels must match their batch sizes.")
            if self.transport_mode != "legacy_row_softmax":
                if not torch.equal(
                    torch.sort(source_labels).values,
                    torch.sort(target_labels).values,
                ):
                    raise ValueError(
                        "Class-conditional uniform-marginal OT requires equal "
                        "source/target class counts in the paired batch."
                    )
                class_support = source_labels[:, None].eq(target_labels[None, :])

        sinkhorn_iterations_used = 0
        if self.transport_mode == "legacy_row_softmax":
            # Exact historical behavior: TransNet logits are treated as the
            # transport logits and row-Softmax is the final conditional map.
            log_kernel = F.log_softmax(transnet_logits, dim=1)
            conditional_plan = log_kernel.exp()
            mass_plan = source_marginal.unsqueeze(1) * conditional_plan
            corrected_cost = geometric_cost
            cost_correction = torch.zeros_like(geometric_cost)
        else:
            # Revised neural cost: a bounded correction prevents TransNet from
            # arbitrarily replacing the geometric metric.
            cost_correction = self.correction_scale * torch.tanh(transnet_logits)
            corrected_cost = normalized_geometric_cost + cost_correction
            transport_logits = -corrected_cost / self.epsilon
            if class_support is not None:
                transport_logits = transport_logits.masked_fill(
                    ~class_support,
                    -torch.inf,
                )
            log_kernel = F.log_softmax(transport_logits, dim=1)

            if self.transport_mode == "sinkhorn":
                mass_plan, sinkhorn_iterations_used = self._sinkhorn_from_log_kernel(
                    log_kernel,
                    source_marginal,
                    target_marginal,
                )
                conditional_plan = mass_plan / source_marginal.unsqueeze(1)
            else:
                conditional_plan = log_kernel.exp()
                mass_plan = source_marginal.unsqueeze(1) * conditional_plan

        row_residual = (
            mass_plan.sum(dim=1) - source_marginal
        ).abs().max()
        column_residual = (
            mass_plan.sum(dim=0) - target_marginal
        ).abs().max()
        marginal_residual = torch.maximum(row_residual, column_residual)
        transport_cost = (mass_plan * corrected_cost).sum()
        entropy = -(
            mass_plan * torch.log(mass_plan.clamp_min(self.numerical_eps))
        ).sum()
        # The Sinkhorn branch solves the entropy-regularized inner problem
        #   min_P <P, C_tilde> - epsilon H(P)
        #   s.t. P 1 = a and P^T 1 = b.
        regularized_ot_objective = transport_cost - self.epsilon * entropy
        correction_regularization = cost_correction.square().mean()
        if class_support is None:
            off_class_mass = mass_plan.new_zeros(())
        else:
            off_class_mass = mass_plan.masked_select(~class_support).sum()

        details = {
            "mass_plan": mass_plan,
            "conditional_plan": conditional_plan,
            "softmax_kernel": log_kernel.exp(),
            "source_marginal": source_marginal,
            "target_marginal": target_marginal,
            "geometric_cost": geometric_cost,
            "geometric_cost_scale": geometric_cost_scale,
            "normalized_geometric_cost": normalized_geometric_cost,
            "corrected_cost": corrected_cost,
            "cost_correction": cost_correction,
            "transport_cost": transport_cost,
            "entropy": entropy,
            "regularized_ot_objective": regularized_ot_objective,
            "row_marginal_residual": row_residual,
            "column_marginal_residual": column_residual,
            "marginal_residual": marginal_residual,
            "correction_regularization": correction_regularization,
            "off_class_mass": off_class_mass,
            "class_conditional": mass_plan.new_tensor(
                float(class_support is not None)
            ),
            "sinkhorn_iterations_used": mass_plan.new_tensor(
                float(sinkhorn_iterations_used)
            ),
        }
        self.last_transport_info = {
            "marginal_residual": float(marginal_residual.detach()),
            "transport_cost": float(transport_cost.detach()),
            "entropy": float(entropy.detach()),
            "regularized_ot_objective": float(
                regularized_ot_objective.detach()
            ),
            "row_marginal_residual": float(row_residual.detach()),
            "column_marginal_residual": float(column_residual.detach()),
            "correction_regularization": float(correction_regularization.detach()),
            "off_class_mass": float(off_class_mass.detach()),
            "sinkhorn_iterations_used": float(sinkhorn_iterations_used),
        }
        return conditional_plan, details

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        return_details: bool = False,
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        source_marginal: Optional[torch.Tensor] = None,
        target_marginal: Optional[torch.Tensor] = None,
    ):
        conditional_plan, details = self.compute_transport(
            source_features,
            target_features,
            source_labels=source_labels,
            target_labels=target_labels,
            source_marginal=source_marginal,
            target_marginal=target_marginal,
        )

        # Barycentric target representation. conditional_plan has row sum 1
        # for every mode, including the Sinkhorn plan after division by a_i.
        transported_target = conditional_plan.matmul(target_features)
        source_count = source_features.shape[0]
        tau = torch.distributions.Beta(2.0, 2.0).sample((source_count, 1)).to(
            device=source_features.device,
            dtype=source_features.dtype,
        )
        linear_interpolation = source_features + tau * (
            transported_target - source_features
        )

        # Neural residual correction retained from the original implementation.
        residual_input = torch.cat(
            [source_features, transported_target, tau], dim=1
        )
        residual = self.mlp(residual_input)
        intermediate_features = linear_interpolation + tau * residual

        details["transported_target"] = transported_target
        details["tau"] = tau
        if return_details:
            return intermediate_features, details
        return intermediate_features
