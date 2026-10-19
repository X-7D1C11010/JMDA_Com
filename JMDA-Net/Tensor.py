"""Tensor alignment with epoch-level closed-form SVD projection updates.

The source/target projection matrices are not optimizer parameters.  They are
updated exactly once at an epoch boundary from a feature bank that represents
the complete training pairing.  During the subsequent mini-batch updates they
remain fixed, while Eq. (8) still back-propagates through the fixed projections
to the modality encoders.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedOuterProductTensor:
    """Exact rank-one representation of per-sample multimodal outer products.

    For factors ``x_1, ..., x_M`` this object represents
    ``x_1 ⨀ ... ⨀ x_M`` without allocating the exponentially large
    dense tensor. All contractions used by TAL are evaluated exactly from the
    factors, which makes the same method practical for VIS/IR/AIS inputs.
    """

    factors: Tuple[torch.Tensor, ...]

    def detach(self) -> "FactorizedOuterProductTensor":
        return FactorizedOuterProductTensor(
            tuple(factor.detach() for factor in self.factors)
        )


class TensorBasedAlignmentStable(nn.Module):
    """High-order tensor alignment using alternating SVD updates.

    Each domain first constructs the sample-wise outer product of all modal
    features. When mode ``m`` is updated, all other projection matrices are
    held fixed and used to contract their corresponding modes. The left/right
    singular vectors of the resulting source-target cross-covariance directly
    replace ``U_m`` and ``V_m``.

    ``U_m`` and ``V_m`` are buffers rather than ``Parameter`` objects. Thus,
    SVD is their only update rule; back-propagation updates the encoders while
    treating the projections as fixed during that optimizer step.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        num_modalities: int = 3,
        max_svd_sweeps: int = 3,
        svd_tolerance: float = 1e-4,
        unpaired_ot_epsilon: float = 0.1,
        unpaired_sinkhorn_iterations: int = 50,
        eps: float = 1e-8,
    ):
        super().__init__()

        if len(input_dims) != num_modalities or len(output_dims) != num_modalities:
            raise ValueError("input_dims/output_dims must match num_modalities.")
        if max_svd_sweeps < 1:
            raise ValueError("max_svd_sweeps must be at least 1.")
        if svd_tolerance <= 0 or unpaired_ot_epsilon <= 0 or eps <= 0:
            raise ValueError(
                "svd_tolerance, unpaired_ot_epsilon, and eps must be positive."
            )
        if unpaired_sinkhorn_iterations < 1:
            raise ValueError("unpaired_sinkhorn_iterations must be at least 1.")
        for mode, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            if output_dim > input_dim:
                raise ValueError(
                    f"Mode {mode}: output_dim ({output_dim}) cannot exceed "
                    f"input_dim ({input_dim})."
                )

        self.num_modalities = num_modalities
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.max_svd_sweeps = max_svd_sweeps
        self.svd_tolerance = svd_tolerance
        self.unpaired_ot_epsilon = float(unpaired_ot_epsilon)
        self.unpaired_sinkhorn_iterations = int(unpaired_sinkhorn_iterations)
        self.eps = eps
        self.last_update_info: Dict[str, object] = {}
        self.register_buffer(
            "projection_update_count", torch.zeros((), dtype=torch.long)
        )

        # Buffers are checkpointed and moved by module.to(device), but are
        # deliberately excluded from gradient-based optimizers.
        for mode, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            source_projection = torch.empty(input_dim, output_dim)
            nn.init.orthogonal_(source_projection)
            # Start both domains in the same latent coordinate system. Every
            # subsequent SVD update produces a paired left/right basis.
            target_projection = source_projection.clone()
            self.register_buffer(f"U_{mode}", source_projection)
            self.register_buffer(f"V_{mode}", target_projection)
            self.register_buffer(f"singular_values_{mode}", torch.zeros(output_dim))

    @property
    def U_matrices(self) -> Tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"U_{mode}") for mode in range(self.num_modalities))

    @property
    def V_matrices(self) -> Tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"V_{mode}") for mode in range(self.num_modalities))

    def _validate_modalities(
        self,
        source_modalities: Sequence[torch.Tensor],
        target_modalities: Sequence[torch.Tensor],
    ) -> None:
        if len(source_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} source modalities, "
                f"got {len(source_modalities)}."
            )
        if len(target_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} target modalities, "
                f"got {len(target_modalities)}."
            )

        source_batch = source_modalities[0].shape[0]
        target_batch = target_modalities[0].shape[0]
        if source_batch != target_batch:
            raise ValueError(
                "Alternating SVD requires paired source/target mini-batches "
                f"of equal size, got {source_batch} and {target_batch}."
            )

        for mode, (source, target, expected_dim) in enumerate(
            zip(source_modalities, target_modalities, self.input_dims)
        ):
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Mode {mode} features must be rank-2 [batch, dim].")
            if source.shape != target.shape:
                raise ValueError(
                    f"Mode {mode} source/target shapes differ: "
                    f"{tuple(source.shape)} vs {tuple(target.shape)}."
                )
            if source.shape[1] != expected_dim:
                raise ValueError(
                    f"Mode {mode} expected feature dim {expected_dim}, "
                    f"got {source.shape[1]}."
                )

    def create_multimodal_tensor(
        self,
        modalities: Sequence[torch.Tensor],
    ) -> FactorizedOuterProductTensor:
        """Construct the exact factorized form of Eq. (4)'s outer product."""
        return FactorizedOuterProductTensor(tuple(modalities))

    @staticmethod
    def mode_n_product(
        tensor: torch.Tensor,
        matrix: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Multiply tensor mode ``mode`` (excluding batch) by a projection."""
        tensor_axis = mode + 1
        if tensor.shape[tensor_axis] != matrix.shape[0]:
            raise ValueError(
                f"Mode {mode} dimension mismatch: "
                f"{tensor.shape[tensor_axis]} vs {matrix.shape[0]}."
            )
        moved = tensor.movedim(tensor_axis, -1)
        projected = torch.matmul(moved, matrix)
        return projected.movedim(-1, tensor_axis)

    def _mode_cross_covariance(
        self,
        source_tensor: FactorizedOuterProductTensor,
        target_tensor: FactorizedOuterProductTensor,
        current_mode: int,
    ) -> torch.Tensor:
        """Contract the tensor cross-covariance over all other modes.

        For a paired rank-one outer product, contraction of mode ``j`` uses
        the inner product between its projected source and target factors,

            <x_s^j U_j, x_t^j V_j>.

        Holding every ``j != current_mode`` projection fixed therefore leaves
        a weighted mode-wise cross-covariance for the SVD update.  Dividing an
        inner product by its projection width, and normalizing the final
        weights by their mean magnitude, changes only a global scale and keeps
        the computation stable for three or more modalities.
        """
        source_features = source_tensor.factors[current_mode]
        target_features = target_tensor.factors[current_mode]
        if source_features.shape[0] != target_features.shape[0]:
            raise ValueError("Source/target observation counts must match.")

        contraction_weight = source_features.new_ones(
            source_features.shape[0], 1
        )
        for mode in range(self.num_modalities):
            if mode == current_mode:
                continue
            source_projected = source_tensor.factors[mode].matmul(
                self.U_matrices[mode]
            )
            target_projected = target_tensor.factors[mode].matmul(
                self.V_matrices[mode]
            )
            paired_inner_product = (
                source_projected * target_projected
            ).mean(dim=1, keepdim=True)
            contraction_weight = contraction_weight * paired_inner_product

        weight_scale = contraction_weight.abs().mean().clamp_min(self.eps)
        contraction_weight = contraction_weight / weight_scale
        source_centered = source_features - source_features.mean(dim=0, keepdim=True)
        target_centered = target_features - target_features.mean(dim=0, keepdim=True)
        denominator = max(source_centered.shape[0] - 1, 1)
        covariance = source_centered.transpose(0, 1).matmul(
            contraction_weight * target_centered
        ) / denominator
        if not torch.isfinite(covariance).all():
            raise FloatingPointError("Non-finite values encountered in SVD cross-covariance.")
        return covariance

    def build_unlabeled_transport_plan(
        self,
        projected_source: Sequence[torch.Tensor],
        projected_target: Sequence[torch.Tensor],
        source_weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build a label-free multimodal Sinkhorn correspondence.

        Unlike the historical UDA fallback, this plan never assumes that row
        ``i`` in the source batch corresponds to row ``i`` in the target
        batch.  The plan is inferred from all pairwise projected features.  A
        non-uniform source marginal may be supplied from detached target
        predictions so partial-set target domains can down-weight unsupported
        source classes without reading target labels.
        """
        if len(projected_source) != self.num_modalities:
            raise ValueError("Projected source modality count is invalid.")
        if len(projected_target) != self.num_modalities:
            raise ValueError("Projected target modality count is invalid.")

        source_count = projected_source[0].shape[0]
        target_count = projected_target[0].shape[0]
        if source_count < 1 or target_count < 1:
            raise ValueError("Unlabeled transport requires non-empty domains.")
        for mode, (source, target) in enumerate(
            zip(projected_source, projected_target)
        ):
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Mode {mode} projected features must be rank-2.")
            if source.shape[1] != target.shape[1]:
                raise ValueError(
                    f"Mode {mode} projected feature widths do not match."
                )
            if source.shape[0] != source_count or target.shape[0] != target_count:
                raise ValueError("All modalities must share their domain batch size.")

        source_joint = torch.cat(
            [F.normalize(features, dim=1, eps=self.eps) for features in projected_source],
            dim=1,
        ) / max(self.num_modalities ** 0.5, 1.0)
        target_joint = torch.cat(
            [F.normalize(features, dim=1, eps=self.eps) for features in projected_target],
            dim=1,
        ) / max(self.num_modalities ** 0.5, 1.0)
        cost = torch.cdist(source_joint, target_joint, p=2).square()
        if not torch.isfinite(cost).all():
            raise FloatingPointError(
                "Non-finite values encountered in unlabeled transport cost."
            )
        cost_scale = cost.detach().mean().clamp_min(self.eps)
        log_kernel = -cost / (cost_scale * self.unpaired_ot_epsilon)

        if source_weights is None:
            source_marginal = cost.new_full(
                (source_count,), 1.0 / source_count
            )
        else:
            source_weights = source_weights.to(device=cost.device, dtype=cost.dtype)
            if source_weights.ndim != 1 or source_weights.numel() != source_count:
                raise ValueError(
                    "source_weights must contain one value per source observation."
                )
            if not torch.isfinite(source_weights).all() or bool(
                (source_weights < 0).any()
            ):
                raise ValueError(
                    "source_weights must contain finite non-negative values."
                )
            source_marginal = source_weights.clamp_min(self.eps)
            source_marginal = source_marginal / source_marginal.sum()
        target_marginal = cost.new_full(
            (target_count,), 1.0 / target_count
        )

        log_source = source_marginal.clamp_min(self.eps).log()
        log_target = target_marginal.clamp_min(self.eps).log()
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        for _ in range(self.unpaired_sinkhorn_iterations):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0), dim=1
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1), dim=0
            )
        mass_plan = torch.exp(
            log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
        )
        row_residual = (mass_plan.sum(dim=1) - source_marginal).abs().max()
        column_residual = (mass_plan.sum(dim=0) - target_marginal).abs().max()
        return mass_plan, {
            "source_marginal": source_marginal,
            "target_marginal": target_marginal,
            "row_marginal_residual": row_residual,
            "column_marginal_residual": column_residual,
            "marginal_residual": torch.maximum(row_residual, column_residual),
            "transport_entropy": -(
                mass_plan * mass_plan.clamp_min(self.eps).log()
            ).sum(),
        }

    def _unpaired_mode_cross_covariance(
        self,
        source_tensor: FactorizedOuterProductTensor,
        target_tensor: FactorizedOuterProductTensor,
        current_mode: int,
        mass_plan: torch.Tensor,
    ) -> torch.Tensor:
        """Contract an all-pairs soft correspondence for one modal update."""
        source_features = source_tensor.factors[current_mode]
        target_features = target_tensor.factors[current_mode]
        if mass_plan.shape != (
            source_features.shape[0],
            target_features.shape[0],
        ):
            raise ValueError("The unlabeled transport plan has an invalid shape.")

        weighted_plan = mass_plan
        for mode in range(self.num_modalities):
            if mode == current_mode:
                continue
            source_projected = F.normalize(
                source_tensor.factors[mode].matmul(self.U_matrices[mode]),
                dim=1,
                eps=self.eps,
            )
            target_projected = F.normalize(
                target_tensor.factors[mode].matmul(self.V_matrices[mode]),
                dim=1,
                eps=self.eps,
            )
            weighted_plan = weighted_plan * source_projected.matmul(
                target_projected.transpose(0, 1)
            )

        weight_scale = weighted_plan.abs().sum().clamp_min(self.eps)
        weighted_plan = weighted_plan / weight_scale
        source_marginal = mass_plan.sum(dim=1)
        target_marginal = mass_plan.sum(dim=0)
        source_mean = (
            source_marginal.unsqueeze(1) * source_features
        ).sum(dim=0, keepdim=True) / source_marginal.sum().clamp_min(self.eps)
        target_mean = (
            target_marginal.unsqueeze(1) * target_features
        ).sum(dim=0, keepdim=True) / target_marginal.sum().clamp_min(self.eps)
        source_centered = source_features - source_mean
        target_centered = target_features - target_mean
        covariance = source_centered.transpose(0, 1).matmul(
            weighted_plan.matmul(target_centered)
        )
        if not torch.isfinite(covariance).all():
            raise FloatingPointError(
                "Non-finite values encountered in unlabeled SVD cross-covariance."
            )
        return covariance

    @staticmethod
    def _align_paired_subspace(
        new_source: torch.Tensor,
        new_target: torch.Tensor,
        old_source: torch.Tensor,
        old_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Choose the least-changing basis inside the optimal SVD subspace.

        Singular vectors are not unique: signs can flip and nearly repeated
        singular values can rotate or exchange coordinates between epochs.
        A shared orthogonal Procrustes rotation of the left/right bases keeps
        their trace objective and contraction inner products unchanged while
        presenting a stable coordinate system to the downstream classifier.
        """
        cross_basis = (
            new_source.transpose(0, 1).matmul(old_source)
            + new_target.transpose(0, 1).matmul(old_target)
        )
        left, _, right_h = torch.linalg.svd(cross_basis, full_matrices=False)
        rotation = left.matmul(right_h)
        return new_source.matmul(rotation), new_target.matmul(rotation)

    @staticmethod
    def _least_changing_orthogonal_completion(
        informative_basis: torch.Tensor,
        old_basis: torch.Tensor,
        total_rank: int,
    ) -> torch.Tensor:
        """Complete an identified SVD subspace without rotating its nullspace.

        Singular vectors attached to numerically zero singular values are not
        identified by the Procrustes objective.  Taking those columns directly
        from a fresh SVD makes them jump arbitrarily between epochs.  Project
        the previous basis into the orthogonal complement of the newly
        identified subspace and retain the closest ``total_rank``-dimensional
        completion instead.  This only chooses among objective-equivalent
        null-space directions; it does not change the identified optimal
        subspace or its trace objective.
        """
        informative_rank = informative_basis.shape[1]
        if informative_rank > total_rank:
            raise ValueError("informative_basis exceeds the requested rank.")
        if old_basis.shape[1] != total_rank:
            raise ValueError("old_basis width must equal total_rank.")
        if informative_rank == total_rank:
            return informative_basis
        if informative_rank == 0:
            return old_basis.clone()

        residual = old_basis - informative_basis.matmul(
            informative_basis.transpose(0, 1).matmul(old_basis)
        )
        left, _, _ = torch.linalg.svd(residual, full_matrices=False)
        needed = total_rank - informative_rank
        completion = left[:, :needed]
        # Remove numerical leakage back into the identified subspace before
        # the reduced QR.  The prefix itself is already orthonormal.
        completion = completion - informative_basis.matmul(
            informative_basis.transpose(0, 1).matmul(completion)
        )
        completion, _ = torch.linalg.qr(completion, mode="reduced")
        return torch.cat((informative_basis, completion[:, :needed]), dim=1)

    @classmethod
    def _stabilize_rank_deficient_pair(
        cls,
        new_source: torch.Tensor,
        new_target: torch.Tensor,
        old_source: torch.Tensor,
        old_target: torch.Tensor,
        informative_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the previous bases for SVD directions with no numerical signal."""
        retained_rank = new_source.shape[1]
        informative_rank = min(max(int(informative_rank), 0), retained_rank)
        source_candidate = cls._least_changing_orthogonal_completion(
            new_source[:, :informative_rank], old_source, retained_rank
        )
        target_candidate = cls._least_changing_orthogonal_completion(
            new_target[:, :informative_rank], old_target, retained_rank
        )
        return cls._align_paired_subspace(
            source_candidate,
            target_candidate,
            old_source,
            old_target,
        )

    @staticmethod
    def _effective_rank(
        covariance: torch.Tensor,
        singular_values: torch.Tensor,
    ) -> Tuple[int, float]:
        """Return the standard dtype-aware numerical matrix rank.

        ``self.eps`` is a normalization guard, not a matrix-rank tolerance.
        Reusing its fixed 1e-8 value for float32 covariances treated round-off
        directions as informative. This matches the default relative cutoff
        used by ``torch.linalg.matrix_rank`` while reusing the existing SVD.
        """
        if singular_values.numel() == 0:
            return 0, 0.0
        leading_value = singular_values[0]
        if not torch.isfinite(leading_value) or leading_value <= 0:
            return 0, 0.0
        dtype_epsilon = torch.finfo(covariance.dtype).eps
        threshold = dtype_epsilon * max(covariance.shape) * leading_value
        effective_rank = int((singular_values > threshold).sum().item())
        return effective_rank, float(threshold)

    @torch.no_grad()
    def _alternating_svd_update(
        self,
        source_tensor: FactorizedOuterProductTensor,
        target_tensor: FactorizedOuterProductTensor,
    ) -> Dict[str, object]:
        """Run Gauss-Seidel mode updates until singular values converge."""
        source_tensor = source_tensor.detach()
        target_tensor = target_tensor.detach()
        previous_singular_values = None
        relative_change = float("inf")
        converged = False
        effective_ranks = []
        stabilized_null_dims = []
        rank_thresholds = []

        for sweep in range(self.max_svd_sweeps):
            current_singular_values = []
            effective_ranks = []
            stabilized_null_dims = []
            rank_thresholds = []

            for mode in range(self.num_modalities):
                # Other modal projections remain fixed at their latest values.
                covariance = self._mode_cross_covariance(
                    source_tensor,
                    target_tensor,
                    mode,
                )

                left, singular_values, right_h = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                rank = self.output_dims[mode]
                new_source = left[:, :rank]
                new_target = right_h.transpose(-2, -1)[:, :rank]
                effective_rank, rank_threshold = self._effective_rank(
                    covariance,
                    singular_values,
                )
                informative_rank = min(effective_rank, rank)
                new_source, new_target = self._stabilize_rank_deficient_pair(
                    new_source,
                    new_target,
                    self.U_matrices[mode],
                    self.V_matrices[mode],
                    informative_rank,
                )

                self.U_matrices[mode].copy_(new_source)
                self.V_matrices[mode].copy_(new_target)
                retained = singular_values[:rank]
                getattr(self, f"singular_values_{mode}").copy_(retained)
                current_singular_values.append(retained.clone())
                effective_ranks.append(effective_rank)
                stabilized_null_dims.append(rank - informative_rank)
                rank_thresholds.append(rank_threshold)

            if previous_singular_values is not None:
                changes = []
                for current, previous in zip(current_singular_values, previous_singular_values):
                    numerator = torch.linalg.vector_norm(current - previous)
                    denominator = torch.linalg.vector_norm(previous).clamp_min(self.eps)
                    changes.append((numerator / denominator).item())
                relative_change = max(changes)
                if relative_change < self.svd_tolerance:
                    converged = True
                    break

            previous_singular_values = current_singular_values

        return {
            "sweeps": sweep + 1,
            "converged": converged,
            "relative_singular_change": relative_change,
            "singular_values": tuple(values.detach().clone() for values in current_singular_values),
            "effective_ranks": tuple(effective_ranks),
            "stabilized_null_dims": tuple(stabilized_null_dims),
            "rank_thresholds": tuple(rank_thresholds),
        }

    @torch.no_grad()
    def _alternating_unpaired_svd_update(
        self,
        source_tensor: FactorizedOuterProductTensor,
        target_tensor: FactorizedOuterProductTensor,
        source_weights: torch.Tensor = None,
    ) -> Dict[str, object]:
        """Update paired bases from a label-free all-pairs OT correspondence."""
        source_tensor = source_tensor.detach()
        target_tensor = target_tensor.detach()
        previous_singular_values = None
        relative_change = float("inf")
        converged = False
        effective_ranks = []
        stabilized_null_dims = []
        rank_thresholds = []
        plan_info = None

        for sweep in range(self.max_svd_sweeps):
            projected_source = [
                features.matmul(projection)
                for features, projection in zip(
                    source_tensor.factors, self.U_matrices
                )
            ]
            projected_target = [
                features.matmul(projection)
                for features, projection in zip(
                    target_tensor.factors, self.V_matrices
                )
            ]
            mass_plan, plan_info = self.build_unlabeled_transport_plan(
                projected_source,
                projected_target,
                source_weights=source_weights,
            )
            current_singular_values = []
            effective_ranks = []
            stabilized_null_dims = []
            rank_thresholds = []

            for mode in range(self.num_modalities):
                covariance = self._unpaired_mode_cross_covariance(
                    source_tensor,
                    target_tensor,
                    mode,
                    mass_plan,
                )
                left, singular_values, right_h = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                rank = self.output_dims[mode]
                new_source = left[:, :rank]
                new_target = right_h.transpose(-2, -1)[:, :rank]
                effective_rank, rank_threshold = self._effective_rank(
                    covariance,
                    singular_values,
                )
                informative_rank = min(effective_rank, rank)
                new_source, new_target = self._stabilize_rank_deficient_pair(
                    new_source,
                    new_target,
                    self.U_matrices[mode],
                    self.V_matrices[mode],
                    informative_rank,
                )
                self.U_matrices[mode].copy_(new_source)
                self.V_matrices[mode].copy_(new_target)
                retained = singular_values[:rank]
                getattr(self, f"singular_values_{mode}").copy_(retained)
                current_singular_values.append(retained.clone())
                effective_ranks.append(effective_rank)
                stabilized_null_dims.append(rank - informative_rank)
                rank_thresholds.append(rank_threshold)

            if previous_singular_values is not None:
                changes = []
                for current, previous in zip(
                    current_singular_values, previous_singular_values
                ):
                    numerator = torch.linalg.vector_norm(current - previous)
                    denominator = torch.linalg.vector_norm(previous).clamp_min(self.eps)
                    changes.append((numerator / denominator).item())
                relative_change = max(changes)
                if relative_change < self.svd_tolerance:
                    converged = True
                    break
            previous_singular_values = current_singular_values

        return {
            "sweeps": sweep + 1,
            "converged": converged,
            "relative_singular_change": relative_change,
            "singular_values": tuple(
                values.detach().clone() for values in current_singular_values
            ),
            "effective_ranks": tuple(effective_ranks),
            "stabilized_null_dims": tuple(stabilized_null_dims),
            "rank_thresholds": tuple(rank_thresholds),
            "unlabeled_plan_marginal_residual": float(
                plan_info["marginal_residual"]
            ),
            "unlabeled_plan_entropy": float(plan_info["transport_entropy"]),
        }

    @torch.no_grad()
    def update_projections(
        self,
        source_modalities: Sequence[torch.Tensor],
        target_modalities: Sequence[torch.Tensor],
    ) -> Dict[str, object]:
        """Update every modal projection once from an epoch feature bank.

        The observations must be paired and cover the intended epoch-level
        statistics.  This method is deliberately separate from ``forward`` so
        a training mini-batch cannot silently redefine the feature coordinate
        system seen by the classifier.
        """
        self._validate_modalities(source_modalities, target_modalities)
        sample_count = source_modalities[0].shape[0]
        maximum_sample_rank = max(sample_count - 1, 0)
        requested_rank = max(self.output_dims)
        if requested_rank > maximum_sample_rank:
            raise ValueError(
                "The requested SVD projection rank is unsupported by the "
                f"epoch feature bank: projection_dim={requested_rank}, "
                f"paired_samples={sample_count}, maximum centered rank="
                f"{maximum_sample_rank}. Reduce projection_dim or provide "
                "more paired training observations."
            )

        source_tensor = self.create_multimodal_tensor(source_modalities)
        target_tensor = self.create_multimodal_tensor(target_modalities)
        update_info = self._alternating_svd_update(source_tensor, target_tensor)
        self.projection_update_count.add_(1)
        update_info.update(
            {
                "sample_count": sample_count,
                "maximum_sample_rank": maximum_sample_rank,
                "update_count": int(self.projection_update_count.item()),
            }
        )
        self.last_update_info = update_info
        return update_info

    @torch.no_grad()
    def update_unpaired_projections(
        self,
        source_modalities: Sequence[torch.Tensor],
        target_modalities: Sequence[torch.Tensor],
        source_weights: torch.Tensor = None,
    ) -> Dict[str, object]:
        """Update projections without labels or artificial positional pairs."""
        if len(source_modalities) != self.num_modalities:
            raise ValueError("Unexpected number of source modalities.")
        if len(target_modalities) != self.num_modalities:
            raise ValueError("Unexpected number of target modalities.")
        source_count = source_modalities[0].shape[0]
        target_count = target_modalities[0].shape[0]
        for mode, (source, target, expected_dim) in enumerate(
            zip(source_modalities, target_modalities, self.input_dims)
        ):
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Mode {mode} features must be rank-2.")
            if source.shape[0] != source_count or target.shape[0] != target_count:
                raise ValueError("All modalities must share their domain sample count.")
            if source.shape[1] != expected_dim or target.shape[1] != expected_dim:
                raise ValueError(f"Mode {mode} feature width is invalid.")
        maximum_sample_rank = max(min(source_count, target_count) - 1, 0)
        requested_rank = max(self.output_dims)
        if requested_rank > maximum_sample_rank:
            raise ValueError(
                "The requested SVD projection rank is unsupported by the "
                f"unpaired feature banks: projection_dim={requested_rank}, "
                f"source_samples={source_count}, target_samples={target_count}, "
                f"maximum rank={maximum_sample_rank}."
            )

        source_tensor = self.create_multimodal_tensor(source_modalities)
        target_tensor = self.create_multimodal_tensor(target_modalities)
        update_info = self._alternating_unpaired_svd_update(
            source_tensor,
            target_tensor,
            source_weights=source_weights,
        )
        self.projection_update_count.add_(1)
        update_info.update(
            {
                "sample_count": (source_count, target_count),
                "maximum_sample_rank": maximum_sample_rank,
                "update_count": int(self.projection_update_count.item()),
            }
        )
        self.last_update_info = update_info
        return update_info

    def _mean_cosine_similarity(
        self,
        source_projected: torch.Tensor,
        target_projected: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (7): batch mean of paired-sample cosine similarities."""
        return F.cosine_similarity(
            source_projected,
            target_projected,
            dim=1,
            eps=self.eps,
        ).mean()

    def compute_alignment_loss(
        self,
        projected_source: Sequence[torch.Tensor],
        projected_target: Sequence[torch.Tensor],
        mass_plan: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute paired SDA Eq. (8) or its soft-correspondence UDA form."""
        if mass_plan is None:
            correlations = [
                self._mean_cosine_similarity(source, target)
                for source, target in zip(projected_source, projected_target)
            ]
        else:
            expected_shape = (
                projected_source[0].shape[0],
                projected_target[0].shape[0],
            )
            if mass_plan.shape != expected_shape:
                raise ValueError(
                    f"Alignment plan shape {tuple(mass_plan.shape)} does not "
                    f"match {expected_shape}."
                )
            normalized_plan = mass_plan / mass_plan.sum().clamp_min(self.eps)
            correlations = []
            for source, target in zip(projected_source, projected_target):
                source_normalized = F.normalize(source, dim=1, eps=self.eps)
                target_normalized = F.normalize(target, dim=1, eps=self.eps)
                cosine_matrix = source_normalized.matmul(
                    target_normalized.transpose(0, 1)
                )
                correlations.append((normalized_plan * cosine_matrix).sum())
        return -torch.stack(correlations).mean()

    def forward(
        self,
        source_modalities: List[torch.Tensor],
        target_modalities: List[torch.Tensor],
        update_projections: bool = False,
        alignment_plan: torch.Tensor = None,
        compute_alignment: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Project a mini-batch with fixed bases and compute Eqs. (7)-(8)."""
        self._validate_modalities(source_modalities, target_modalities)

        if update_projections:
            raise RuntimeError(
                "Per-mini-batch SVD updates are disabled. Call "
                "tal_module.update_projections(...) once at the epoch boundary "
                "with the complete paired feature bank, then call forward with "
                "update_projections=False."
            )

        # Paper-ordered step: each domain first constructs its own high-order
        # multimodal tensor using sample-wise outer products.
        source_tensor = self.create_multimodal_tensor(source_modalities)
        target_tensor = self.create_multimodal_tensor(target_modalities)

        projected_source = [
            feature.matmul(projection)
            for feature, projection in zip(source_modalities, self.U_matrices)
        ]
        projected_target = [
            feature.matmul(projection)
            for feature, projection in zip(target_modalities, self.V_matrices)
        ]
        if compute_alignment:
            alignment_loss = self.compute_alignment_loss(
                projected_source,
                projected_target,
                mass_plan=alignment_plan,
            )
        else:
            alignment_loss = projected_source[0].new_zeros(())
        return projected_source, projected_target, alignment_loss
