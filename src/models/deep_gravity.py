# Deep Gravity Allocation Model with CSR batching & batched inference
from __future__ import annotations
import torch
import torch.nn as nn

def segment_logsumexp(scores: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
    """Numerically stable segment log-sum-exp over origins using PyTorch native scatter ops."""
    max_scores = torch.full((num_origins,), -float("inf"), dtype=scores.dtype, device=scores.device)
    max_scores.scatter_reduce_(0, origin_idx, scores, reduce="amax", include_self=False)
    shifted = scores - max_scores[origin_idx]
    sum_exp = torch.zeros((num_origins,), dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, origin_idx, torch.exp(shifted))
    return max_scores + torch.log(sum_exp.clamp_min(1e-12))


def build_origin_csr(pair_o_idx: torch.Tensor, num_origins: int):
    """
    Builds CSR index structure to slice pairs by origin in O(1) without scanning.
    
    Returns:
        sort_perm: (E,) permutation index that sorts pair_o_idx.
        offsets: (num_origins + 1,) CSR start and end pointers.
        active_origins: (N_active,) 1D tensor of origin IDs that have >= 1 pair.
    """
    sort_perm = torch.argsort(pair_o_idx)
    sorted_o_idx = pair_o_idx[sort_perm]
    counts = torch.bincount(sorted_o_idx, minlength=num_origins)
    offsets = torch.zeros(num_origins + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    active_origins = torch.nonzero(counts > 0).squeeze(-1)
    return sort_perm, offsets, active_origins


class DeepGravityAllocationModel(nn.Module):
    """
    Deep Gravity Destination Allocation Network (Simini et al., Nature Communications 2021).
    Adapted to the known-positive support Omega_c benchmark.

    Architecture:
        15 hidden layers feed-forward MLP:
        - 6 hidden layers of 256 units with LeakyReLU
        - 9 hidden layers of 128 units with LeakyReLU
        - 1 final linear layer to scalar score s_ij
        Total parameters: exactly 507,905.
    """

    def __init__(self, in_dim: int = 53):
        super().__init__()
        layers = []
        # First 6 hidden layers of 256 units
        layers.append(nn.Linear(in_dim, 256))
        layers.append(nn.LeakyReLU())
        for _ in range(5):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.LeakyReLU())
        # Next 9 hidden layers of 128 units
        layers.append(nn.Linear(256, 128))
        layers.append(nn.LeakyReLU())
        for _ in range(8):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.LeakyReLU())
        # Final linear layer to scalar score s_ij
        layers.append(nn.Linear(128, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        """Compute scalar logit s_ij for each pair."""
        return self.net(pair_feat).squeeze(-1)

    def compute_log_probs(self, pair_feat: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
        """Compute log p_ij = s_ij - logsumexp_{k in Omega_{c,i}}(s_ik) for candidate pairs."""
        scores = self.forward(pair_feat)
        lse = segment_logsumexp(scores, origin_idx, num_origins)
        return scores - lse[origin_idx]

    def compute_probs(self, pair_feat: torch.Tensor, origin_idx: torch.Tensor, num_origins: int) -> torch.Tensor:
        """Compute destination allocation probabilities p_ij."""
        log_p = self.compute_log_probs(pair_feat, origin_idx, num_origins)
        return torch.exp(log_p)

    def compute_loss_batch(
        self,
        batch_pair_feat: torch.Tensor,
        batch_origin_compact_idx: torch.Tensor,
        batch_trips: torch.Tensor,
        num_batch_origins: int,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss ONLY on pairs belonging to the current batch of origins.
        
        Args:
            batch_pair_feat: (E_batch, 53) feature matrix for pairs of the selected origins.
            batch_origin_compact_idx: (E_batch,) origin index remapped to [0, num_batch_origins - 1].
            batch_trips: (E_batch,) ground-truth flows on Omega_c for these pairs.
            num_batch_origins: Number of origins in this mini-batch (e.g. <= 64).
        """
        log_p = self.compute_log_probs(batch_pair_feat, batch_origin_compact_idx, num_batch_origins)
        
        # Origin total flow: O_i = sum_{j in Omega_{c,i}} t_ij
        O_i = torch.zeros(num_batch_origins, dtype=batch_trips.dtype, device=batch_trips.device)
        O_i.scatter_add_(0, batch_origin_compact_idx, batch_trips)
        
        # Empirical destination proportion: q_ij = t_ij / O_i
        q_ij = batch_trips / O_i[batch_origin_compact_idx].clamp_min(1e-12)
        
        # Pair cross-entropy component: - q_ij * log p_ij
        pair_ce = - q_ij * log_p
        
        # Origin loss: L_i = sum_{j in Omega_{c,i}} - q_ij * log p_ij
        origin_loss = torch.zeros(num_batch_origins, dtype=log_p.dtype, device=log_p.device)
        origin_loss.scatter_add_(0, batch_origin_compact_idx, pair_ce)
        
        active = O_i > 0
        if not active.any():
            return torch.tensor(0.0, device=batch_pair_feat.device, requires_grad=True)
        return origin_loss[active].mean()

    @torch.no_grad()
    def compute_probs_batched_csr(
        self,
        sorted_pair_feat: torch.Tensor,
        sort_perm: torch.Tensor,
        offsets: torch.Tensor,
        active_origins: torch.Tensor,
        total_pairs: int,
        batch_origins: int = 64,
        max_pairs_per_batch: int = 250000,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Compute destination probabilities in origin-chunked batches via CSR representation.
        Prevents forward-passing entire cities at once, capping peak memory.
        Writes probabilities back into the original pair order.
        """
        self.eval()
        sorted_probs = torch.zeros(total_pairs, dtype=torch.float32, device=device)
        
        i = 0
        n_active = len(active_origins)
        while i < n_active:
            batch_orig = []
            curr_pairs = 0
            while i < n_active and len(batch_orig) < batch_origins:
                orig_id = active_origins[i].item()
                st = offsets[orig_id].item()
                en = offsets[orig_id + 1].item()
                pair_count = en - st
                if pair_count == 0:
                    i += 1
                    continue
                if batch_orig and (curr_pairs + pair_count > max_pairs_per_batch):
                    break
                batch_orig.append(orig_id)
                curr_pairs += pair_count
                i += 1
                
            if not batch_orig:
                if i < n_active:
                    batch_orig.append(active_origins[i].item())
                    i += 1
                else:
                    break
                    
            num_b_orig = len(batch_orig)
            b_feat_list = []
            b_compact_list = []
            b_slice_ranges = []
            
            for compact_id, orig_id in enumerate(batch_orig):
                st = offsets[orig_id].item()
                en = offsets[orig_id + 1].item()
                b_feat_list.append(sorted_pair_feat[st:en])
                b_compact_list.append(torch.full((en - st,), compact_id, dtype=torch.long, device=device))
                b_slice_ranges.append((st, en))
                
            b_feat = torch.cat(b_feat_list, dim=0)
            b_compact = torch.cat(b_compact_list, dim=0)
            
            b_probs = self.compute_probs(b_feat, b_compact, num_b_orig)
            
            offset_ptr = 0
            for st, en in b_slice_ranges:
                length = en - st
                sorted_probs[st:en] = b_probs[offset_ptr : offset_ptr + length]
                offset_ptr += length
                
        # Invert permutation back to original unsorted pair order
        probs_original_order = torch.zeros(total_pairs, dtype=torch.float32, device=device)
        probs_original_order[sort_perm] = sorted_probs
        return probs_original_order
