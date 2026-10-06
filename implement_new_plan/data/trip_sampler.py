"""
Multinomial Trip Sampler for M_q condition.

Draws m random trips according to the categorical distribution over Omega_c:
    p_{ij} = T^{GT}_{ij} / sum_{a,b} T^{GT}_{ab}

From the sampled trips, estimates the empirical distance distribution:
    \tilde{Y}_D^{(m)}[k] = sum_{ij in B_k} n_{ij} / m

Grid:
    m in {100, 500, 1k, 5k, 10k, 50k, 100k, inf}
"""

import numpy as np
import torch


M_GRID = [100, 500, 1000, 5000, 10000, 50000, 100000, float("inf")]


def sample_multinomial_yd(
    pair_trips: torch.Tensor,
    bin_labels: torch.Tensor,
    m: int | float,
    seed: int = 42,
) -> np.ndarray:
    """
    Samples m trips from multinomial distribution and returns the 4-bin distribution \tilde{Y}_D^{(m)}.

    Args:
        pair_trips: (E,) positive trip counts.
        bin_labels: (E,) bin index (0..3).
        m: number of trips to sample (float('inf') returns exact oracle).
        seed: random seed for reproducibility.

    Returns:
        np.ndarray of shape (4,) representing bin proportions.
    """
    trips = pair_trips.detach().cpu().numpy().astype(np.float64)
    bins = bin_labels.detach().cpu().numpy().astype(np.int64)

    total_trips = np.sum(trips)
    if total_trips <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25])

    # If m is infinity or m >= total_trips, return the oracle
    if np.isinf(m):
        yd = np.zeros(4, dtype=np.float64)
        for k in range(4):
            yd[k] = np.sum(trips[bins == k])
        return yd / total_flow if (total_flow := np.sum(yd)) > 0 else np.array([0.25, 0.25, 0.25, 0.25])

    m = int(m)
    p_vals = trips / total_trips

    rng = np.random.default_rng(seed)
    sampled_counts = rng.multinomial(m, p_vals)  # (E,) counts of sampled trips

    yd_m = np.zeros(4, dtype=np.float64)
    for k in range(4):
        yd_m[k] = np.sum(sampled_counts[bins == k])

    return yd_m / float(m)


if __name__ == "__main__":
    trips = torch.tensor([10.0, 90.0, 200.0, 700.0])
    bins = torch.tensor([0, 1, 2, 3])
    print("Oracle:", sample_multinomial_yd(trips, bins, float("inf")))
    print("m=100:", sample_multinomial_yd(trips, bins, 100, seed=1))
    print("m=10000:", sample_multinomial_yd(trips, bins, 10000, seed=1))
