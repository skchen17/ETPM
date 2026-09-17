import torch

from etrcm.memory import (
    decay_memory,
    external_delta_write,
    normalize,
    readout_conserving_consolidation,
)


def test_readout_conservation() -> None:
    torch.manual_seed(1)
    F = torch.randn(32, 32, dtype=torch.float64)
    M = torch.randn(32, 32, dtype=torch.float64)
    q = normalize(torch.randn(32, dtype=torch.float64))
    before = F + M
    F_after, M_after, _ = readout_conserving_consolidation(F, M, q, 0.1)
    assert torch.allclose(before, F_after + M_after, atol=1e-12, rtol=1e-12)


def test_repeated_consolidation_analytic_solution() -> None:
    torch.manual_seed(2)
    F = torch.randn(32, 32, dtype=torch.float64)
    M = torch.randn(32, 32, dtype=torch.float64)
    q = normalize(torch.randn(32, dtype=torch.float64))
    initial = F @ q
    gamma = 0.13
    steps = 17
    for _ in range(steps):
        F, M, _ = readout_conserving_consolidation(F, M, q, gamma)
    expected = (1.0 - gamma) ** steps * initial
    assert torch.allclose(F @ q, expected, atol=1e-11, rtol=1e-11)


def test_no_direct_evidence_amplification() -> None:
    torch.manual_seed(3)
    F = torch.randn(32, 32, dtype=torch.float64)
    M = torch.randn(32, 32, dtype=torch.float64)
    original = F + M
    for _ in range(50):
        q = normalize(torch.randn(32, dtype=torch.float64))
        F, M, _ = readout_conserving_consolidation(F, M, q, 0.1)
    assert torch.allclose(F + M, original, atol=2e-12, rtol=2e-12)


def test_equal_timescale_null() -> None:
    torch.manual_seed(4)
    A = torch.randn(32, 32, dtype=torch.float64)
    split = torch.rand_like(A)
    F1, M1 = split * A, (1.0 - split) * A
    F2, M2 = 0.25 * A, 0.75 * A
    rho = 0.97
    F1, M1 = decay_memory(F1, M1, rho, rho)
    F2, M2 = decay_memory(F2, M2, rho, rho)
    q = normalize(torch.randn(32, dtype=torch.float64))
    assert torch.allclose((F1 + M1) @ q, (F2 + M2) @ q, atol=1e-12)


def test_external_delta_update_converges() -> None:
    torch.manual_seed(5)
    F = torch.zeros(32, 32, dtype=torch.float64)
    M = torch.zeros_like(F)
    key = normalize(torch.randn(32, dtype=torch.float64))
    value = torch.randn(32, dtype=torch.float64)
    errors = []
    for _ in range(20):
        F, M, _ = external_delta_write(F, M, key, value, eta=0.5)
        errors.append(float(torch.linalg.vector_norm((F + M) @ key - value)))
    assert all(right < left for left, right in zip(errors[:-1], errors[1:]))
    assert errors[-1] < 2e-5

