import numpy as np
import matplotlib.pyplot as plt


def build_aubry_andre(
    L,
    t,
    lam,
    beta,
    phi=0.0,
):
    """
    Build a dense single-particle
    Aubry-André Hamiltonian
    with open boundary conditions.
    """

    H = np.zeros(
        (L, L),
        dtype=float,
    )

    sites = np.arange(L)

    onsite = lam * np.cos(
        2 * np.pi * beta * sites + phi
    )

    # On-site quasiperiodic energies
    H[np.diag_indices(L)] = onsite

    # Nearest-neighbor hopping
    for n in range(L - 1):

        H[n, n + 1] = -t
        H[n + 1, n] = -t

    return H


# -------------------------
# Parameters
# -------------------------

L = 300
t = 1.0

beta = (
    np.sqrt(5) - 1
) / 2

phi = 0.0

lambda_values = [
    1.0,
    2.0,
    3.0,
]


sites = np.arange(L)


# -------------------------
# Solve three regimes
# -------------------------

for lam in lambda_values:

    H = build_aubry_andre(
        L=L,
        t=t,
        lam=lam,
        beta=beta,
        phi=phi,
    )

    energies, states = np.linalg.eigh(H)

    # Choose state close to center of spectrum
    alpha = L // 2

    psi = states[:, alpha]

    probability = np.abs(psi) ** 2

    print(
        f"lambda = {lam:.1f}, "
        f"E = {energies[alpha]:.6f}, "
        f"norm = {probability.sum():.12f}"
    )

    plt.figure(figsize=(10, 4))

    plt.plot(
        sites,
        probability,
    )

    plt.xlabel("Site n")
    plt.ylabel(r"$|\psi_n|^2$")

    plt.title(
        rf"Aubry-André: "
        rf"$\lambda/t={lam/t:.1f}$"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/stage3_state_lambda_{lam:.1f}.png",
        dpi=200,
    )

    plt.show()