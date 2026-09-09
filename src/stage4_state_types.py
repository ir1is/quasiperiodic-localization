import numpy as np
import matplotlib.pyplot as plt


def aubry_andre_hamiltonian(N, t, V, beta, phi=0.0):
    """
    Dense single-particle Aubry-André Hamiltonian.

    Basis:
        |0>, |1>, ..., |N-1>

    Hopping:
        H[n,n+1] = -t

    Quasiperiodic onsite potential:
        H[n,n] = V cos(2 pi beta n + phi)
    """

    n = np.arange(N)

    onsite = V * np.cos(
        2 * np.pi * beta * n + phi
    )

    H = np.diag(onsite)

    for i in range(N - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t

    return H


def state_near_zero_energy(H):
    """
    Diagonalize H and return the eigenstate
    whose energy is closest to E = 0.
    """

    energies, states = np.linalg.eigh(H)

    alpha = np.argmin(np.abs(energies))

    E = energies[alpha]
    psi = states[:, alpha]

    return E, psi


def ipr(psi):
    """
    Inverse Participation Ratio:

        IPR = sum_n |psi_n|^4
    """

    return np.sum(np.abs(psi) ** 4)


# --------------------------------------------------
# Parameters
# --------------------------------------------------

N = 377
t = 1.0

beta = (np.sqrt(5) - 1) / 2

phi = 0.0

V_values = {
    "extended": 1.0,
    "critical": 2.0,
    "localized": 3.0,
}

sites = np.arange(N)


# --------------------------------------------------
# Calculate the three regimes
# --------------------------------------------------

for name, V in V_values.items():

    H = aubry_andre_hamiltonian(
        N=N,
        t=t,
        V=V,
        beta=beta,
        phi=phi,
    )

    E, psi = state_near_zero_energy(H)

    probability = np.abs(psi) ** 2

    I = ipr(psi)

    print()
    print("=" * 50)
    print(name.upper())
    print("=" * 50)

    print(f"V/t       = {V/t:.3f}")
    print(f"E         = {E:.8f}")
    print(f"IPR       = {I:.8f}")
    print(f"1/N       = {1/N:.8f}")
    print(f"sum |psi|² = {np.sum(probability):.12f}")

    plt.figure(figsize=(10, 4))

    plt.plot(
        sites,
        probability,
        linewidth=1,
    )

    plt.xlabel("site n")
    plt.ylabel(r"$|\psi_n|^2$")

    plt.title(
        rf"{name.capitalize()} state: "
        rf"$V/t={V/t:.1f}$, "
        rf"$IPR={I:.4f}$"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/stage4_{name}.png",
        dpi=250,
    )

    plt.show()