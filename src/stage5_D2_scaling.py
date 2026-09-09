import numpy as np
import matplotlib.pyplot as plt


def aubry_andre_hamiltonian(N, t, V, beta, phi=0.0):

    n = np.arange(N)

    onsite = V * np.cos(
        2 * np.pi * beta * n + phi
    )

    H = np.diag(onsite)

    for i in range(N - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t

    return H


def state_near_zero(H):

    energies, states = np.linalg.eigh(H)

    alpha = np.argmin(
        np.abs(energies)
    )

    return states[:, alpha]


def ipr(psi):

    return np.sum(
        np.abs(psi) ** 4
    )


# Fibonacci system sizes
sizes = np.array([
    55,
    89,
    144,
    233,
    377,
    610,
])

t = 1.0

beta = (
    np.sqrt(5) - 1
) / 2

phi = 0.0

V_values = {
    "extended": 1.0,
    "critical": 2.0,
    "localized": 3.0,
}


for name, V in V_values.items():

    ipr_values = []

    for N in sizes:

        print(
            f"{name:10s} "
            f"N={N:4d}"
        )

        H = aubry_andre_hamiltonian(
            N=N,
            t=t,
            V=V,
            beta=beta,
            phi=phi,
        )

        psi = state_near_zero(H)

        I = ipr(psi)

        ipr_values.append(I)

    ipr_values = np.array(
        ipr_values
    )

    # log(IPR) = -D2 log(N) + constant
    coefficients = np.polyfit(
        np.log(sizes),
        np.log(ipr_values),
        1,
    )

    slope = coefficients[0]

    D2 = -slope

    print()
    print(
        f"{name}: D2 ≈ {D2:.4f}"
    )

    fitted_ipr = np.exp(
        np.polyval(
            coefficients,
            np.log(sizes),
        )
    )

    plt.figure(figsize=(7, 5))

    plt.loglog(
        sizes,
        ipr_values,
        "o",
        label="numerical IPR",
    )

    plt.loglog(
        sizes,
        fitted_ipr,
        "--",
        label=rf"fit: $D_2={D2:.3f}$",
    )

    plt.xlabel(r"$N$")
    plt.ylabel(r"$P_2 = \mathrm{IPR}$")

    plt.title(
        rf"{name.capitalize()}: "
        rf"$V/t={V/t:.1f}$"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"outputs/stage5_D2_{name}.png",
        dpi=250,
    )

    plt.show()