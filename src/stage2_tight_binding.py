import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Parameters
# -------------------------

L = 100
t = 1.0


# -------------------------
# Build dense Hamiltonian
# -------------------------

H = np.zeros(
    (L, L),
    dtype=float,
)

for n in range(L - 1):

    H[n, n + 1] = -t
    H[n + 1, n] = -t


# -------------------------
# Inspect matrix
# -------------------------

print("Hamiltonian shape:")
print(H.shape)

print("\nTop-left 8 x 8 block:")
print(H[:8, :8])

print("\nHermitian check:")
print(
    np.allclose(
        H,
        H.T.conj(),
    )
)


# -------------------------
# Diagonalization
# -------------------------

energies, states = np.linalg.eigh(H)


print("\nLowest five energies:")
print(energies[:5])

print("\nHighest five energies:")
print(energies[-5:])


# -------------------------
# Spectrum
# -------------------------

state_index = np.arange(L)

plt.figure(figsize=(8, 5))

plt.plot(
    state_index,
    energies,
    ".",
)

plt.xlabel("Eigenstate index")
plt.ylabel("Energy")
plt.title("1D tight-binding spectrum")

plt.tight_layout()

plt.savefig(
    "outputs/stage2_spectrum.png",
    dpi=200,
)

plt.show()


# -------------------------
# Plot selected eigenstates
# -------------------------

selected_states = [
    0,
    L // 4,
    L // 2,
]

sites = np.arange(L)

for alpha in selected_states:

    psi = states[:, alpha]

    probability = np.abs(psi) ** 2

    plt.figure(figsize=(9, 4))

    plt.plot(
        sites,
        probability,
    )

    plt.xlabel("Site n")
    plt.ylabel(r"$|\psi_n|^2$")
    plt.title(
        f"Tight-binding eigenstate {alpha}, "
        f"E = {energies[alpha]:.4f}"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/stage2_state_{alpha}.png",
        dpi=200,
    )

    plt.show()


# -------------------------
# Normalization check
# -------------------------

alpha = L // 2

norm = np.sum(
    np.abs(states[:, alpha]) ** 2
)

print(
    "\nNormalization of selected state:",
    norm,
)