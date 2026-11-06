"""First BdG calculation: a clean square lattice with prescribed s-wave pairing.

Run:
    python 2.1_homogeneous_bdg.py
    python 2.1_homogeneous_bdg.py --L 12 --mu -1 --delta 0.5
    python 2.1_homogeneous_bdg.py --delta 0 --output output/2.1_normal_limit

Model and conventions
---------------------
One orbital per site, both electron spins, periodic boundaries, no magnetic field.
Nearest-neighbour hopping is -t. Set t = a = 1; energies are measured from mu.
The on-site singlet pair potential Delta is uniform, real, and supplied as input.
There is no gap-equation iteration or separate Hartree term in this exercise.

Normal matrix: h_ij = -t for neighbours and h_ii = -mu.
Reduced Nambu basis: (c_1,up, ..., c_N,up, c_1,down^dagger, ..., c_N,down^dagger).
BdG matrix: [[h, Delta I], [Delta* I, -h*]]. Its size is 2N by 2N, N = L**2.
This is one of the two equivalent spin blocks of the full spinful BdG problem.
Negative BdG eigenvalues are the redundant partners of positive excitations.

References in the supplied material
-----------------------------------
R. Oliveira, MSc thesis (2022), Appendix A, printed pp. 57-60:
    mean-field Hamiltonian (A.4), reduced BdG equation (A.16).
J.-X. Zhu, Bogoliubov-de Gennes Method and Its Applications (2016):
    lattice s-wave equation (2.44), p. 43; local DOS (2.47), p. 44;
    homogeneous spectrum and coherence factors (1.109), (1.111), p. 27.
The square-lattice dispersion replaces the continuum dispersion in that reference.

Outputs: two PNG figures and numerical checks printed to the terminal.
All text and axes are in English. NumPy and Matplotlib are the only dependencies.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normal_hamiltonian(L, t, mu):
    """Build h = H_TB - mu I on an L by L periodic square lattice."""
    N = L * L
    h = np.zeros((N, N))
    for y in range(L):
        for x in range(L):
            i = x + L * y
            h[i, i] = -mu

            # Include each +x and +y bond once, together with its reverse.
            right = (x + 1) % L + L * y
            up = x + L * ((y + 1) % L)
            h[i, right] = h[right, i] = -t
            h[i, up] = h[up, i] = -t
    return h


def bdg_hamiltonian(h, delta):
    """Couple the particle and hole sectors by uniform on-site pairing."""
    pairing = delta * np.eye(h.shape[0])
    return np.block([[h, pairing], [pairing.conj().T, -h.conj()]])


def momentum_reference(L, t, mu, delta):
    """Independent analytic solution on exactly the same finite momentum grid."""
    q = 2 * np.pi * np.fft.fftfreq(L)  # q = k a; fftfreq builds the grid only.
    qx, qy = np.meshgrid(q, q, indexing="xy")
    xi = (-2 * t * (np.cos(qx) + np.cos(qy)) - mu).ravel()
    E = np.sqrt(xi**2 + abs(delta)**2)
    ratio = np.divide(xi, E, out=np.zeros_like(xi), where=E > 0)
    u2 = 0.5 * (1 + ratio)
    v2 = 0.5 * (1 - ratio)
    return xi, E, u2, v2


def gaussian_dos(energy_grid, levels, weights, eta):
    """Broaden weighted delta peaks; eta is a numerical energy resolution."""
    dos = np.zeros_like(energy_grid)
    for level, weight in zip(levels, weights):
        gaussian = np.exp(-0.5 * ((energy_grid - level) / eta)**2)
        gaussian /= np.sqrt(2 * np.pi) * eta
        dos += weight * gaussian
    return dos


def check_solutions(h, H_bdg, normal_levels, levels, vectors, xi_k, E_k, delta):
    """Verify matrix solutions against independent finite-lattice physics."""
    reference = np.sort(np.concatenate((-E_k, E_k)))
    residual = H_bdg @ vectors - vectors * levels[None, :]
    checks = {
        "Normal matrix Hermiticity": np.max(np.abs(h - h.conj().T)),
        "BdG matrix Hermiticity": np.max(np.abs(H_bdg - H_bdg.conj().T)),
        "Normal real-space / k-space spectrum error":
            np.max(np.abs(normal_levels - np.sort(xi_k))),
        "BdG real-space / k-space spectrum error":
            np.max(np.abs(levels - reference)),
        "Positive / negative partner error": np.max(np.abs(levels + levels[::-1])),
        "Largest eigenpair residual": np.max(np.linalg.norm(residual, axis=0)),
        "Largest eigenvector normalization error":
            np.max(np.abs(np.sum(np.abs(vectors)**2, axis=0) - 1)),
    }
    scale = max(1.0, np.max(np.abs(levels)))
    for label, error in checks.items():
        print(f"{label}: {error:.3e}")
        if error > 1e-9 * scale:
            raise RuntimeError(f"Validation failed: {label}")

    minimum = np.min(np.abs(levels))
    expected = np.sqrt(np.min(np.abs(xi_k))**2 + abs(delta)**2)
    print(f"Minimum |E| / t: {minimum:.8f}; finite-grid prediction: {expected:.8f}")
    print(f"Prescribed Delta / t: {delta:.8f}")
    print("CHECKS_OK")


def plot_spectrum(levels, E_k, L, t, mu, delta, output):
    """Compare complete spectra and illustrate the homogeneous BdG branches."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    reference = np.sort(np.concatenate((-E_k, E_k)))
    index = np.arange(len(levels))
    axes[0].plot(index, reference / t, color="#177d91", lw=2.5,
                 label="Analytic finite-grid energies")
    axes[0].plot(index, levels / t, "o", ms=3, markevery=2,
                 mfc="none", mec="#b2446b", label="Real-space diagonalization")
    axes[0].set(xlabel="Sorted BdG eigenvalue index", ylabel=r"Energy $E/t$",
                title="A. Same finite system: all signed BdG eigenvalues")

    q = np.linspace(-np.pi, np.pi, 801)
    xi = -2 * t * (np.cos(q) + 1) - mu  # Momentum cut q_y = 0.
    E = np.sqrt(xi**2 + abs(delta)**2)
    axes[1].plot(q, xi / t, "--", color="0.6", lw=1.4,
                 label=r"Normal particle/hole branches: $\pm\xi$")
    axes[1].plot(q, -xi / t, "--", color="0.6", lw=1.4)
    axes[1].plot(q, E / t, color="#177d91", lw=2, label=r"Paired branches: $\pm E$")
    axes[1].plot(q, -E / t, color="#177d91", lw=2)
    axes[1].set(xlabel=r"Momentum component $q_x=k_xa$ (dimensionless)",
                ylabel=r"Energy $E/t$", title=r"B. Analytic momentum reference: $q_y=0$")
    axes[1].set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    for ax in axes:
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axhspan(-abs(delta) / t, abs(delta) / t, color="#d7edf0", alpha=0.6,
                   zorder=0)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(fr"Uniform s-wave pairing | $L={L}$, $\mu/t={mu/t:g}$, "
                 fr"$\Delta/t={delta/t:g}$ | prescribed $\Delta$", fontsize=14)
    fig.savefig(output / "bdg_spectrum.png", dpi=180)
    plt.close(fig)


def plot_dos(grid, normal_dos, bdg_dos, reference_dos, L, t, delta, eta, output):
    """Plot the electron DOS per site and per spin, rather than a Nambu count."""
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(grid / t, t * normal_dos, color="0.55", lw=1.8,
            label=r"Normal electrons: $\Delta=0$")
    ax.plot(grid / t, t * bdg_dos, color="#177d91", lw=2.2,
            label="Paired electrons: real-space eigenvectors")
    ax.plot(grid[::22] / t, t * reference_dos[::22], "o", ms=3.5,
            mfc="none", mec="#b2446b", label="Paired electrons: momentum reference")
    ax.axvspan(-abs(delta) / t, abs(delta) / t, color="#d7edf0", alpha=0.5)
    for edge in {-abs(delta) / t, abs(delta) / t}:
        ax.axvline(edge, color="#177d91", ls=":", lw=1)
    ax.set(xlabel=r"Energy relative to $\mu$: $E/t$",
           ylabel=r"Electron DOS per site, per spin: $t\rho(E)$", ylim=(0, None),
           title=fr"Finite square lattice | $L={L}$, $\Delta/t={delta/t:g}$, "
                 fr"Gaussian width $\eta/t={eta/t:g}$")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    fig.savefig(output / "electron_dos.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=12, help="Sites per direction (at least 3).")
    parser.add_argument("--mu", type=float, default=-1.0, help="Chemical potential mu/t.")
    parser.add_argument("--delta", type=float, default=0.5, help="Prescribed real Delta/t.")
    parser.add_argument("--eta", type=float, default=0.08, help="Gaussian DOS width eta/t.")
    parser.add_argument("--output", type=Path, default=Path("output/2.1_homogeneous_bdg"))
    args = parser.parse_args()
    if args.L < 3:
        parser.error("Use L >= 3 so the four periodic neighbours are distinct.")
    if not np.all(np.isfinite([args.mu, args.delta, args.eta])) or args.eta <= 0:
        parser.error("mu and delta must be finite; eta must be finite and positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    t = 1.0
    N = args.L**2

    # 1. Build the familiar tight-binding matrix, now measured from mu.
    h = normal_hamiltonian(args.L, t, args.mu)
    normal_levels = np.linalg.eigvalsh(h)

    # 2. Add the particle/hole sectors and solve the BdG eigenvalue problem.
    H_bdg = bdg_hamiltonian(h, args.delta)
    levels, vectors = np.linalg.eigh(H_bdg)
    u = vectors[:N, :]  # Electron components of ALL signed eigenvectors.
    v = vectors[N:, :]  # Hole components; together u and v have unit norm.
    print(f"L = {args.L}; N = {N}; h shape = {h.shape}; BdG shape = {H_bdg.shape}")
    print(f"Eigenvector parts: u shape = {u.shape}; v shape = {v.shape}")

    # 3. Solve the same homogeneous system analytically in momentum space.
    xi_k, E_k, u2_k, v2_k = momentum_reference(args.L, t, args.mu, args.delta)
    check_solutions(h, H_bdg, normal_levels, levels, vectors, xi_k, E_k, args.delta)

    # 4. Electron spectral weights prevent counting Nambu partners twice.
    # Summing the electron components of ALL signed states also works at Delta=0.
    electron_weights = np.sum(np.abs(u)**2, axis=0) / N
    bound = max(np.max(np.abs(normal_levels)), np.max(np.abs(levels))) + 6 * args.eta
    grid = np.linspace(-bound, bound, 2401)
    normal_dos = gaussian_dos(grid, normal_levels, np.full(N, 1 / N), args.eta)
    bdg_dos = gaussian_dos(grid, levels, electron_weights, args.eta)
    reference_dos = gaussian_dos(grid, np.concatenate((-E_k, E_k)),
                                 np.concatenate((v2_k, u2_k)) / N, args.eta)
    dos_error = np.max(np.abs(bdg_dos - reference_dos))
    areas = [np.trapezoid(dos, grid) for dos in (normal_dos, bdg_dos, reference_dos)]
    print(f"Electron DOS real-space / k-space maximum error: {dos_error:.3e}")
    print("Integrated electron DOS (normal, BdG, reference): "
          + ", ".join(f"{area:.10f}" for area in areas))
    if dos_error > 1e-8 or max(abs(area - 1) for area in areas) > 1e-5:
        raise RuntimeError("Electron DOS comparison or normalization failed.")

    # 5. Save two figures. A dense plotting grid does not add physical states.
    plot_spectrum(levels, E_k, args.L, t, args.mu, args.delta, args.output)
    plot_dos(grid, normal_dos, bdg_dos, reference_dos, args.L, t,
             args.delta, args.eta, args.output)
    print(f"Figures saved to: {args.output.resolve()}")
    print("FIXED_DELTA_REFERENCE_OK: Delta was supplied, not solved self-consistently.")


if __name__ == "__main__":
    main()
