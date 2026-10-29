#!/usr/bin/env python3
"""1.1: A square-lattice tight-binding experiment on separability breaking.

Source
--------------
Miguel Goncalves, PhD thesis, Eq. (3.50). 

    H = +t sum_<i,j> (|i><j| + |j><i|) + sum_i U_i |i><i|
    U[y,x] = V/(1-alpha*cos(theta_x)) + V/(1-alpha*cos(theta_y))
             + Vprime*cos(theta_x)*cos(theta_y)
    theta_x = 2*pi*(p/L)*x + phi_x, and similarly for y.

Each unordered nearest-neighbour bond is included ONCE, with its reverse.
We keep the source's POSITIVE hopping sign. Earlier 0.x exercises used -t.
For odd periodic rings those signs cannot simply be interchanged by a
periodic staggered gauge transformation. The clean reference here is
E/t = 2*cos(k_x*a) + 2*cos(k_y*a).

One orbital per site, one passive spin sector, no magnetic field, no pairing.
All input/output energies use t as the unit: t=1 and a=1 internally.

Our experiment
--------------
1. check 2D E spectrum against its finite momentum grid (V'=0).
2. At Vprime=0 solve two 1D problems, then construct exact product states.
3. 2d Hamiltonian with V'=0, keeping the hopping unchanged.
4. Measure real/momentum IPR and the best product-state weight from an SVD.
5. Check eigenpair residuals and save numerical data and provenance.

The SVD weight is an extra diagnostic for this experiment. It is not an
electron-electron entanglement measure and is not a localization classifier.
The eigenvalue algorithms themselves are standard LAPACK / ARPACK routines.


Examples 
---------------------------------------------------
python 1.1_qp_square_tb.py
python 1.1_qp_square_tb.py --alpha 0.5 --output output/1.1_bounded
python 1.1_qp_square_tb.py --L 89 --solver sparse --states 64 --sigma 0.8

"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigsh, splu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


COLORS = ["#186b80", "#c48016", "#b5466d", "#6757a5", "#3f7e44"]


def fibonacci_approximant(length: int) -> tuple[int, float]:
    """Return p and tau=p/L, with L a Fibonacci number of at least 3."""
    previous, current = 1, 2
    while current < length:
        previous, current = current, previous + current
    if current != length or length < 3:
        raise ValueError("L must be a Fibonacci number >= 3: 3, 5, 8, 13, 21, ...")
    return previous, previous / current


def onsite_1d(theta: np.ndarray, V: float, alpha: float) -> np.ndarray:
    """Evaluate the source potential without clipping or shifting it."""
    if V == 0.0:
        return np.zeros_like(theta)
    # Stable evaluation near a pole: 1-alpha*cos(theta)
    # = (1-alpha) + 2*alpha*sin(theta/2)^2.
    denominator = (1.0 - alpha) + 2.0 * alpha * np.sin(theta / 2.0) ** 2
    if np.min(np.abs(denominator)) < 1.0e-14:
        raise ValueError("A sampled point is at/too close to a potential pole. "
                         "Choose another phase or use alpha<1 explicitly.")
    return V / denominator


def make_1d(local_energy: np.ndarray) -> sparse.csr_matrix:
    """A periodic chain with +1 hopping, including the closing bond."""
    L = len(local_energy)
    i = np.arange(L)
    j = (i + 1) % L
    rows = np.concatenate((i, j, i))
    columns = np.concatenate((j, i, i))
    values = np.concatenate((np.ones(L), np.ones(L), local_energy))
    return sparse.coo_matrix((values, (rows, columns)), shape=(L, L)).tocsr()


def make_2d(onsite: np.ndarray) -> sparse.csr_matrix:
    """Build the real-space H independently of the tensor-sum reference.

    The reshape convention is Psi[y,x], so site i=x+L*y.
    A row contains its onsite energy and four nearest-neighbour hoppings.
    """
    L = onsite.shape[0]
    sites = np.arange(L * L).reshape(L, L)
    i = sites.ravel()
    j_x = np.roll(sites, -1, axis=1).ravel()
    j_y = np.roll(sites, -1, axis=0).ravel()
    rows = np.concatenate((i, j_x, i, j_y, i))
    columns = np.concatenate((j_x, i, j_y, i, i))
    values = np.concatenate((np.ones(4 * L * L), onsite.ravel()))
    H = sparse.coo_matrix((values, (rows, columns)),
                          shape=(L * L, L * L)).tocsr()
    H.eliminate_zeros()
    return H


def potential_components(L: int, V: float, alpha: float,
                         phases: tuple[float, float]):
    """Return Ux, Uy and C[y,x]=cos(theta_y)*cos(theta_x)."""
    _, tau = fibonacci_approximant(L)
    n = np.arange(L)
    theta_x = 2 * np.pi * tau * n + phases[0]
    theta_y = 2 * np.pi * tau * n + phases[1]
    vx = onsite_1d(theta_x, V, alpha)
    vy = onsite_1d(theta_y, V, alpha)
    cross = np.outer(np.cos(theta_y), np.cos(theta_x))
    return vx, vy, cross


def tensor_action(vector: np.ndarray, hx: sparse.csr_matrix,
                  hy: sparse.csr_matrix, cross: np.ndarray,
                  Vprime: float) -> np.ndarray:
    """Apply H using the L-by-L wavefunction, independently of make_2d."""
    L = hx.shape[0]
    psi = vector.reshape(L, L)
    # H_y acts on rows; H_x acts on columns. H_x is real symmetric.
    return (hy @ psi + (hx @ psi.T).T + Vprime * cross * psi).ravel()


def product_solution(hx: sparse.csr_matrix, hy: sparse.csr_matrix,
                     states: int | None = None, sigma: float = 0.8):
    """Exact Vprime=0 solution: E_ab=ex[a]+ey[b], psi_ab=uy_b*ux_a."""
    ex, ux = linalg.eigh(hx.toarray())
    ey, uy = linalg.eigh(hy.toarray())
    sums = (ey[:, None] + ex[None, :]).ravel()
    if states is None:
        selected = np.argsort(sums)
    else:
        selected = np.argsort(np.abs(sums - sigma), kind="stable")[:states]
        selected = selected[np.argsort(sums[selected])]
    L = len(ex)
    vectors = np.empty((L * L, len(selected)))
    for column, index in enumerate(selected):
        b, a = divmod(int(index), L)
        vectors[:, column] = np.outer(uy[:, b], ux[:, a]).ravel()
    return sums[selected], vectors


def solve_targeted(H: sparse.csr_matrix, states: int, sigma: float,
                   seed: int, tolerance: float = 1.0e-10):
    """Shift-invert Lanczos with explicit sparse LU and recorded cost.

    This is scipy/ARPACK, not a new eigensolver. LU fill-in may be much larger
    than H's own sparse storage. One job's cost must be measured before HPC.
    """
    start = perf_counter()
    shifted = H.tocsc() - sigma * sparse.eye(H.shape[0], format="csc")
    factor = splu(shifted)
    factor_seconds = perf_counter() - start
    inverse = LinearOperator(H.shape, matvec=factor.solve, dtype=np.float64)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    energies, vectors = eigsh(
        H, k=states, sigma=sigma, which="LM", OPinv=inverse,
        v0=rng.normal(size=H.shape[0]), tol=tolerance,
        ncv=min(H.shape[0], max(2 * states + 1, 20)),
    )
    order = np.argsort(energies)
    details = {"factorization_seconds": factor_seconds,
               "iteration_seconds": perf_counter() - start,
               "lu_factor_nnz": int(factor.L.nnz + factor.U.nnz)}
    return energies[order], vectors[:, order], details


def observables(H: sparse.csr_matrix, energies: np.ndarray,
                vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Real/k IPR, product-state weight and eigenpair certification."""
    L = int(round(np.sqrt(H.shape[0])))
    wavefunctions = vectors.T.reshape(-1, L, L)
    probabilities = np.abs(wavefunctions) ** 2
    norms = probabilities.sum(axis=(1, 2))
    fourier = np.fft.fft2(wavefunctions, axes=(1, 2), norm="ortho")
    ipr_r = (probabilities ** 2).sum(axis=(1, 2)) / norms ** 2
    ipr_k = (np.abs(fourier) ** 4).sum(axis=(1, 2)) / norms ** 2
    product_weight = np.array([
        linalg.svdvals(psi, check_finite=False)[0] ** 2 / norm
        for psi, norm in zip(wavefunctions, norms)
    ])
    residual = np.linalg.norm(H @ vectors - vectors * energies[None, :], axis=0)
    relative_residual = residual / np.maximum(1.0, np.abs(energies))
    if np.max(np.abs(norms - 1.0)) > 1.0e-8:
        raise RuntimeError("Eigenvector normalization check failed.")
    if np.max(relative_residual) > 1.0e-7:
        raise RuntimeError("Eigenpair residual check failed.")
    for ipr in (ipr_r, ipr_k):
        if np.any(ipr < 1.0 / L ** 2 - 1.0e-9) or np.any(ipr > 1.0 + 1.0e-9):
            raise RuntimeError("IPR is outside its mathematical bounds.")
    if np.any(product_weight < 1.0 / L - 1.0e-9) or np.any(product_weight > 1.0 + 1.0e-9):
        raise RuntimeError("Product-state weight is outside its bounds.")
    return {"ipr_real": ipr_r, "ipr_momentum": ipr_k,
            "product_weight": product_weight,
            "residual_norm": residual, "relative_residual": relative_residual}


def best_product(vector: np.ndarray, L: int) -> np.ndarray:
    """Normalized best rank-one approximation in the x/y tensor product.

    For normalized Psi=U diag(s) Vh, w_product=s[0]^2 is the maximum squared
    overlap with any normalized f(y)*g(x). A product eigenstate has weight 1.
    """
    u, _, vh = linalg.svd(vector.reshape(L, L), full_matrices=False)
    return np.outer(u[:, 0], vh[0, :])


def gaussian_dos(energies: np.ndarray, grid: np.ndarray,
                 eta: float, total_sites: int) -> np.ndarray:
    """Full-spectrum DOS per site per spin, with integral 1 on the whole axis.

    The plotted interval need not contain the entire unbounded-potential
    spectrum. We do not renormalize that interval to area 1.
    """
    result = np.zeros_like(grid)
    for start in range(0, len(energies), 256):
        z = (grid[:, None] - energies[None, start:start + 256]) / eta
        result += np.exp(-0.5 * z ** 2).sum(axis=1)
    return result / (total_sites * eta * np.sqrt(2 * np.pi))


def validate_algorithms() -> dict[str, float | str]:
    """Small independent physics checks; inexpensive enough for every run."""
    clean_error = 0.0
    for L in (5, 8):
        H = make_2d(np.zeros((L, L)))
        q = 2 * np.pi * np.arange(L) / L
        exact = np.sort((2 * np.cos(q[:, None]) + 2 * np.cos(q[None, :])).ravel())
        clean_error = max(clean_error, float(np.max(np.abs(linalg.eigvalsh(H.toarray()) - exact))))

    L = 8
    phases = (0.37, 1.13)
    vx, vy, cross = potential_components(L, 1.0, 1.0, phases)
    hx, hy = make_1d(vx), make_1d(vy)
    base = vy[:, None] + vx[None, :]
    H0 = make_2d(base)
    energies, vectors = product_solution(hx, hy)
    sep_error = float(np.max(np.abs(linalg.eigvalsh(H0.toarray()) - energies)))
    sep_metrics = observables(H0, energies, vectors)
    product_error = float(np.max(np.abs(sep_metrics["product_weight"] - 1.0)))
    # A uniform real-space plane wave must have IPR_r=1/N and IPR_k=1.
    uniform = np.full((L * L, 1), 1.0 / L)
    plane_metrics = observables(make_2d(np.zeros((L, L))), np.array([4.0]), uniform)
    fourier_error = float(max(abs(plane_metrics["ipr_real"][0] - 1.0 / L ** 2),
                               abs(plane_metrics["ipr_momentum"][0] - 1.0)))

    rng = np.random.default_rng(71)
    trial = rng.normal(size=L * L)
    trial /= np.linalg.norm(trial)
    H = make_2d(base + 0.4 * cross)
    action_error = float(np.max(np.abs(H @ trial - tensor_action(trial, hx, hy, cross, 0.4))))
    symmetry_error = float(np.max(np.abs((H - H.T).toarray())))
    full = linalg.eigvalsh(H.toarray())
    partial, _, _ = solve_targeted(H, 12, 0.8, seed=71)
    expected = np.sort(full[np.argsort(np.abs(full - 0.8))[:12]])
    sparse_error = float(np.max(np.abs(partial - expected)))
    vx_periodic, vy_periodic, _ = potential_components(L, 1.0, 1.0,
                                                       (phases[0] + 2 * np.pi, phases[1] + 2 * np.pi))
    phase_error = float(max(np.max(np.abs(vx - vx_periodic)), np.max(np.abs(vy - vy_periodic))))

    errors = {"clean_spectrum_max_abs_error_t": clean_error,
              "separable_spectrum_max_abs_error_t": sep_error,
              "product_weight_max_abs_error": product_error,
              "plane_wave_ipr_max_abs_error": fourier_error,
              "tensor_action_max_abs_error_t": action_error,
              "symmetry_max_abs_error_t": symmetry_error,
              "sparse_vs_dense_max_abs_error_t": sparse_error,
              "phase_periodicity_max_abs_error_t": phase_error}
    if max(errors.values()) > 1.0e-7:
        raise RuntimeError(f"Independent validation failed: {errors}")
    return {"status": "PASS", **errors}


def save_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


def style_plots() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 12, "figure.dpi": 120,
                         "savefig.dpi": 180, "axes.spines.top": False,
                         "axes.spines.right": False})


def finish_figure(fig, path: Path, footer: str) -> None:
    # Reserve a bottom margin so the caption never overlaps axis labels.
    fig.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 0.925))
    fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_potential(output: Path, base: np.ndarray, cross: np.ndarray,
                   last_coupling: float, args) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    full = base + last_coupling * cross
    norm = SymLogNorm(linthresh=2.0, vmin=min(0.0, base.min(), full.min()),
                      vmax=max(base.max(), full.max(), 1.0))
    extent = (-0.5, args.L - 0.5, -0.5, args.L - 0.5)
    for ax, data, title in [(axes[0], base, "Separable onsite potential"),
                             (axes[2], full, "Total onsite potential")]:
        im = ax.imshow(data, origin="lower", extent=extent, norm=norm, cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=r"Onsite energy $U/t$")
    amplitude = max(abs(last_coupling), 1.0e-12)
    im = axes[1].imshow(last_coupling * cross, origin="lower", extent=extent,
                        cmap="RdBu_r", norm=Normalize(-amplitude, amplitude))
    axes[1].set_title(rf"Added coupling term: $V'/t={last_coupling:g}$")
    fig.colorbar(im, ax=axes[1], label=r"Added onsite energy $\Delta U/t$")
    for ax in axes:
        ax.set_xlabel(r"Site coordinate $x/a$")
        ax.set_ylabel(r"Site coordinate $y/a$")
    fig.suptitle(rf"2D quasiperiodic tight binding | $L={args.L}$, $V/t={args.V:g}$, $\alpha={args.alpha:g}$")
    finish_figure(fig, output / "potential_landscape.png",
                  "First/last panels: symmetric logarithmic color scale. The hopping is +t in every case.")


def draw_spectrum(output: Path, cases: list[dict], args) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    ipr_norm = Normalize(-np.log10(args.L ** 2), 0.0)
    scatter_plot = None
    for case in cases:
        visible = (case["energies"] >= args.emin) & (case["energies"] <= args.emax)
        scatter_plot = axes[0].scatter(
            np.full(visible.sum(), case["coupling"]), case["energies"][visible],
            c=np.log10(case["metrics"]["ipr_real"][visible]),
            cmap="plasma", norm=ipr_norm, s=12, alpha=0.9,
        )
    axes[0].set(xlabel=r"Nonseparable onsite amplitude $V'/t$", ylabel=r"Eigenenergy $E/t$",
                ylim=(args.emin, args.emax), title="Energy levels colored by real-space IPR")
    axes[0].set_xticks([case["coupling"] for case in cases])
    axes[0].margins(x=0.15)
    fig.colorbar(scatter_plot, ax=axes[0], label=r"$\log_{10}(\mathrm{IPR}_r)$")
    if args.solver == "dense":
        grid = np.linspace(args.emin, args.emax, 1401)
        for index, case in enumerate(cases):
            dos = gaussian_dos(case["energies"], grid, args.eta, args.L ** 2)
            axes[1].plot(grid, dos, color=COLORS[index % len(COLORS)],
                          label=rf"$V'/t={case['coupling']:g}$", lw=1.8)
            save_csv(output / f"dos_case_{index:02d}.csv", ["energy_over_t", "t_times_dos_per_site_per_spin"], zip(grid, dos))
        axes[1].set(xlabel=r"Energy $E/t$", ylabel=r"DOS per site per spin: $t\rho(E)$",
                    title=rf"Full-spectrum Gaussian DOS | $\eta/t={args.eta:g}$")
        footer = "Low-energy view. The complete spectrum is used for DOS; the plotted energy window is not renormalized."
    else:
        for index, case in enumerate(cases):
            axes[1].plot(np.arange(1, len(case["energies"]) + 1), case["energies"],
                          ".-", color=COLORS[index % len(COLORS)], ms=3,
                          label=rf"$V'/t={case['coupling']:g}$")
        axes[1].set(xlabel="Rank within the returned eigenpairs", ylabel=r"Eigenenergy $E/t$",
                    title=rf"Targeted eigenpairs near $\sigma/t={args.sigma:g}$")
        footer = "Partial spectra only. Their energy ranges may differ. A targeted sample is not a bulk DOS estimate."
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    fig.suptitle(rf"Changing separability | $L={args.L}$, $N={args.L**2}$, $\alpha={args.alpha:g}$ | one phase configuration")
    finish_figure(fig, output / "spectrum_and_dos.png", footer)


def draw_diagnostics(output: Path, cases: list[dict], args) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for index, case in enumerate(cases):
        energy = case["energies"]
        visible = (energy >= args.emin) & (energy <= args.emax)
        for ax, name in zip(axes, ("ipr_real", "ipr_momentum", "product_weight")):
            ax.scatter(energy[visible], case["metrics"][name][visible], s=15,
                        color=COLORS[index % len(COLORS)], alpha=0.75,
                        label=rf"$V'/t={case['coupling']:g}$")
    for ax, name in zip(axes[:2], ("Real-space", "Momentum-space")):
        ax.set_yscale("log")
        ax.axhline(1 / args.L ** 2, color="black", ls="--", lw=1,
                    label=r"Uniform probability: $1/N$")
        ax.set_ylim(0.7 / args.L ** 2, 1.2)
        ax.set_ylabel("Inverse participation ratio")
        ax.set_title(f"{name} concentration")
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(ylim=(0, 1.06), ylabel=r"Best product-state weight $w_{\rm prod}$",
                title="Can the state be written as f(y) g(x)?")
    for ax in axes:
        ax.set_xlabel(r"Eigenenergy $E/t$")
        ax.set_xlim(args.emin, args.emax)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="best", fontsize=9, frameon=False)
    axes[2].legend(loc="lower right", fontsize=9, frameon=False)
    fig.suptitle(rf"Localization and factorization are different questions | $L={args.L}$, $\alpha={args.alpha:g}$")
    finish_figure(fig, output / "localization_and_factorization.png",
                  "Finite-size diagnostics only. A single L and phase configuration cannot establish a critical phase.")


def draw_wavefunctions(output: Path, cases: list[dict], args) -> list[dict]:
    selected_cases = [cases[0], cases[-1]] if len(cases) > 1 else [cases[0]]
    fig, axes = plt.subplots(len(selected_cases), 2, figsize=(10.7, 4.8 * len(selected_cases)),
                              constrained_layout=True, squeeze=False)
    items = []
    densities = []
    for case in selected_cases:
        index = int(np.argmin(np.abs(case["energies"] - args.sigma)))
        psi = case["vectors"][:, index].reshape(args.L, args.L)
        product = best_product(psi.ravel(), args.L)
        densities.append((np.abs(psi) ** 2, np.abs(product) ** 2))
        items.append({"Vprime_over_t": case["coupling"], "returned_state_index": index,
                      "energy_over_t": float(case["energies"][index]),
                      "ipr_real": float(case["metrics"]["ipr_real"][index]),
                      "product_weight": float(case["metrics"]["product_weight"][index])})
    vmax = max(float(d.max()) for pair in densities for d in pair)
    norm = LogNorm(vmin=max(1e-12, vmax * 1e-8), vmax=vmax)
    for row, (pair, item) in enumerate(zip(densities, items)):
        for column, density in enumerate(pair):
            ax = axes[row, column]
            # Only the display floor is applied; stored wavefunctions are unchanged.
            image = ax.imshow(np.maximum(density, norm.vmin), origin="lower", cmap="magma", norm=norm)
            title = "Computed eigenstate" if column == 0 else "Best normalized product approximation"
            ax.set_title(title + "\n" + rf"$V'/t={item['Vprime_over_t']:g}$, $E/t={item['energy_over_t']:.4f}$")
            ax.set_xlabel(r"Site coordinate $x/a$")
            ax.set_ylabel(r"Site coordinate $y/a$")
        axes[row, 0].text(0.02, 0.03,
                         rf"$\mathrm{{IPR}}_r={item['ipr_real']:.4f}$; $w_{{\rm prod}}={item['product_weight']:.4f}$",
                         transform=axes[row, 0].transAxes, color="white", fontsize=10,
                         bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"})
    fig.colorbar(image, ax=axes.ravel().tolist(), label=r"Site probability $|\psi(x,y)|^2$ (log color scale)", shrink=0.8)
    fig.suptitle(rf"Real-space eigenstates and their product approximation | target $E/t\approx{args.sigma:g}$")
    finish_figure(fig, output / "wavefunctions.png",
                  "Each row independently selects the eigenstate closest to the target energy. Rows do not track one eigenstate.")
    return items


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=21, help="Linear size, a Fibonacci number (default: 21).")
    parser.add_argument("--V", type=float, default=1.0, help="V/t (default: 1).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Potential parameter, 0<=alpha<=1 (default: 1, thesis).")
    parser.add_argument("--couplings", type=float, nargs="+", default=[0.0, 0.2, 1.0], help="Vprime/t values (default: 0 0.2 1).")
    parser.add_argument("--solver", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--states", type=int, default=64, help="Number of targeted eigenpairs in sparse mode.")
    parser.add_argument("--sigma", type=float, default=0.8, help="Target energy E/t for sparse solve and state images.")
    parser.add_argument("--seed", type=int, default=1701, help="Reproducible phase and ARPACK seed.")
    parser.add_argument("--phi-x", type=float, help="Override x phase (radians); provide both phases.")
    parser.add_argument("--phi-y", type=float, help="Override y phase (radians); provide both phases.")
    parser.add_argument("--emin", type=float, default=-3.0, help="Lower plotted energy E/t.")
    parser.add_argument("--emax", type=float, default=6.0, help="Upper plotted energy E/t.")
    parser.add_argument("--eta", type=float, default=0.08, help="Gaussian DOS broadening eta/t; dense mode only.")
    parser.add_argument("--save-eigenvectors", action="store_true", help="Save all returned vectors; can create large files.")
    parser.add_argument("--output", type=Path, default=Path("output/1.1_qp_square_tb"))
    args = parser.parse_args()
    finite = [args.V, args.alpha, args.sigma, args.emin, args.emax, args.eta, *args.couplings]
    finite += [value for value in (args.phi_x, args.phi_y) if value is not None]
    if not all(np.isfinite(finite)):
        parser.error("All numerical inputs must be finite.")
    try:
        fibonacci_approximant(args.L)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.alpha <= 1 or args.V < 0:
        parser.error("Use 0<=alpha<=1 and V>=0.")
    if args.emax <= args.emin or args.eta <= 0:
        parser.error("Use emax>emin and eta>0.")
    if (args.phi_x is None) != (args.phi_y is None):
        parser.error("Provide both --phi-x and --phi-y, or neither.")
    if len(set(args.couplings)) != len(args.couplings) or 0.0 not in args.couplings:
        parser.error("Couplings must be unique and include 0 for the exact product reference.")
    args.couplings = sorted(args.couplings)
    if args.couplings[0] != 0.0:
        parser.error("This experiment uses nonnegative Vprime values.")
    if args.solver == "sparse" and not 1 <= args.states < args.L ** 2 - 1:
        parser.error("Sparse mode needs 1<=states<N-1.")
    if args.solver == "dense" and args.L ** 2 > 4096:
        parser.error("The teaching pilot limits dense mode to N<=4096; use --solver sparse for larger L.")
    return args


def main() -> None:
    args = parse_arguments()
    start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    validation = validate_algorithms()
    print("Independent physics/solver checks: PASS", flush=True)
    rng = np.random.default_rng(args.seed)
    phases = tuple(rng.uniform(0, 2 * np.pi, 2)) if args.phi_x is None else (args.phi_x, args.phi_y)
    p, tau = fibonacci_approximant(args.L)
    vx, vy, cross = potential_components(args.L, args.V, args.alpha, phases)
    hx, hy = make_1d(vx), make_1d(vy)
    base = vy[:, None] + vx[None, :]
    cases = []
    case_reports = []
    for index, coupling in enumerate(args.couplings):
        tick = perf_counter()
        onsite = base + coupling * cross
        H = make_2d(onsite)
        details = {}
        if coupling == 0.0:
            energies, vectors = product_solution(hx, hy, None if args.solver == "dense" else args.states, args.sigma)
            method = "Exact 1D product construction"
        elif args.solver == "dense":
            energies, vectors = linalg.eigh(H.toarray(), driver="evd")
            method = "Dense symmetric LAPACK eigh (evd)"
        else:
            energies, vectors, details = solve_targeted(H, args.states, args.sigma, args.seed + index)
            method = "Sparse shift-invert ARPACK/Lanczos with SuperLU"
        solve_seconds = perf_counter() - tick
        tick = perf_counter()
        metrics = observables(H, energies, vectors)
        diagnostic_seconds = perf_counter() - tick
        case = {"coupling": coupling, "energies": energies, "vectors": vectors, "metrics": metrics}
        cases.append(case)
        prefix = args.output / f"case_{index:02d}"
        fields = ["returned_state_index", "energy_over_t", *metrics.keys()]
        save_csv(prefix.with_suffix(".csv"), fields,
                 ([i, energy, *(metrics[key][i] for key in metrics)] for i, energy in enumerate(energies)))
        data = {"energies_over_t": energies, "onsite_over_t": onsite, **metrics}
        representative = int(np.argmin(np.abs(energies - args.sigma)))
        data["representative_eigenvector"] = vectors[:, representative].reshape(args.L, args.L)
        if args.save_eigenvectors:
            data["eigenvectors"] = vectors
        np.savez_compressed(prefix.with_suffix(".npz"), **data)
        visible = (energies >= args.emin) & (energies <= args.emax)
        report = {"case_index": index, "Vprime_over_t": coupling, "solver": method,
                  "spectrum_complete": args.solver == "dense", "returned_eigenpairs": len(energies),
                  "returned_energy_min_over_t": float(energies.min()), "returned_energy_max_over_t": float(energies.max()),
                  "states_in_plot_window": int(visible.sum()),
                  "solve_seconds": solve_seconds, "diagnostic_seconds": diagnostic_seconds,
                  "hamiltonian_nnz": int(H.nnz),
                  "hamiltonian_csr_bytes": int(H.data.nbytes + H.indices.nbytes + H.indptr.nbytes),
                  "max_relative_eigenpair_residual": float(metrics["relative_residual"].max()),
                  "window_median_ipr_real": float(np.median(metrics["ipr_real"][visible])) if visible.any() else None,
                  "window_median_product_weight": float(np.median(metrics["product_weight"][visible])) if visible.any() else None,
                  "window_min_product_weight": float(metrics["product_weight"][visible].min()) if visible.any() else None,
                  **details}
        case_reports.append(report)
        print(f"Vprime/t={coupling:g}: {len(energies)} eigenpairs, "
              f"max residual={report['max_relative_eigenpair_residual']:.2e}, "
              f"solve={solve_seconds:.3f} s", flush=True)

    # Certify the exact product construction against an independent 2D solve
    # for the user's actual pilot parameters, not only the small self-check.
    if args.solver == "dense":
        direct = linalg.eigvalsh(make_2d(base).toarray())
        error = float(np.max(np.abs(direct - cases[0]["energies"])))
        validation["actual_pilot_product_spectrum_max_abs_error_t"] = error
        scale = max(1.0, float(np.max(np.abs(direct))))
        if error > 1e-10 * scale:
            raise RuntimeError("The actual 2D pilot disagrees with its exact product spectrum.")

    style_plots()
    draw_potential(args.output, base, cross, cases[-1]["coupling"], args)
    draw_spectrum(args.output, cases, args)
    draw_diagnostics(args.output, cases, args)
    representatives = draw_wavefunctions(args.output, cases, args)
    parameters = vars(args).copy()
    parameters["output"] = str(args.output.resolve())
    parameters.update({"phi_x_radians": float(phases[0]), "phi_y_radians": float(phases[1]),
                       "tau_numerator": p, "tau_denominator": args.L, "tau": tau,
                       "N_sites": args.L ** 2, "hopping_matrix_element_over_t": 1.0,
                       "boundary_conditions": "Periodic in x and y; no boundary twist"})
    report = {"experiment": "Controlled separability breaking in 2D quasiperiodic tight binding",
              "source": "Goncalves PhD thesis, Sec. 3.5.2, printed p. 115, Eq. (3.50)",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "parameters": parameters, "validation": validation, "cases": case_reports,
              "representative_states": representatives,
              "dense_hamiltonian_bytes_if_allocated": 8 * args.L ** 4,
              "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                              "matplotlib": matplotlib.__version__, "platform": platform.platform(),
                              "pbs_job_id": os.environ.get("PBS_JOBID"),
                              "blas_thread_settings": {key: os.environ.get(key) for key in
                                                       ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "total_wall_seconds": perf_counter() - start,
              "limits": ["One finite rational approximant and one phase configuration per run.",
                         "The product weight measures x/y factorization, not localization.",
                         "The alpha=1 potential is not clipped or bounded artificially.",
                         "The DOS uses the full spectrum only; a plotted energy window is not renormalized.",
                         "No critical-phase, level-statistics or thermodynamic conclusion is inferred from this pilot.",
                         "No magnetic field, electron-electron interaction or superconducting term is included."]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results saved to: {args.output.resolve()}", flush=True)
    print("Finite-size pilot complete. No thermodynamic phase classification was performed.")


if __name__ == "__main__":
    main()
