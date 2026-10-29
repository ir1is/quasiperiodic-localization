#!/usr/bin/env python3
"""Exercise 11.9: anisotropic nearest-neighbour tight binding in 2D.

BOOK: derive the dispersion and expand near its minimum.
EXTENSIONS: finite real-space diagonalization, hopping-ratio study, DOS,
and a finite-difference estimate of the effective masses from real-space H.

MODEL AND CONVENTIONS
  One orbital per site; one spin; a square L x L sample with periodic edges.
  H_ii = epsilon0; H_ij = -tx along x, -ty along y, zero otherwise.
  tx > 0, ty >= 0; r = ty/tx. Both are NEAREST-neighbour hoppings.
  E(qx,qy) = epsilon0 - 2*tx*cos(qx) - 2*ty*cos(qy).
  qx = kx*a, qy = ky*a; allowed q = 2*pi*m/L.
  Exercise 11.9 calls the signed matrix elements t1 and t2:
  in that literal notation, t1=-tx and t2=-ty for our convention.

RUN
  python 0.3_anisotropic_tb.py --L 20 --ratios 0.5 --output output/0.3_pilot
  python 0.3_anisotropic_tb.py
  
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TX = 1.0
EPSILON0 = 0.0


def build_hamiltonian(L, tx, ty, epsilon0):
    """The construction from 0.1, with separate x and y bond strengths."""
    N = L * L
    H = np.zeros((N, N))
    for y in range(L):
        for x in range(L):
            i = x + L*y
            H[i, i] = epsilon0
            j = (x + 1) % L + L*y
            H[i, j] = H[j, i] = -tx
            j = x + L*((y + 1) % L)
            H[i, j] = H[j, i] = -ty
    return H


def dispersion(qx, qy, tx, ty, epsilon0):
    """Analytic solution of Exercise 11.9 in our negative-hopping convention."""
    return epsilon0 - 2*tx*np.cos(qx) - 2*ty*np.cos(qy)


def analytic_spectrum(L, tx, ty, epsilon0):
    """All eigenvalues of the SAME finite periodic sample, sorted."""
    q = 2*np.pi*np.arange(L)/L
    qx, qy = np.meshgrid(q, q)
    return np.sort(dispersion(qx, qy, tx, ty, epsilon0).ravel())


def density_of_states(energies, energy_grid, eta):
    """Gaussian-broadened delta sum, normalized to integral one."""
    rho = np.zeros_like(energy_grid)
    for energy in energies:
        rho += np.exp(-0.5*((energy_grid - energy)/eta)**2)
    return rho/(len(energies)*np.sqrt(2*np.pi)*eta)


def plane_wave_energy(H, L, qx, qy):
    """Label a state by allowed momentum and evaluate <psi_q|H|psi_q>.

    A sorted eigenvalue index does not specify momentum. Instead we use
    the normalized plane-wave ansatz, Eq. (11.5), generalized to 2D.
    Only allowed momenta 2*pi*m/L are used with this periodic matrix.
    """
    y, x = np.indices((L, L))
    psi = np.exp(-1j*(qx*x + qy*y)).ravel()/L
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    if np.linalg.norm(Hpsi - energy*psi) > 1e-10*max(1.0, np.max(abs(H))):
        raise RuntimeError("The plane wave is not an eigenstate: check momentum and H.")
    return float(energy)


def masses_from_real_space(H, L, tx, ty):
    """Central finite difference of E(q), evaluated using real-space H.

    Since q=k*a, m/m0 = tx / (d**2 E/dq**2).
    The finite step dq=2*pi/L introduces a controlled finite-size error.
    """
    dq = 2*np.pi/L
    E0 = plane_wave_energy(H, L, 0, 0)
    cx = (plane_wave_energy(H, L, dq, 0)
          + plane_wave_energy(H, L, -dq, 0) - 2*E0)/dq**2
    cy = (plane_wave_energy(H, L, 0, dq)
          + plane_wave_energy(H, L, 0, -dq) - 2*E0)/dq**2
    return tx/cx, np.inf if ty == 0 else tx/cy


def plot_spectra_and_dos(cases, energy_grid, tx, eta, L, output):
    fig, axes = plt.subplots(len(cases), 2, figsize=(11.4, 2.8*len(cases)),
                             squeeze=False, layout="constrained")
    for row, case in enumerate(cases):
        ratio, E, E_k, rho = case["ratio"], case["E"], case["E_k"], case["rho"]
        ax, bx = axes[row]
        ax.plot(np.arange(len(E)), E_k/tx, color="#be8125", lw=2.5,
                label="Analytic: same finite sample")
        ax.plot(np.arange(len(E)), E/tx, ".", color="#187d96", ms=2.5,
                label="Real-space diagonalization")
        ax.set(title=rf"Spectrum | $r=t_y/t_x={ratio:g}$",
               xlabel=r"Sorted eigenstate index $\alpha$ (dimensionless)",
               ylabel=r"$E_\alpha/t_x$ (dimensionless)")
        bx.plot(energy_grid/tx, tx*case["rho_k"], "--", color="#be8125", lw=2.8,
                label="DOS from analytic k-spectrum: same eta")
        bx.plot(energy_grid/tx, tx*rho, color="#187d96", lw=1.6,
                label="DOS from real-space eigenvalues")
        if ratio > 0:
            saddles = np.unique(np.round([EPSILON0 - 2*(tx-case["ty"]),
                                         EPSILON0 + 2*(tx-case["ty"])], 12))
            for index, saddle in enumerate(saddles):
                bx.axvline(saddle/tx, color="#ad426a", ls="--", lw=1,
                           label="Analytic saddle energy" if index == 0 else None)
        else:
            bx.text(0.5, 0.94, "Decoupled 1D chains", ha="center", va="top",
                    transform=bx.transAxes, fontsize=9)
        bx.set(title=rf"DOS | $\eta/t_x={eta/tx:.2f}$",
               xlabel=r"$E/t_x$ (dimensionless)",
               ylabel=r"$t_x\rho_\eta(E)$ (per site, one spin)", ylim=(0, None))
        for panel in (ax, bx):
            panel.grid(alpha=0.18)
            panel.legend(fontsize=7, loc="best")
    fig.suptitle(f"Anisotropic NN tight binding | {L} x {L} sites | periodic boundaries")
    fig.savefig(output / "spectrum_and_dos.png", dpi=160)
    plt.close(fig)


def plot_edges_and_masses(cases, L, tx, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    rmax = max(1.0, max(case["ratio"] for case in cases))
    r = np.linspace(0, rmax, 301)
    axes[0].plot(r, EPSILON0/tx - 2*(1+r), color="#187d96", label="Analytic band minimum")
    axes[0].plot(r, EPSILON0/tx + 2*(1+r), color="#be8125", label="Analytic band maximum")
    for index, case in enumerate(cases):
        axes[0].plot([case["ratio"]]*2, [case["E"][0]/tx, case["E"][-1]/tx],
                     "o", color="#ad426a", ms=5,
                     label="Real-space eigenvalues" if index == 0 else None)
    axes[0].set(title="A. Band edges and bandwidth",
                xlabel=r"$r=t_y/t_x$ (dimensionless)", ylabel=r"$E/t_x$ (dimensionless)")

    positive_r = np.linspace(0.05, rmax, 301)
    axes[1].plot(r, np.full_like(r, 0.5), color="#187d96", label=r"Analytic $m_x^*/m_0$")
    axes[1].plot(positive_r, 1/(2*positive_r), color="#be8125", label=r"Analytic $m_y^*/m_0$")
    for index, case in enumerate(cases):
        axes[1].plot(case["ratio"], case["mx_fd"], "o", color="#187d96",
                     label=r"Real-space finite difference: $x$" if index == 0 else None)
        if case["ratio"] > 0:
            axes[1].plot(case["ratio"], case["my_fd"], "s", color="#be8125",
                         label=r"Real-space finite difference: $y$" if not any(
                             c["ratio"] > 0 for c in cases[:index]) else None)
    axes[1].text(0.04, 0.96, r"$r=0$: flat in $q_y$, $1/m_y^*=0$",
                 transform=axes[1].transAxes, va="top", fontsize=9)
    axes[1].set(title="B. Effective masses near the band minimum",
                xlabel=r"$r=t_y/t_x$ (dimensionless)",
                ylabel=r"$m^*/m_0$ (dimensionless)", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(rf"$m_0=\hbar^2/(t_xa^2)$ | numerical curvature step $\Delta q=2\pi/{L}$")
    fig.savefig(output / "band_edges_and_masses.png", dpi=160)
    plt.close(fig)


def plot_dispersion_and_curvature(case, L, tx, output):
    ty, ratio = case["ty"], case["ratio"]
    q = np.linspace(-np.pi, np.pi, 181)
    qx, qy = np.meshgrid(q, q)
    E = dispersion(qx, qy, tx, ty, EPSILON0)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    image = axes[0].pcolormesh(qx, qy, E/tx, shading="auto", cmap="RdBu_r")
    axes[0].contour(qx, qy, E/tx, levels=7, colors="black", linewidths=0.5, alpha=0.45)
    fig.colorbar(image, ax=axes[0], label=r"$E/t_x$ (dimensionless)")
    axes[0].set(title="A. Analytic dispersion: first Brillouin zone",
                xlabel=r"$q_x=k_xa$ (dimensionless)", ylabel=r"$q_y=k_ya$ (dimensionless)",
                aspect="equal")
    for axis in (axes[0].xaxis, axes[0].yaxis):
        axis.set_ticks([-np.pi, 0, np.pi])
        axis.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

    cuts = case["cuts"]
    qlimit = max(1.05, 1.05*np.max(abs(cuts[:, 0])))
    qc = np.linspace(-qlimit, qlimit, 301)
    axes[1].plot(qc, 2*(1-np.cos(qc)), color="#187d96", label="Exact x-direction cut")
    axes[1].plot(qc, 2*ratio*(1-np.cos(qc)), color="#be8125", label="Exact y-direction cut")
    axes[1].plot(qc, qc**2, "--", color="#187d96", label=r"Quadratic: $q^2$")
    axes[1].plot(qc, ratio*qc**2, "--", color="#be8125", label=r"Quadratic: $rq^2$")
    axes[1].plot(cuts[:, 0], cuts[:, 1], "o", color="#187d96", ms=4,
                 label="Real-space plane-wave energies: x")
    axes[1].plot(cuts[:, 0], cuts[:, 2], "s", color="#be8125", ms=4,
                 label="Real-space plane-wave energies: y")
    axes[1].set(title="B. Quadratic approximation near the minimum",
                xlabel=r"$q=k_\nu a$ (dimensionless), $\nu=x$ or $y$",
                ylabel=r"$[E(q)-E_{\min}]/t_x$ (dimensionless)", ylim=(0, None))
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=7, loc="upper center")
    fig.suptitle(rf"Exercise 11.9 | $r=t_y/t_x={ratio:g}$ | real-space sample $L={L}$")
    fig.savefig(output / "dispersion_and_curvature.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=20, help="Even number of sites per direction, at least 4.")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.0], help="Values of r=ty/tx.")
    parser.add_argument("--eta", type=float, default=0.12, help="Gaussian standard deviation in units of tx.")
    parser.add_argument("--output", type=Path, help="Directory for figures, CSV data and report.")
    args = parser.parse_args()
    if args.L < 4 or args.L % 2:
        parser.error("Use an even L >= 4, so the finite grid includes the band extrema.")
    if not np.isfinite(args.eta) or args.eta <= 0:
        parser.error("eta must be finite and positive.")
    if any(not np.isfinite(r) or r < 0 for r in args.ratios):
        parser.error("Hopping ratios must be finite and nonnegative.")

    L, tx, eta = args.L, TX, args.eta*TX
    ratios = list(dict.fromkeys(args.ratios))
    output = args.output or Path(__file__).resolve().parent / "output" / f"0.3_anisotropic_tb_L{L}"
    output.mkdir(parents=True, exist_ok=True)
    half_width = 2*tx*(1 + max(ratios))
    energy_grid = np.linspace(EPSILON0-half_width-8*eta, EPSILON0+half_width+8*eta, 2001)
    dE = energy_grid[1] - energy_grid[0]
    cases, rows = [], []
    report = ["Exercise 11.9: anisotropic nearest-neighbour tight binding",
              f"L={L}; N={L*L}; tx={tx}; epsilon0={EPSILON0}; eta/tx={args.eta}",
              "Periodic boundaries; equal lattice spacings ax=ay=a; one orbital and one spin.",
              "Mass unit m0=hbar^2/(tx*a^2); masses are not specified in kilograms.",
              "Numerical masses use real-space H and allowed momenta +/-2*pi/L.",
              "Their finite-difference bias decreases as L increases.", ""]

    for ratio in ratios:
        ty = ratio*tx
        print(f"Computing r={ratio:g}: L={L}, H is {L*L} x {L*L} ...", flush=True)
        H = build_hamiltonian(L, tx, ty, EPSILON0)
        E = np.linalg.eigvalsh(H)
        E_k = analytic_spectrum(L, tx, ty, EPSILON0)
        error = np.max(abs(E-E_k))/tx
        if error > 1e-10:
            raise RuntimeError("Real-space and analytic finite-sample spectra disagree.")
        rho = density_of_states(E, energy_grid, eta)
        rho_k = density_of_states(E_k, energy_grid, eta)
        dos_error = tx*np.max(abs(rho-rho_k))
        if dos_error > 1e-10:
            raise RuntimeError("Real-space and analytic finite-sample DOS disagree.")
        area = np.sum(rho)*dE
        if not np.isclose(area, 1.0, atol=1e-8, rtol=0):
            raise RuntimeError("The sampled DOS is not normalized; refine the energy grid.")
        mx_fd, my_fd = masses_from_real_space(H, L, tx, ty)
        mx_exact, my_exact = 0.5, np.inf if ratio == 0 else 1/(2*ratio)
        dq = 2*np.pi/L
        momenta = dq*np.arange(-min(3, (L-1)//2), min(3, (L-1)//2)+1)
        E0 = EPSILON0 - 2*(tx+ty)
        cuts = np.array([(q, (plane_wave_energy(H,L,q,0)-E0)/tx,
                         (plane_wave_energy(H,L,0,q)-E0)/tx) for q in momenta])
        del H
        case = dict(ratio=ratio, ty=ty, E=E, E_k=E_k, rho=rho, rho_k=rho_k,
                    mx_fd=mx_fd, my_fd=my_fd, cuts=cuts)
        cases.append(case)
        rows.append([ratio, L, L*L, E[0]/tx, E[-1]/tx, (E[-1]-E[0])/tx,
                     error, dos_error, area, mx_exact, mx_fd, my_exact, my_fd])
        tag = f"r{ratio:g}".replace(".", "p")
        np.savetxt(output / f"spectrum_{tag}.csv", np.column_stack((np.arange(L*L), E/tx, E_k/tx)),
                   delimiter=",", comments="", header="alpha,E_real_over_tx,E_analytic_over_tx")
        np.savetxt(output / f"curvature_{tag}.csv", cuts, delimiter=",", comments="",
                   header="q_k_times_a,deltaE_x_over_tx,deltaE_y_over_tx")
        line = (f"r={ratio:g}: Emin/tx={E[0]/tx:.8f}, Emax/tx={E[-1]/tx:.8f}, "
                f"W/tx={(E[-1]-E[0])/tx:.8f}, max spectral error/tx={error:.3e}, DOS integral={area:.10f}\n"
                f"  max DOS difference times tx={dos_error:.3e}\n"
                f"  mx/m0: exact={mx_exact:.8g}, finite difference={mx_fd:.8g}; "
                f"my/m0: exact={my_exact:.8g}, finite difference={my_fd:.8g}")
        report.append(line)
        print(line)

    np.savetxt(output / "summary.csv", rows, delimiter=",", comments="",
               header="ratio_ty_over_tx,L,N,Emin_over_tx,Emax_over_tx,bandwidth_over_tx,max_error_over_tx,max_DOS_difference_times_tx,DOS_integral,mx_exact_over_m0,mx_fd_over_m0,my_exact_over_m0,my_fd_over_m0")
    dos_columns = [energy_grid/tx]
    dos_headers = ["E_over_tx"]
    for case in cases:
        dos_columns += [tx*case["rho"], tx*case["rho_k"]]
        dos_headers += [f"tx_rho_real_r{case['ratio']:g}", f"tx_rho_k_r{case['ratio']:g}"]
    np.savetxt(output / "dos.csv", np.column_stack(dos_columns), delimiter=",",
               comments="", header=",".join(dos_headers))
    plot_spectra_and_dos(cases, energy_grid, tx, eta, L, output)
    plot_edges_and_masses(cases, L, tx, output)
    focus = next((c for c in cases if np.isclose(c["ratio"], 0.5)), cases[0])
    plot_dispersion_and_curvature(focus, L, tx, output)
    report += ["", "DOS uses the real-space eigenvalues, per site and one spin.",
               "It is compared with the DOS of the analytic finite k-grid, using the same Gaussian width.",
               "Gaussian tails outside the band do not represent additional eigenvalues.",
               "Saddle markers show analytic energies; finite-size DOS maxima need not coincide with them.",
               "At r=0 the sample is L independent chains: E has no qy dependence and my is infinite.",
               "The dispersion/curvature figure uses r=0.5 when present, otherwise the first ratio.",
               "References: Simon, pp. 99 (11.2), 101 (11.4)-(11.5), 103 (effective mass),",
               "110 (Exercise 11.9), 177 (16.1), 30 (DOS discussion around 4.8-4.10)."]
    (output / "report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(f"\nFigures, raw data and report saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
