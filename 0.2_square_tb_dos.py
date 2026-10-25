#!/usr/bin/env python3
"""0.2: normalni 2D tight binding - od energija do gustoce stanja.

Sto racunamo?
  1. Sastavimo H u REALNOM prostoru za L = 10, 20, 30.
  2. Rijesimo H psi = E psi i zadrzimo sve energije.
  3. Provjerimo ih formulom u k-prostoru za ISTI konacni sustav.
  4. Iz energija izracunamo rho(E): broj stanja po energiji i po mjestu.

Model je isti kao u 0.1: jedna orbitala po mjestu, jedan spin,
homogeni NN hopping, a = 1, periodicki rubovi. 

  H0 = epsilon0 * sum_i c_i^dagger c_i
       - t * sum_<i,j> (c_i^dagger c_j + c_j^dagger c_i)

  E(k) = epsilon0 - 2*t*(cos(kx) + cos(ky))
  kx = 2*pi*mx/L; ky = 2*pi*my/L; mx,my = 0,...,L-1.

  rho(E) = (1/N) * sum_alpha delta(E - E_alpha)


"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np



L_VRIJEDNOSTI = (10, 20, 30)
t = 1.0
epsilon0 = 0.0
eta = 0.12 * t


def izgradi_hamiltonijan(L, t, epsilon0):
    N = L * L
    H = np.zeros((N, N))

    for y in range(L):
        for x in range(L):
            i = x + L * y
            H[i, i] = epsilon0

            # Desni susjed, ukljucujuci vezu preko ruba.
            j = (x + 1) % L + L * y
            H[i, j] = -t
            H[j, i] = -t

            # Susjedni redak, ukljucujuci vezu preko ruba.
            j = x + L * ((y + 1) % L)
            H[i, j] = -t
            H[j, i] = -t

    return H


def analiticke_energije(L, t, epsilon0):
    """Sve L*L energije iz Simonove (16.1)."""
    energije = []
    for mx in range(L):
        kx = 2 * np.pi * mx / L
        for my in range(L):
            ky = 2 * np.pi * my / L
            E = epsilon0 - 2 * t * (np.cos(kx) + np.cos(ky))
            energije.append(E)
    return np.sort(energije)


def gustoca_stanja(energije, energijska_mreza, eta):
    """Svako stanje daje jedan Gaussov vrh; na kraju dijelimo s N."""
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Za prikaz gustoce stanja koristi eta > 0.")

    rho = np.zeros_like(energijska_mreza)
    for E_alpha in energije:
        rho += np.exp(-0.5 * ((energijska_mreza - E_alpha) / eta) ** 2)

    N = len(energije)
    rho /= N * np.sqrt(2 * np.pi) * eta
    return rho


def main():
    izlaz = Path(__file__).resolve().parent / "output"
    izlaz.mkdir(exist_ok=True)

    # Ista energijska mreza i isto eta za sve velicine sustava.
    # Dodamo 8*eta preko rubova pojasa da obuhvatimo Gaussove repove.
    E_mreza = np.linspace(epsilon0 - 4*t - 8*eta,
                          epsilon0 + 4*t + 8*eta, 1601)
    dE = E_mreza[1] - E_mreza[0]
    rezultati = []
    izvjestaj = ["0.2: normalni 2D tight binding; periodicki rubovi",
                 f"t={t}; epsilon0={epsilon0}; eta={eta}",
                 "DOS je normiran po mjestu i po jednom spinu.", ""]

    # 1. Za svaku velicinu radimo zasebnu REAL-SPACE dijagonalizaciju.
    for L in L_VRIJEDNOSTI:
        N = L * L
        print(f"Racunam L={L}: {N} mjesta, H je {N} x {N} ...", flush=True)
        H = izgradi_hamiltonijan(L, t, epsilon0)

        # Kao eigh iz 0.1, ali vraca samo energije. Za DOS ne trebamo psi.
        E_real = np.linalg.eigvalsh(H)
        del H

        # 2. Usporedba mora koristiti iste L, t, epsilon0 i rubne uvjete.
        E_k = analiticke_energije(L, t, epsilon0)
        greska = np.max(np.abs(E_real - E_k))
        if not np.allclose(E_real, E_k, atol=1e-10*t, rtol=1e-12):
            raise RuntimeError(f"Za L={L} energije se ne slazu s k-formulom.")

        # 3. DOS brojimo iz E_real, dakle iz spektra real-space matrice.
        rho = gustoca_stanja(E_real, E_mreza, eta)
        integral = np.sum(rho) * dE
        if not np.isclose(integral, 1.0, atol=1e-8, rtol=0):
            raise RuntimeError("Mreza ne opisuje dobro integral DOS-a.")
        rezultati.append((L, E_real, E_k, rho))

        np.savetxt(izlaz / f"0.2_energije_L{L}.csv",
                   np.column_stack((np.arange(N), E_real, E_k)),
                   delimiter=",", header="alpha,E_real,E_k", comments="",
                   fmt=("%d", "%.15e", "%.15e"))
        redak = (f"L={L:2d}; N={N:3d}; Emin={E_real[0]:.8f}; "
                 f"Emax={E_real[-1]:.8f}; max|E_real-E_k|={greska:.3e}; "
                 f"integral rho dE={integral:.10f}")
        print(redak)
        izvjestaj.append(redak)

    np.savetxt(izlaz / "0.2_dos.csv",
               np.column_stack([E_mreza] + [r[3] for r in rezultati]),
               delimiter=",", comments="",
               header="E," + ",".join(f"rho_L{r[0]}" for r in rezultati))

    # 4. Dva panela: provjera spektra i novi rezultat - gustoca stanja.
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    L, E_real, E_k, _ = rezultati[-1]
    alpha = np.arange(len(E_real))
    axes[0].plot(alpha, E_k/t, color="#b27524", linewidth=2.5,
                 label="k-formula: isti konacni sustav")
    axes[0].plot(alpha, E_real/t, ".", color="#167d9a", markersize=2,
                 label="Realni prostor: dijagonalizacija")
    axes[0].set(title=f"A. Sve energije | L = {L}, N = {L*L}",
                xlabel=r"Redni broj stanja $\alpha$ nakon sortiranja",
                ylabel=r"$E_\alpha/t$")

    boje = ("#b27524", "#ad426a", "#167d9a")
    for indeks, (L, _, _, rho) in enumerate(rezultati):
        axes[1].plot(E_mreza/t, t*rho, color=boje[indeks % len(boje)],
                     linewidth=1.9, label=f"L = {L}, N = {L*L}")
    axes[1].set(title=rf"B. Gustoca stanja | $\eta/t={eta/t:.2f}$",
                xlabel=r"$E/t$", ylabel=r"$t\,\rho_\eta(E)$", ylim=(0, None))

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Kvadratna resetka | NN hopping | periodicki rubovi | jedan spin")
    fig.savefig(izlaz / "0.2_square_tb_dos.png", dpi=170)
    plt.close(fig)

    izvjestaj += ["", "CSV datoteke sadrze sve energije i izracunati DOS.",
                  "Gaussovi repovi izvan pojasa nisu dodatna svojstvena stanja.",
                  "Eta je ista za sve L; usporedba pri fiksnoj rezoluciji."]
    (izlaz / "0.2_report.txt").write_text("\n".join(izvjestaj) + "\n", encoding="utf-8")
    print(f"\nSlika i podaci spremljeni u: {izlaz}")


if __name__ == "__main__":
    main()
