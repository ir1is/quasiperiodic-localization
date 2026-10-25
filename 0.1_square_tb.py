#!/usr/bin/env python3
"""Normalni tight-binding model: kvadratna resetka, korak po korak.

Pitanje: koje su dopustene energije i pripadajuce valne funkcije?
Glavni postupak: sastavi H u bazi lokalnih orbitala i rijesi H psi = E psi.
Referenca: usporedi energije s poznatom formulom u k-prostoru.

MODEL
  H0 = epsilon0 * sum_i n_i
       - t * sum_<i,j> (c_i^dagger c_j + c_j^dagger c_i)
Svaki par <i,j> u ovoj sumi brojimo jednom. Racunamo jedan spin.
Jedna orbitala po mjestu; homogeni najblizi-susjedni hopping; a = 1.
Periodicki rubovi: suprotni rubovi resetke su povezani.

REFERENCE (otisnute stranice)
  Simon, The Oxford Solid State Basics:
    str. 101, (11.4): matrica za lanac;
    str. 177, (16.1): E(k) = epsilon0 - 2*t*cos(kx*a) - 2*t*cos(ky*a).
  Zhu, Bogoliubov-de Gennes Method and Its Applications:
    str. 38, (2.3) ima hopping + lokalnu energiju + lokalni spinski clan.
    Ovdje biramo spin-ocuvajuci NN hopping, epsilon_i = epsilon0 i Omega = 0.
  Annett, Superconductivity, Superfluids and Condensates:
    str. 138, (6.57): epsilon_k - mu je normalni dio BdG matrice.
    Ovdje racunamo samo normalni model, s mu = 0; nema BdG racuna.

POKRETANJE
  python square_tb_osnove.py
Potrebni paketi: numpy, matplotlib. Parametre mijenjaj odmah ispod importa.
Skripta ispisuje rezultate i sprema square_tb_osnove.png pokraj ove datoteke.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Spremanje slike radi i na Padobranu, bez zaslona.
import matplotlib.pyplot as plt
import numpy as np


# 1. PARAMETRI: za prvi prolaz ostavi L = 4, pa je H velicine 16 x 16.
L = 4
t = 1.0             # Energijska skala hoppinga; uzimamo t > 0.
epsilon0 = 0.0      # Jednaka lokalna energija na svakom mjestu.


def izgradi_hamiltonijan(L, t, epsilon0):
    """h_ii = epsilon0; h_ij = -t za susjede; ostalo je nula."""
    if L < 3:
        raise ValueError("Za ovaj zapis periodickih veza koristi L >= 3.")
    if t <= 0:
        raise ValueError("U ovom uvodnom primjeru koristi t > 0.")

    N = L * L
    H = np.zeros((N, N))

    for y in range(L):
        for x in range(L):
            i = x + L * y   # Mjesto (x, y) ima jedan indeks i.

            # Lokalna energija: dijagonala matrice.
            H[i, i] = epsilon0

            # Susjed u smjeru +x. Modulo povezuje zadnji i prvi stupac.
            j = (x + 1) % L + L * y
            H[i, j] = -t
            H[j, i] = -t   # Ista veza u obrnutom smjeru.

            # Susjed u smjeru +y. Modulo povezuje zadnji i prvi redak.
            j = x + L * ((y + 1) % L)
            H[i, j] = -t
            H[j, i] = -t

    return H


def analiticke_energije(L, t, epsilon0):
    """Referenca za ISTU konacnu periodicku resetku; a = 1."""
    rezultat = []
    for mx in range(L):
        kx = 2 * np.pi * mx / L
        for my in range(L):
            ky = 2 * np.pi * my / L
            E = epsilon0 - 2 * t * (np.cos(kx) + np.cos(ky))
            rezultat.append(E)
    return np.sort(rezultat)


def main():
    # 2. REALNI PROSTOR: matrica u bazi orbitala na pojedinim mjestima.
    N = L * L
    H = izgradi_hamiltonijan(L, t, epsilon0)

    # 3. NUMERICKO RJESENJE: energije rastu redom; stupci psi su vektori.
    energije, psi = np.linalg.eigh(H)
    # Za stanje alpha: H @ psi[:, alpha] = energije[alpha] * psi[:, alpha].

    # 4. REFERENCA U k-PROSTORU: provjera svih N energija.
    referenca_k = analiticke_energije(L, t, epsilon0)
    greska = np.max(np.abs(energije - referenca_k))
    if not np.allclose(energije, referenca_k, atol=1e-10, rtol=1e-12):
        raise RuntimeError("Numericke i analiticke energije se ne slazu.")

    # 5. FIZICKO ZNACENJE VEKTORA: najniza energija je energije[0].
    psi0 = psi[:, 0]
    vjerojatnost = np.abs(psi0) ** 2

    print(f"Resetka: {L} x {L}; broj orbitala po spinu: N = {N}")
    print(f"Hamiltonova matrica: {H.shape}; periodicki rubni uvjeti")
    print(f"t = {t}; epsilon0 = {epsilon0}; mu = 0")
    if N <= 25:
        print("\nIndeksi mjesta: i = x + L*y (x ide po stupcima, y po redcima)")
        print(np.arange(N).reshape(L, L))
        print("\nH: redak i i stupac j odgovaraju mjestima i i j")
        with np.printoptions(precision=2, suppress=True, linewidth=150):
            print(H)

    print("\nStanje alpha     E iz matrice       E iz k-formule")
    for alpha in range(N):
        print(f"{alpha:8d}       {energije[alpha]:12.7f}      {referenca_k[alpha]:12.7f}")
    print(f"\nNajveca razlika energija: {greska:.3e}")
    print(f"Najniza energija: {energije[0]:.7f}")
    print(f"Zbroj |psi0_i|^2: {vjerojatnost.sum():.12f}")
    if N <= 25:
        print("\n|psi0_i|^2 na resetki: vjerojatnost za jedno najnize stanje")
        with np.printoptions(precision=4, suppress=True):
            print(vjerojatnost.reshape(L, L))
    print(f"Za ovaj model ocekujemo jednoliko |psi0_i|^2 = 1/N = {1/N:.7f}")

    # 6. SLIKA: matrica je glavni rezultat; k-formula je referenca.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    skala = max(t, abs(epsilon0))
    slika = axes[0].imshow(H, cmap="RdBu_r", vmin=-skala, vmax=skala)
    axes[0].set(title="A. Hamiltonijan u realnom prostoru",
                xlabel="Indeks mjesta j", ylabel="Indeks mjesta i")
    fig.colorbar(slika, ax=axes[0], label=r"$h_{ij}$", shrink=0.8)

    alpha = np.arange(N)
    axes[1].plot(alpha, referenca_k, "o", markersize=8, markerfacecolor="none",
                 color="#d48b23", label="k-prostor: analiticka formula")
    axes[1].plot(alpha, energije, ".", markersize=6, color="#227c9d",
                 label="Realni prostor: dijagonalizacija")
    axes[1].set(title="B. Dopustene energije istog sustava",
                xlabel="Redni broj nakon sortiranja", ylabel="Energija")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    if N <= 25:
        axes[0].set_xticks(alpha)
        axes[0].set_yticks(alpha)
        axes[1].set_xticks(alpha[::2])
    fig.suptitle(f"Normalni tight-binding model | {L} x {L} mjesta | periodicki rubovi")
    putanja = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(putanja, dpi=150)
    plt.close(fig)
    print(f"\nSlika spremljena: {putanja}")


if __name__ == "__main__":
    main()
