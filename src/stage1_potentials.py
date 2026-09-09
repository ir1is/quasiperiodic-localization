import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Parameters
# -------------------------

L = 200
lam = 1.0
phi = 0.0

n = np.arange(L)

# Golden-ratio-related irrational frequency
beta = (np.sqrt(5) - 1) / 2


# -------------------------
# 1. Periodic potential
# -------------------------

period = 5

V_periodic = lam * np.cos(
    2 * np.pi * n / period
)


# -------------------------
# 2. Quasiperiodic potential
# -------------------------

V_quasiperiodic = lam * np.cos(
    2 * np.pi * beta * n + phi
)


# -------------------------
# 3. Random potential
# -------------------------

rng = np.random.default_rng(seed=42)

V_random = rng.uniform(
    low=-lam,
    high=lam,
    size=L,
)


# -------------------------
# Real-space plots
# -------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
)

axes[0].plot(n, V_periodic)
axes[0].set_ylabel("V_n")
axes[0].set_title("Periodic potential")

axes[1].plot(n, V_quasiperiodic)
axes[1].set_ylabel("V_n")
axes[1].set_title("Quasiperiodic potential")

axes[2].plot(n, V_random)
axes[2].set_xlabel("Lattice site n")
axes[2].set_ylabel("V_n")
axes[2].set_title("Random potential")

plt.tight_layout()

plt.savefig(
    "outputs/stage1_real_space.png",
    dpi=200,
)

plt.show()


# -------------------------
# Fourier transforms
# -------------------------

fft_periodic = np.fft.fft(V_periodic)
fft_quasiperiodic = np.fft.fft(V_quasiperiodic)
fft_random = np.fft.fft(V_random)

frequencies = np.fft.fftfreq(L)


# Only keep non-negative frequencies
mask = frequencies >= 0


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
)

axes[0].plot(
    frequencies[mask],
    np.abs(fft_periodic[mask]),
)
axes[0].set_title("Periodic: Fourier spectrum")
axes[0].set_ylabel("|FFT|")

axes[1].plot(
    frequencies[mask],
    np.abs(fft_quasiperiodic[mask]),
)
axes[1].set_title("Quasiperiodic: Fourier spectrum")
axes[1].set_ylabel("|FFT|")

axes[2].plot(
    frequencies[mask],
    np.abs(fft_random[mask]),
)
axes[2].set_title("Random: Fourier spectrum")
axes[2].set_ylabel("|FFT|")
axes[2].set_xlabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/stage1_fourier.png",
    dpi=200,
)

plt.show()