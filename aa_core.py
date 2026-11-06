"""Square-lattice AA numerics, t=a=1, one spin sector, periodic boundaries.

Source: M. Goncalves, PhD thesis (2023), Sec. 1.2.3, pp. 19-21.
Separable: H is the negative of the WHOLE Eq. (1.26); eigenvectors unchanged.
Nonseparable: Eq. (1.29), exactly at zero phases; shifted extension otherwise.
Both use -t nearest-neighbour hopping. No magnetic field, pairing, or interaction.

HPC experiment: select k eigenstates closest to a common target E/t at each L.
This is NOT the previous full fixed-energy-window average. Its energy resolution
changes with L. Always inspect the saved energy ranges before interpreting slopes.
Small nonseparable systems use dense ED; large ones use ARPACK shift-invert.
Separable systems use exact 1D products without constructing N-by-N eigenvectors.
Individual-state IPR/SVD diagnostics depend on basis choices within degeneracies.
"""

from time import perf_counter
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigsh, splu

MODELS = ("separable", "nonseparable")


def fibonacci_numerator(L):
    a, b = 1, 2
    while b < L:
        a, b = b, a + b
    if b != L or L < 3:
        raise ValueError("Use Fibonacci sizes >=3, e.g. 13 21 34 55 89 144 233 377 610.")
    return a


def onsite_potential(L, p, V, phases, model):
    theta_x = 2 * np.pi * (p / L) * np.arange(L) + phases[0]
    theta_y = 2 * np.pi * (p / L) * np.arange(L) + phases[1]
    ux, uy = -V * np.cos(theta_x), -V * np.cos(theta_y)
    if model == "separable":
        onsite = ux[None, :] + uy[:, None]
    elif model == "nonseparable":
        onsite = -2 * V * np.cos(theta_y)[:, None] * np.cos(theta_x)[None, :]
    else:
        raise ValueError("Unknown model")
    return onsite, ux, uy


def square_hamiltonian(onsite):
    """Five nonzero entries per row: onsite and four nearest neighbours."""
    L = onsite.shape[0]
    sites = np.arange(L * L).reshape(L, L)
    i = sites.ravel()
    right, up = np.roll(sites, -1, axis=1).ravel(), np.roll(sites, -1, axis=0).ravel()
    rows, cols = np.concatenate((i, right, i, up, i)), np.concatenate((right, i, up, i, i))
    values = np.concatenate((-np.ones(4 * L * L), onsite.ravel()))
    return sparse.coo_matrix((values, (rows, cols)), shape=(L * L, L * L)).tocsr()


def chain(onsite):
    h = np.diag(onsite.copy())
    for i in range(len(onsite)):
        j = (i + 1) % len(onsite)
        h[i, j] = h[j, i] = -1.0
    return h


def nearest_order(energies, target):
    # Stable, deterministic tie-breaking; ties are still physically degenerate.
    return np.argsort(np.abs(energies - target), kind="stable")


def product_candidates(ux, uy, target, count):
    ex, vx = linalg.eigh(chain(ux))
    ey, vy = linalg.eigh(chain(uy))
    all_energies = (ey[:, None] + ex[None, :]).ravel()
    order = nearest_order(all_energies, target)[:count]
    L = len(ux)
    # Only count product vectors, never a full (L^2)-by-(L^2) Kronecker matrix.
    vectors = np.column_stack([np.kron(vy[:, j // L], vx[:, j % L]) for j in order])
    return all_energies[order], vectors, np.sort(all_energies)


def sparse_candidates(H, target, count, seed, tol, maxiter):
    """Factor H-sigma*I once, then reuse its inverse action in Lanczos.

    A recorded tiny shift avoids an exactly singular factorization at E=target.
    Extra candidates are computed, then selection is made relative to target.
    SciPy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
    """
    N = H.shape[0]
    identity = sparse.eye(N, format="csc")
    factor_started = perf_counter()
    for offset in (1e-6, 1e-5, 1e-4):
        sigma = target + offset * max(1.0, abs(target))
        try:
            lu = splu((H - sigma * identity).tocsc())
            break
        except RuntimeError:
            if offset == 1e-4:
                raise
    factor_seconds = perf_counter() - factor_started
    operator = LinearOperator(H.shape, matvec=lu.solve, dtype=np.float64)
    v0 = np.random.default_rng(seed).normal(size=N)
    E, vectors = eigsh(H, k=count, sigma=sigma, which="LM", OPinv=operator,
                      v0=v0, ncv=min(N - 1, max(2 * count + 1, 30)),
                      tol=tol, maxiter=maxiter)
    return E, vectors, {"sigma_used": sigma, "lu_nnz": int(lu.L.nnz + lu.U.nnz),
                        "factor_seconds": factor_seconds}


def observables(H, energies, vectors, L, exact_products=False):
    norms = np.sum(np.abs(vectors)**2, axis=0)
    residuals = np.linalg.norm(H @ vectors - vectors * energies, axis=0) / np.maximum(1, np.abs(energies))
    orthogonality = float(np.max(np.abs(vectors.T.conj() @ vectors - np.eye(len(energies)))))
    if np.max(np.abs(norms - 1)) > 1e-7 or residuals.max() > 1e-7 or orthogonality > 1e-7:
        raise RuntimeError("Eigenpair normalization, orthogonality, or residual check failed.")
    ipr_r, ipr_k, weights = [], [], []
    for index in range(len(energies)):
        psi = vectors[:, index].reshape(L, L) / np.sqrt(norms[index])
        ipr_r.append(float(np.sum(np.abs(psi)**4)))
        ipr_k.append(float(np.sum(np.abs(np.fft.fft2(psi, norm="ortho"))**4)))
        # Product rank is known exactly. Verify the SVD on the first small product.
        if exact_products:
            if index == 0 and L <= 89 and abs(linalg.svdvals(psi)[0]**2 - 1) > 1e-9:
                raise RuntimeError("Product-basis SVD verification failed.")
            weights.append(1.0)
        else:
            weights.append(float(linalg.svdvals(psi, check_finite=False)[0]**2))
    ipr_r, ipr_k, weights = map(np.asarray, (ipr_r, ipr_k, weights))
    if (min(ipr_r.min(), ipr_k.min()) < 1 / L**2 - 1e-9
            or max(ipr_r.max(), ipr_k.max()) > 1 + 1e-9
            or weights.min() < 1 / L - 1e-9 or weights.max() > 1 + 1e-9):
        raise RuntimeError("IPR or product weight is outside its mathematical bounds.")
    return {"ipr_real": ipr_r, "ipr_momentum": ipr_k, "product_weight": weights,
            "relative_residual": residuals}, orthogonality


def solve(task, settings):
    started = perf_counter()
    L, p, V, model = task["L"], task["p"], task["V"], task["model"]
    target, k = settings["target"], settings["states"]
    N = L**2
    if k < 2 or k + 10 >= N:
        raise ValueError("Require 2 <= states < L^2-10.")
    onsite, ux, uy = onsite_potential(L, p, V, task["phases"], model)
    H = square_hamiltonian(onsite)
    count = min(N - 2, k + max(8, k // 2))
    exact_products = model == "separable" or V == 0
    extra = {"sigma_used": None, "lu_nnz": None, "factor_seconds": 0.0}
    solve_started = perf_counter()
    full_spectrum = None
    if exact_products:
        E, vectors, all_E = product_candidates(ux, uy, target, count)
        backend = "1D product eigenbasis"
        if N <= settings["dense_limit"]:
            full_spectrum = all_E
            direct = linalg.eigvalsh(H.toarray())
            if np.max(np.abs(direct - all_E)) > 1e-8:
                raise RuntimeError("Product spectrum disagrees with independent 2D construction.")
    elif N <= settings["dense_limit"]:
        all_E, all_vectors = linalg.eigh(H.toarray(), driver="evd")
        order = nearest_order(all_E, target)[:count]
        E, vectors = all_E[order], all_vectors[:, order]
        full_spectrum = all_E
        backend = "dense reference"
    else:
        E, vectors, extra = sparse_candidates(H, target, count, task["solver_seed"], settings["tol"], settings["maxiter"])
        backend = "sparse shift-invert"
    solve_seconds = perf_counter() - solve_started
    order = nearest_order(E, target)
    boundary_tie = bool(np.isclose(abs(E[order[k - 1]] - target), abs(E[order[k]] - target), atol=1e-8, rtol=1e-7))
    E, vectors = E[order[:k]], vectors[:, order[:k]]
    sort = np.argsort(E, kind="stable")
    E, vectors = E[sort], vectors[:, sort]
    metric_started = perf_counter()
    metrics, orthogonality = observables(H, E, vectors, L, exact_products)
    metric_seconds = perf_counter() - metric_started
    representative = int(np.argmin(np.abs(E - target)))
    arrays = {"energies": E, **metrics}
    if task["phase_index"] == 0:
        arrays["representative_psi"] = vectors[:, representative].reshape(L, L)
    if full_spectrum is not None:
        arrays["full_spectrum"] = full_spectrum
    metadata = {"backend": backend, "exact_product_basis": exact_products,
                "matrix_shape": [N, N], "matrix_nnz": int(H.nnz), "states": k,
                "energy_low": float(E.min()), "energy_high": float(E.max()),
                "max_distance_to_target": float(np.max(np.abs(E - target))),
                "boundary_tie": boundary_tie,
                "selected_degeneracy": bool(np.any(np.diff(E) < 1e-8)),
                "max_relative_residual": float(metrics["relative_residual"].max()),
                "orthogonality_error": orthogonality,
                "representative_index": representative,
                "representative_energy": float(E[representative]),
                "solve_seconds": solve_seconds, "diagnostic_seconds": metric_seconds,
                "total_seconds": perf_counter() - started, **extra}
    return arrays, metadata
