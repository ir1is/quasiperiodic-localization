#!/usr/bin/env python3
"""Reproducible PBS sweep and JOINT all-L figures for the two square AA models.

Commands: prepare, task, status, analyze, environment-ready, check-env.
Dependencies: numpy, scipy, matplotlib. A requirements.txt file is optional.
The complete Padobran bundle also supplies hpc/submit.sh and PBS wrappers.
Heavy work runs in PBS jobs, not on login nodes.
The main output is RUN/figures/all_L_scaling.png: every L on the SAME axes.
No separate per-L figure directories are created.

Selection: a fixed number of nearest-target eigenstates, not the full fixed
energy window used in the desktop pilot. Check all_L_energy_coverage.png.
Full DOS is produced ONLY from complete small-system reference spectra.
Scientific source: Goncalves thesis, Sec. 1.2.3, Eqs. (1.26)-(1.30), pp. 19-21.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PRESETS = {
    "pilot": {"sizes": [13, 21, 34, 55], "phases": 2},
    "production": {"sizes": [13, 21, 34, 55, 89, 144, 233, 377, 610], "phases": 8},
    "benchmark": {"sizes": [233, 377, 610], "phases": 1, "V": [2.0], "models": ["nonseparable"]},
}
MODELS = ("separable", "nonseparable")
COLORS = ("#187d96", "#be8125", "#ad426a", "#6652a0", "#378051")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def requirements_hash():
    """Include an external dependency manifest when one is supplied."""
    path = ROOT / "requirements.txt"
    return file_hash(path) if path.is_file() else None


def code_hash():
    sources = {name: file_hash(ROOT / name) for name in ("aa_hpc.py", "aa_core.py")}
    sources["requirements.txt"] = requirements_hash()
    return digest(sources)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def numerator(L):
    a, b = 1, 2
    while b < L:
        a, b = b, a + b
    if b != L or L < 3:
        raise ValueError("Use Fibonacci sizes >=3.")
    return a


def prepare(args):
    preset = PRESETS[args.preset]
    sizes = sorted(args.L or preset["sizes"])
    amplitudes = sorted(set(args.V or preset.get("V", [1.0, 2.0, 4.0])))
    phases = args.phases if args.phases is not None else preset["phases"]
    if len(set(sizes)) != len(sizes) or phases < 1 or args.seed < 0:
        raise ValueError("Use distinct sizes, phases>=1, and seed>=0.")
    if (not all(math.isfinite(x) for x in [*amplitudes, args.target, args.tol])
            or min(amplitudes) < 0 or args.tol <= 0 or args.maxiter < 1
            or args.states < 2 or args.states + 10 >= min(sizes)**2
            or not 0 <= args.dense_limit <= 2500):
        raise ValueError("Invalid V, solver tolerance, state count, or dense limit.")
    pairs = [(L, numerator(L)) for L in sizes]
    config = {"sizes": sizes, "V": amplitudes, "phase_count": phases,
              "models": preset.get("models", list(MODELS)), "phase_seed": args.seed,
              "states": args.states, "target": args.target, "tol": args.tol,
              "maxiter": args.maxiter, "dense_limit": args.dense_limit,
              "selection": "k_nearest_target"}
    rng = random.Random(args.seed)
    phase_list = [[0.0, 0.0]] + [[rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)] for _ in range(phases - 1)]
    tasks = []
    for L, p in pairs:
        for V in amplitudes:
            for model in config["models"]:
                for phase_index, phase in enumerate(phase_list):
                    task = {"L": L, "p": p, "tau": p / L, "V": V, "model": model,
                            "phase_index": phase_index, "phases": phase}
                    task["solver_seed"] = int(digest({"seed": args.seed, **task})[:8], 16)
                    task["index"] = len(tasks) + 1
                    tasks.append(task)
    if (args.run_dir / "plan.json").exists():
        plan = load_plan(args.run_dir)
        if plan["config"] != config:
            raise ValueError("This run already has a different configuration. Use a new run ID.")
        print(f"Reusing immutable plan: {len(plan['tasks'])} tasks (completed tasks will be verified and skipped).")
        return
    plan = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_hash(), "config": config, "tasks": tasks}
    plan["plan_hash"] = digest(plan)
    atomic_json(args.run_dir / "plan.json", plan)
    print(f"Prepared {len(tasks)} tasks. L={sizes}; {phases} phases; {args.states} nearest-target states.")


def load_plan(run_dir):
    plan = json.loads((run_dir / "plan.json").read_text())
    expected = plan["plan_hash"]
    if digest({k: v for k, v in plan.items() if k != "plan_hash"}) != expected:
        raise ValueError("The immutable plan hash does not match.")
    if plan["code_sha256"] != code_hash():
        raise ValueError("Code changed after this plan was created. Use a new run ID with the new code.")
    return plan


def task_dir(run_dir, index):
    return run_dir / "tasks" / f"task_{index:05d}"


def completed_metadata(run_dir, task, plan):
    directory = task_dir(run_dir, task["index"])
    marker = directory / "DONE.json"
    if not marker.exists():
        return None
    done = json.loads(marker.read_text())
    if done["plan_hash"] != plan["plan_hash"]:
        raise ValueError(f"Task {task['index']}: completion belongs to a different plan.")
    for name in ("result.npz", "metadata.json"):
        if file_hash(directory / name) != done["sha256"][name]:
            raise ValueError(f"Task {task['index']}: checksum failed for {name}.")
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["task"] != task or meta["plan_hash"] != plan["plan_hash"] or meta["code_sha256"] != plan["code_sha256"]:
        raise ValueError(f"Task {task['index']}: task or code provenance mismatch.")
    return meta


def run_task(args):
    import fcntl
    import resource
    import numpy as np
    from aa_core import solve
    plan = load_plan(args.run_dir)
    if not 1 <= args.index <= len(plan["tasks"]):
        raise ValueError("Task index is outside the plan.")
    task = plan["tasks"][args.index - 1]
    directory = task_dir(args.run_dir, args.index)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed_metadata(args.run_dir, task, plan) is not None:
            print(f"Task {args.index}: verified complete, skipped.", flush=True)
            return
        try:
            arrays, meta = solve(task, plan["config"])
            meta.update({"task": task, "plan_hash": plan["plan_hash"], "code_sha256": plan["code_sha256"],
                         "host": socket.gethostname(), "pbs_job_id": os.environ.get("PBS_JOBID"),
                         "context": "PBS" if os.environ.get("PBS_JOBID") else "local",
                         "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                         "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}})
            with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
                temporary = f.name
            os.replace(temporary, directory / "result.npz")
            atomic_json(directory / "metadata.json", meta)
            atomic_json(directory / "DONE.json", {"plan_hash": plan["plan_hash"],
                        "sha256": {name: file_hash(directory / name) for name in ("result.npz", "metadata.json")}})
            (directory / "FAILED.json").unlink(missing_ok=True)
            print(f"Task {args.index}: L={task['L']} {task['model']} V/t={task['V']:g}, "
                  f"phase={task['phase_index']}, residual={meta['max_relative_residual']:.2e}, "
                  f"backend={meta['backend']}, {meta['total_seconds']:.3f}s", flush=True)
        except Exception as exc:
            atomic_json(directory / "FAILED.json", {"task": task, "error": repr(exc)})
            raise


def scan(run_dir, require_complete=False):
    plan = load_plan(run_dir)
    completed, missing, failed = [], [], []
    for task in plan["tasks"]:
        meta = completed_metadata(run_dir, task, plan)
        if meta is not None:
            completed.append(meta)
        else:
            missing.append(task["index"])
            if (task_dir(run_dir, task["index"]) / "FAILED.json").exists():
                failed.append(task["index"])
    print(f"Verified complete: {len(completed)}/{len(plan['tasks'])}; failed markers: {len(failed)}.")
    if missing:
        print(f"Missing/incomplete indices (first 20): {missing[:20]}")
    if failed:
        print(f"Failed indices (first 20): {failed[:20]}")
    if require_complete and missing:
        raise ValueError("Analysis requires the complete requested ensemble. Wait or resubmit failed/missing tasks.")
    return plan, completed


def write_rows(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def analyze(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plan, metadata = scan(args.run_dir, require_complete=True)
    config = plan["config"]
    sizes, amplitudes, models = config["sizes"], config["V"], config["models"]
    cases, rows, state_rows = [], [], []
    for meta in metadata:
        task = meta["task"]
        with np.load(task_dir(args.run_dir, task["index"]) / "result.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if len(arrays["energies"]) != config["states"]:
            raise ValueError("A result has the wrong number of targeted states.")
        row = {"model": task["model"], "V": task["V"], "L": task["L"], "phase_index": task["phase_index"],
               "log_ipr_real": float(np.log(arrays["ipr_real"]).mean()),
               "log_ipr_momentum": float(np.log(arrays["ipr_momentum"]).mean()),
               "product_weight": float(arrays["product_weight"].mean()),
               **{k: meta[k] for k in ("energy_low", "energy_high", "total_seconds", "peak_rss_MiB", "boundary_tie", "selected_degeneracy")}}
        rows.append(row)
        for i, energy in enumerate(arrays["energies"]):
            state_rows.append({"task_index": task["index"], "L": task["L"], "model": task["model"], "V": task["V"],
                               "phase_index": task["phase_index"], "energy": float(energy),
                               **{k: float(arrays[k][i]) for k in ("ipr_real", "ipr_momentum", "product_weight", "relative_residual")}})
        cases.append({"meta": meta, "task": task, "arrays": arrays})
    summaries = []
    for model in models:
        for V in amplitudes:
            for L in sizes:
                group = [r for r in rows if r["model"] == model and r["V"] == V and r["L"] == L]
                if len(group) != config["phase_count"]:
                    raise ValueError("A size is missing phase configurations.")
                s = {"model": model, "V": V, "L": L, "N": L**2, "phases": len(group)}
                for key in ("ipr_real", "ipr_momentum"):
                    logs = np.array([r["log_" + key] for r in group])
                    s[key], s[key + "_low"], s[key + "_high"] = map(float, np.exp([logs.mean(), logs.min(), logs.max()]))
                w = np.array([r["product_weight"] for r in group])
                s.update(product_weight=float(w.mean()), product_weight_low=float(w.min()), product_weight_high=float(w.max()),
                         energy_low=min(r["energy_low"] for r in group), energy_high=max(r["energy_high"] for r in group),
                         seconds=float(np.mean([r["total_seconds"] for r in group])),
                         rss_MiB=max(r["peak_rss_MiB"] for r in group))
                summaries.append(s)
    figures = args.run_dir / "figures"
    merged = args.run_dir / "merged"
    figures.mkdir(exist_ok=True)
    merged.mkdir(exist_ok=True)
    title_names = {"separable": "Separable AA", "nonseparable": "Nonseparable AA, Eq. (1.29)"}

    def points(model, V):
        return sorted((s for s in summaries if s["model"] == model and s["V"] == V), key=lambda s: s["L"])

    def finish(fig, filename, title, footer):
        fig.suptitle(title, fontsize=14, y=0.98)
        fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=8, color="#444444")
        fig.savefig(figures / filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def size_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(sizes, labels=[str(L) for L in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Linear size L (sites per direction)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.18)

    fig, axes = plt.subplots(len(models), 3, figsize=(14.5, 4 * len(models) + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.16, wspace=0.3, hspace=0.45)
    for row, model in enumerate(models):
        for vi, V in enumerate(amplitudes):
            group = points(model, V)
            for col, key in enumerate(("ipr_real", "ipr_momentum", "product_weight")):
                ax = axes[row, col]
                ax.plot(sizes, [s[key] for s in group], "o-", color=COLORS[vi % len(COLORS)], label=rf"$V/t={V:g}$")
                ax.fill_between(sizes, [s[key + "_low"] for s in group], [s[key + "_high"] for s in group], color=COLORS[vi % len(COLORS)], alpha=0.12)
        for col, label in enumerate(("Real-space IPR", "Momentum-space IPR", "SVD product weight")):
            ax = axes[row, col]
            size_axis(ax)
            ax.set_title(title_names[model] + "\n" + label)
            if col < 2:
                ax.set_yscale("log")
                ax.plot(sizes, 1 / np.array(sizes, dtype=float)**2, "k--", lw=1, label=r"Uniform: $1/L^2$")
                ax.set_ylabel("Typical inverse participation ratio")
            else:
                ax.axhline(1, color="black", ls="--", lw=1)
                ax.set(ylim=(0, 1.08), ylabel=r"Mean best product weight $w_{\rm prod}$")
            if row == 0:
                ax.legend(frameon=False, fontsize=8)
    finish(fig, "all_L_scaling.png", rf"All sizes on the same axes | {config['states']} states nearest $E/t={config['target']:g}$ | {config['phase_count']} phases",
           "IPR: geometric mean over states, then equal-weight geometric mean over phases. SVD: arithmetic means. Shading: phase min/max.\n"
           "Energy resolution changes with L: inspect all_L_energy_coverage.png. Individual-state metrics are basis dependent within degeneracies.")

    slopes = []
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models) + 0.5), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.19, wspace=0.3, hspace=0.45)
    pair_labels = [f"{a}-{b}" for a, b in zip(sizes[:-1], sizes[1:])]
    for row, model in enumerate(models):
        for vi, V in enumerate(amplitudes):
            group = points(model, V)
            for col, key in enumerate(("ipr_real", "ipr_momentum")):
                values = [-np.log(b[key] / a[key]) / np.log(b["L"] / a["L"]) for a, b in zip(group[:-1], group[1:])]
                axes[row, col].plot(np.arange(len(values)), values, "o-", color=COLORS[vi % len(COLORS)], label=rf"$V/t={V:g}$")
                for a, b, value in zip(group[:-1], group[1:], values):
                    slopes.append({"model": model, "V": V, "space": key, "L1": a["L"], "L2": b["L"], "D2_effective": float(value)})
        for col in (0, 1):
            ax = axes[row, col]
            ax.axhline(0, color="black", ls=":", lw=1)
            ax.axhline(2, color="black", ls="--", lw=1)
            ax.set(xticks=np.arange(len(pair_labels)), xticklabels=pair_labels, xlabel="Adjacent sizes L1-L2", ylabel=r"Effective exponent $D_2^{\rm eff}$",
                   title=title_names[model] + ("\nReal space" if col == 0 else "\nMomentum space"))
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.legend(frameon=False, fontsize=8)
    finish(fig, "all_L_effective_slopes.png", r"Effective slopes: $D_2^{\rm eff}=-\Delta\ln(\mathrm{IPR}_{\rm typ})/\Delta\ln L$",
           "In 2D, area-filling scaling gives D2=2; size-independent IPR gives D2=0. Values are not clipped.\n"
           "These are finite-size estimates, not automatic phase labels or converged fractal dimensions.")

    fig, axes = plt.subplots(len(models), len(amplitudes), figsize=(4.5 * len(amplitudes), 3.6 * len(models) + 1), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15, wspace=0.3, hspace=0.45)
    for row, model in enumerate(models):
        for col, V in enumerate(amplitudes):
            group = points(model, V)
            ax = axes[row, col]
            lo, hi = [s["energy_low"] - config["target"] for s in group], [s["energy_high"] - config["target"] for s in group]
            ax.fill_between(sizes, lo, hi, alpha=0.2, color=COLORS[col % len(COLORS)])
            ax.plot(sizes, lo, "o-", lw=1, color=COLORS[col % len(COLORS)])
            ax.plot(sizes, hi, "o-", lw=1, color=COLORS[col % len(COLORS)])
            ax.axhline(0, color="black", lw=0.8, ls="--")
            size_axis(ax)
            ax.set(title=title_names[model] + "\n" + rf"$V/t={V:g}$", ylabel=r"Selected energy relative to target $(E-E_0)/t$")
    finish(fig, "all_L_energy_coverage.png", "Actual selected energy ranges versus L",
           "Bounds include all selected states and all phases. Fixed k is not a fixed energy window or a full spectrum.\n"
           "A persistent gap or a changing energy range can affect apparent scaling; inspect this plot together with IPR.")

    representatives = [c for c in cases if "representative_psi" in c["arrays"]]
    vmax = max(np.max(np.abs(c["arrays"]["representative_psi"])**2) for c in representatives)
    norm = LogNorm(1e-10, vmax)
    fig, axes = plt.subplots(len(models) * len(amplitudes), len(sizes), figsize=(2.35 * len(sizes) + 1, 2.45 * len(models) * len(amplitudes) + 1), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.91, top=0.92, bottom=0.08, wspace=0.38, hspace=0.55)
    for mi, model in enumerate(models):
        for vi, V in enumerate(amplitudes):
            row = mi * len(amplitudes) + vi
            for col, L in enumerate(sizes):
                c = next(c for c in representatives if c["task"]["model"] == model and c["task"]["V"] == V and c["task"]["L"] == L)
                prob = np.abs(c["arrays"]["representative_psi"])**2
                image = axes[row, col].imshow(np.maximum(prob, norm.vmin), origin="lower", cmap="magma", norm=norm)
                i = c["meta"]["representative_index"]
                axes[row, col].set(title=rf"$L={L}$, $E/t={c['meta']['representative_energy']:.3f}$", xlabel=r"$x/a$", ylabel=r"$y/a$")
                axes[row, col].tick_params(labelsize=7)
                axes[row, col].title.set_fontsize(9)
                axes[row, col].text(0.02, 0.02, f"IPR={c['arrays']['ipr_real'][i]:.3g}\nw={c['arrays']['product_weight'][i]:.3f}",
                                   transform=axes[row, col].transAxes, color="white", fontsize=7,
                                   bbox={"facecolor": "black", "alpha": 0.7, "edgecolor": "none"})
                if col == 0:
                    axes[row, col].set_ylabel(("Separable" if model == "separable" else "Nonseparable") + rf", $V/t={V:g}$" + "\n" + r"$y/a$", fontsize=9)
    cax = fig.add_axes([0.94, 0.28, 0.01, 0.42])
    fig.colorbar(image, cax=cax, label=r"Site probability $|\psi(x,y)|^2$ (shared log scale)")
    finish(fig, "all_L_wavefunctions.png", "Spatial states at ALL sizes in ONE figure",
           "Columns: L. Rows: model and V/t. First phase configuration, one nearest-target state per panel.\n"
           "These are independently selected eigenstates, not one tracked state. Scaling statistics use every selected state and phase.")

    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.5 * len(models) + 1), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.15, hspace=0.45, wspace=0.3)
    for row, model in enumerate(models):
        for vi, V in enumerate(amplitudes):
            group = points(model, V)
            for col, key in enumerate(("seconds", "rss_MiB")):
                ax = axes[row, col]
                ax.plot(sizes, [s[key] for s in group], "o-", color=COLORS[vi % len(COLORS)], label=rf"$V/t={V:g}$")
                size_axis(ax)
                ax.set_yscale("log")
                ax.set(title=title_names[model], ylabel="Mean solve + diagnostics (s)" if col == 0 else "Maximum process peak RSS (MiB)")
                if row == 0:
                    ax.legend(frameon=False, fontsize=8)
    contexts = sorted(set(m["context"] for m in metadata))
    finish(fig, "hpc_cost.png", "Recorded computational cost versus L",
           f"Execution context: {', '.join(contexts)}. Local demonstration timings are not Padobran benchmarks.\n"
           "Separable models exploit exact 1D products. Sparse LU fill-in, Lanczos vectors, and SVD work add costs beyond CSR storage.")

    dense_sizes = [L for L in sizes if all("full_spectrum" in c["arrays"] for c in cases if c["task"]["L"] == L)]
    if dense_sizes:
        dense_cases = [c for c in cases if c["task"]["L"] in dense_sizes]
        low = min(c["arrays"]["full_spectrum"].min() for c in dense_cases) - 0.6
        high = max(c["arrays"]["full_spectrum"].max() for c in dense_cases) + 0.6
        grid = np.linspace(low, high, 1801)
        fig, axes = plt.subplots(len(models), len(amplitudes), figsize=(4.5 * len(amplitudes), 3.5 * len(models) + 1), squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15, wspace=0.3, hspace=0.45)
        for row, model in enumerate(models):
            for col, V in enumerate(amplitudes):
                for li, L in enumerate(dense_sizes):
                    group = [c for c in dense_cases if c["task"]["model"] == model and c["task"]["V"] == V and c["task"]["L"] == L]
                    rho = np.zeros_like(grid)
                    for c in group:
                        E = c["arrays"]["full_spectrum"]
                        for start in range(0, len(E), 128):
                            rho += np.exp(-0.5 * ((grid[:, None] - E[None, start:start + 128]) / .08)**2).sum(axis=1) / (len(group) * len(E) * .08 * np.sqrt(2 * np.pi))
                    area = np.sum(.5 * (rho[1:] + rho[:-1]) * np.diff(grid))
                    if abs(area - 1) > 1e-5:
                        raise ValueError("Complete-reference DOS failed normalization.")
                    axes[row, col].plot(grid, rho, label=f"L={L}", color=COLORS[li % len(COLORS)])
                axes[row, col].set(title=title_names[model] + "\n" + rf"$V/t={V:g}$", xlabel=r"Energy $E/t$", ylabel=r"$t\rho(E)$ per site, one spin", ylim=(0, None))
                axes[row, col].legend(frameon=False, fontsize=8)
        finish(fig, "small_L_DOS.png", r"Full-spectrum reference DOS at small L | $\eta/t=0.08$",
               "Computed from COMPLETE spectra, normalized to area 1. No large-system DOS is inferred from the targeted eigenstates.")
    write_rows(merged / "observables.csv", state_rows)
    write_rows(merged / "per_phase.csv", rows)
    write_rows(merged / "size_summary.csv", summaries)
    write_rows(merged / "effective_slopes.csv", slopes)
    atomic_json(merged / "analysis.json", {"plan_hash": plan["plan_hash"], "code_sha256": code_hash(),
                "complete_tasks": len(metadata), "contexts": contexts, "config": config,
                "boundary_tie_tasks": sum(m["boundary_tie"] for m in metadata),
                "degenerate_tasks": sum(m["selected_degeneracy"] for m in metadata),
                "interpretation": "k-nearest sampling; inspect energy coverage; finite-size slopes are not phase labels"})
    print(f"JOINT main figure: {(figures / 'all_L_scaling.png').resolve()}")
    print(f"JOINT spatial figure: {(figures / 'all_L_wavefunctions.png').resolve()}")


def environment(write=False):
    import numpy as np
    import scipy.linalg
    np.testing.assert_allclose(scipy.linalg.eigvalsh(np.diag([1., 2.])), [1., 2.])
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}
    stamp = {"requirements_sha256": requirements_hash(), "versions": versions,
             "python_prefix": str(Path(sys.prefix).resolve())}
    path = ROOT / ".environment-ready.json"
    if write:
        atomic_json(path, stamp)
        print("SETUP_OK")
    elif not path.exists() or json.loads(path.read_text()) != stamp:
        raise ValueError("Environment is not ready or changed. Finish bootstrap before submitting tasks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--preset", choices=PRESETS, default="pilot")
    prep.add_argument("--L", type=int, nargs="+")
    prep.add_argument("--V", type=float, nargs="+")
    prep.add_argument("--phases", type=int)
    prep.add_argument("--seed", type=int, default=2026)
    prep.add_argument("--states", type=int, default=32)
    prep.add_argument("--target", type=float, default=0.)
    prep.add_argument("--tol", type=float, default=1e-10)
    prep.add_argument("--maxiter", type=int, default=20000)
    prep.add_argument("--dense-limit", type=int, default=1600)
    task = sub.add_parser("task")
    task.add_argument("--run-dir", type=Path, required=True)
    task.add_argument("--index", type=int, required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--require-complete", action="store_true")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--run-dir", type=Path, required=True)
    count = sub.add_parser("count")
    count.add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("environment-ready")
    sub.add_parser("check-env")
    args = parser.parse_args()
    try:
        if args.command == "prepare": prepare(args)
        elif args.command == "task": run_task(args)
        elif args.command == "status": scan(args.run_dir, args.require_complete)
        elif args.command == "analyze": analyze(args)
        elif args.command == "count": print(len(load_plan(args.run_dir)["tasks"]))
        else: environment(write=args.command == "environment-ready")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
