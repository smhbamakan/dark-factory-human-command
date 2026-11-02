"""Simulation experiments X1-X10 (Table 4). Usage: python3 experiments.py X1 X2 ..."""
import sys, time, json, math, itertools
import numpy as np, pandas as pd
from dataclasses import replace, asdict
from multiprocessing import Pool
from mhc2 import P, simulate

import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(ROOT, "results") + "/"; _os.makedirs(OUT, exist_ok=True)
BASE = P()

def _job(args):
    tag, kw, seed, keep_series = args
    p = replace(BASE, **kw)
    r = simulate(p, seed=seed)
    row = {k: v for k, v in r.items() if not k.startswith("_")}
    row.update(tag); row["seed"] = seed
    if keep_series: row["_series"] = r.get("_series")
    return row

def run_jobs(jobs, name, keep_series=False):
    t0 = time.time()
    with Pool(2) as pool:
        rows = pool.map(_job, [(tag, kw, s, keep_series) for tag, kw, s in jobs], chunksize=8)
    df = pd.DataFrame(rows)
    if keep_series:
        ser = np.vstack(df.pop("_series").values); np.save(OUT + f"{name}_series.npy", ser)
    df.to_csv(OUT + f"{name}.csv", index=False)
    print(name, len(rows), "runs", round(time.time() - t0, 1), "s", flush=True)
    return df

TAUS = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]
SEEDS = range(1000, 1020)

def X1():   # darkness curve, factorial N x k
    jobs = [(dict(N=N, k=k, tau=t), dict(N=N, k=k, tau=t), s)
            for N in (40, 80) for k in (1, 2) for t in TAUS for s in SEEDS]
    run_jobs(jobs, "X1")

def X1b():  # counterexamples / boundary conditions for P1
    variants = {
        "baseline": {},
        "AI_superior": dict(a_A=4.0, b_A=0.5),
        "ample_capacity_k4": dict(k=4),
        "calibrated_uncertainty": dict(kappa=0.0),
        "no_skill_decay": dict(half_life_h=1e9),
        "all_three": dict(k=4, kappa=0.0, half_life_h=1e9),
    }
    jobs = [(dict(variant=v, tau=t), dict(tau=t, **kw), s) for v, kw in variants.items() for t in TAUS for s in SEEDS]
    run_jobs(jobs, "X1b")

def X2():   # fine threshold grid x staffing at fixed plant
    taus = np.round(np.arange(-1.0, 3.01, 0.1), 2)
    jobs = [(dict(k=k, tau=float(t)), dict(N=60, k=k, tau=float(t)), s) for k in (1, 2, 3, 4) for t in taus for s in SEEDS]
    run_jobs(jobs, "X2")

def X3():   # twin fidelity components and engagement specifications
    jobs = []
    for mode in ("logistic", "linear", "constant", "forced"):
        for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            jobs += [(dict(part="quality", eng_mode=mode, completeness=c), dict(eng_mode=mode, completeness=c, eps0=0.05), s) for s in SEEDS]
    for lag in (0, 10, 30, 60, 120, 300):
        jobs += [(dict(part="lag", lag_s=lag), dict(lag_s=float(lag)), s) for s in SEEDS]
    for es in (0.0, 0.15, 0.3, 0.5):
        for rc in (6.0, 12.0, 48.0):
            jobs += [(dict(part="model_error", eps_shift=es, recal_h=rc), dict(eps_shift=es, recal_h=rc), s) for s in SEEDS]
    run_jobs(jobs, "X3")

def X4():   # drills x escalation, 30 reps
    jobs = [(dict(tau=t, adapt=a, drill=d), dict(tau=t, adapt_tau=a, drill_rate_day=d), s)
            for t in (2.5, 4.0) for a in (False, True) for d in (0, 2, 4, 8) for s in range(1000, 1030)]
    run_jobs(jobs, "X4")

def X5():   # dispatch policy x arrival burstiness
    jobs = [(dict(policy=pol, arrivals=arr, tau=t), dict(policy=pol, arrivals=arr, tau=t), s)
            for pol in ("FCFS", "EDF", "SEV") for arr in ("poisson", "shock") for t in (0.0, 1.0) for s in SEEDS]
    run_jobs(jobs, "X5")

VARIANTS = {
    "baseline": {},
    "eng_constant": dict(eng_mode="constant"),
    "eng_linear": dict(eng_mode="linear"),
    "vig_none": dict(vig_mode="none"),
    "vig_monotone": dict(vig_mode="monotone"),
    "vig_fatigue": dict(vig_mode="fatigue"),
    "forget_power": dict(forget="power"),
    "hetero_operators": dict(sd_aH=0.5),
    "three_actions": dict(n_actions=3),
    "kappa_0": dict(kappa=0.0),
    "rho_sz_neg": dict(rho_sz=-0.3),
    "rho_sz_0": dict(rho_sz=0.0),
    "rho_sz_high": dict(rho_sz=0.6),
    "expire_abandon": dict(expire_mode="abandon"),
    "ai_latency_2min": dict(ai_lat_min=2.0),
    "shock_arrivals": dict(arrivals="shock"),
}
def X6():   # structural robustness of P1, P3, P4
    jobs = []
    seeds = range(1000, 1012)
    for v, kw in VARIANTS.items():
        for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            jobs += [(dict(variant=v, block="P1", tau=t), dict(tau=t, **kw), s) for s in seeds]
        for c in (0.2, 0.9):
            jobs += [(dict(variant=v, block="P3", completeness=c), dict(completeness=c, **kw), s) for s in seeds]
        for a in (False, True):
            for d in (0, 4):
                jobs += [(dict(variant=v, block="P4", adapt=a, drill=d), dict(tau=4.0, adapt_tau=a, drill_rate_day=d, **kw), s) for s in seeds]
    run_jobs(jobs, "X6")

def X7():   # global sensitivity: Latin hypercube
    rng = np.random.default_rng(42)
    ranges = dict(lam_cell=(0.15, 0.40), serv_mean_min=(4.0, 9.0), hscale=(0.6, 1.5), half_life_h=(72.0, 720.0),
                  kappa=(0.0, 1.0), sigma_u=(0.3, 1.0), completeness=(0.5, 1.0), z_shift=(0.5, 1.5),
                  a_A=(1.8, 3.2), b_H=(0.3, 0.9), rho_sz=(-0.3, 0.6), eps_shift=(0.0, 0.4), k=(1, 2.999))
    n = 160; d = len(ranges)
    U = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    rows = []
    for i in range(n):
        kv = {name: lo + U[i, j] * (hi - lo) for j, (name, (lo, hi)) in enumerate(ranges.items())}
        kv["k"] = int(kv["k"]); rows.append(kv)
    pd.DataFrame(rows).to_csv(OUT + "X7_design.csv", index_label="scenario")
    jobs = []
    seeds = range(1000, 1006)
    for i, kv in enumerate(rows):
        kw = {k: v for k, v in kv.items() if k != "hscale"}
        kw["sev_horizon_min"] = tuple(np.array(BASE.sev_horizon_min) * kv["hscale"])
        for t in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 8.0):
            jobs += [(dict(scenario=i, block="P1", tau=t), dict(tau=t, **kw), s) for s in seeds]
        for c in (0.2,):
            jobs += [(dict(scenario=i, block="P3", completeness=c), dict(**{**kw, "completeness": c}), s) for s in seeds]
        for a in (False, True):
            for dd in (0, 4):
                jobs += [(dict(scenario=i, block="P4", adapt=a, drill=dd), dict(tau=4.0, adapt_tau=a, drill_rate_day=dd, **kw), s) for s in seeds]
    run_jobs(jobs, "X7")

def X9():   # recovery trajectories
    configs = {"tau_-0.5": dict(tau=-0.5), "tau_1": dict(tau=1.0), "tau_4": dict(tau=4.0),
               "tau_4_adaptive": dict(tau=4.0, adapt_tau=True), "tau_4_adaptive_drills": dict(tau=4.0, adapt_tau=True, drill_rate_day=4)}
    jobs = [(dict(config=c), kw, s) for c, kw in configs.items() for s in SEEDS]
    run_jobs(jobs, "X9", keep_series=True)

def X10():  # staffing x vigilance specification (underload check for Pareto)
    jobs = [(dict(k=k, tau=t, vig=v), dict(N=60, k=k, tau=t, vig_mode=v), s)
            for v in ("invU", "none") for k in (1, 2, 3) for t in (-0.5, 0.0, 0.2, 0.5, 1.0, 1.2, 2.0) for s in SEEDS]
    run_jobs(jobs, "X10")

if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
