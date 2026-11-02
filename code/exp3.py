"""Additional experiments for manuscript revision v0.7 (uses mhc3 = mhc2 + extensions).
Usage: python3 exp3.py CHECK X11 X12 X13 X14"""
import sys, time, math
import numpy as np, pandas as pd
from dataclasses import replace
from multiprocessing import Pool
from mhc3 import P, simulate

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_ext") + "/"; os.makedirs(OUT, exist_ok=True)
BASE = P()
SEEDS = range(1000, 1020)
TAUS = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]


def _job(args):
    tag, kw, seed = args
    r = simulate(replace(BASE, **kw), seed=seed)
    row = {k: v for k, v in r.items() if not k.startswith("_")}
    row.update(tag); row["seed"] = seed
    return row


def run_jobs(jobs, name):
    t0 = time.time()
    with Pool(2) as pool:
        rows = pool.map(_job, jobs, chunksize=8)
    df = pd.DataFrame(rows); df.to_csv(OUT + f"{name}.csv", index=False)
    print(name, len(rows), "runs", round(time.time() - t0, 1), "s", flush=True)
    return df


def CHECK():   # defaults of mhc3 must reproduce mhc2 exactly (non-abandon runs)
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import mhc2
    bad = 0
    for kw in (dict(), dict(tau=-1.0, k=2), dict(policy="EDF", arrivals="shock", drill_rate_day=4, adapt_tau=True)):
        for s in (1000, 1001):
            a = mhc2.simulate(replace(mhc2.P(), **kw), seed=s); b = simulate(replace(BASE, **kw), seed=s)
            for k, v in a.items():
                if k.startswith("_"): continue
                if not (v == b[k] or (isinstance(v, float) and math.isnan(v) and math.isnan(b[k]))): bad += 1
    print("CHECK mismatches:", bad)


def X11():     # X1 grid again: ECP for alternative p_crit, AR with halt-execution time
    jobs = [(dict(N=N, k=k, tau=t), dict(N=N, k=k, tau=t), s) for N in (40, 80) for k in (1, 2) for t in TAUS for s in SEEDS]
    run_jobs(jobs, "X11")


def X12():     # corrected abandon mode; difficulty-dependent decision time
    variants = {"baseline": {}, "expire_abandon_fixed": dict(expire_mode="abandon"),
                "serv_difficulty": dict(serv_z=0.3, serv_zref=1.55)}
    jobs = []; seeds = range(1000, 1012)
    for v, kw in variants.items():
        for t in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
            jobs += [(dict(variant=v, block="P1", tau=t), dict(tau=t, **kw), s) for s in seeds]
        for c in (0.2, 0.9):
            jobs += [(dict(variant=v, block="P3", completeness=c), dict(completeness=c, **kw), s) for s in seeds]
        for a in (False, True):
            for d in (0, 4):
                jobs += [(dict(variant=v, block="P4", adapt=a, drill=d), dict(tau=4.0, adapt_tau=a, drill_rate_day=d, **kw), s) for s in seeds]
    run_jobs(jobs, "X12")


def X13():     # AI-initiated safe stop: does the loss-optimal threshold still favour heavy escalation?
    taus = list(np.round(np.arange(-1.0, 3.01, 0.25), 2)) + [8.0]
    stops = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, float("inf")]
    jobs = [(dict(k=k, tau=float(t), tau_stop=st), dict(N=60, k=k, tau=float(t), tau_stop=st), s)
            for k in (1, 2) for t in taus for st in stops if st == float("inf") or st < t for s in SEEDS]
    run_jobs(jobs, "X13")


def X14():     # global sensitivity including behavioural parameters
    rng = np.random.default_rng(43)
    ranges = dict(lam_cell=(0.15, 0.40), serv_mean_min=(4.0, 9.0), hscale=(0.6, 1.5), half_life_h=(72.0, 720.0),
                  kappa=(0.0, 1.0), sigma_u=(0.3, 1.0), completeness=(0.5, 1.0), z_shift=(0.5, 1.5),
                  a_A=(1.8, 3.2), b_H=(0.3, 0.9), rho_sz=(-0.3, 0.6), eps_shift=(0.0, 0.4), k=(1, 2.999),
                  kF=(5.0, 20.0), F0=(0.4, 0.7), omega=(0.2, 0.6), zeta=(0.4, 1.2), psi_floor=(0.4, 0.9),
                  w_star=(0.3, 0.8), eta_real=(0.01, 0.06), eta_drill=(0.03, 0.12))
    n = 160; d = len(ranges)
    U = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    rows = []
    for i in range(n):
        kv = {nm: lo + U[i, j] * (hi - lo) for j, (nm, (lo, hi)) in enumerate(ranges.items())}
        kv["k"] = int(kv["k"]); rows.append(kv)
    pd.DataFrame(rows).to_csv(OUT + "X14_design.csv", index_label="scenario")
    jobs = []; seeds = range(1000, 1006)
    for i, kv in enumerate(rows):
        kw = {k: v for k, v in kv.items() if k != "hscale"}
        kw["sev_horizon_min"] = tuple(np.array(BASE.sev_horizon_min) * kv["hscale"])
        for t in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 8.0):
            jobs += [(dict(scenario=i, block="P1", tau=t), dict(tau=t, **kw), s) for s in seeds]
        jobs += [(dict(scenario=i, block="P3", completeness=0.2), dict(**{**kw, "completeness": 0.2}), s) for s in seeds]
        for a in (False, True):
            for dd in (0, 4):
                jobs += [(dict(scenario=i, block="P4", adapt=a, drill=dd), dict(tau=4.0, adapt_tau=a, drill_rate_day=dd, **kw), s) for s in seeds]
    run_jobs(jobs, "X14")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        globals()[nm]()
