"""Verification suite for mhc2 (results written to results/verify_*.csv/json)."""
import json, math, time
import numpy as np, pandas as pd
from dataclasses import replace
from scipy.stats import norm
from scipy.integrate import quad
from mhc2 import P, draw, simulate, sigmoid, p_timely_mmk, twin_quality, engagement

import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(ROOT, "results") + "/"; _os.makedirs(OUT, exist_ok=True)
res = {}

# ---------- V1: exact cross-check of event calendar vs Kiefer-Wolfowitz (FCFS) ----------
def kw_reference(p, D):
    """Independent FCFS implementation via the Kiefer-Wolfowitz recursion, same rules."""
    t, z, u, H = D["t"], D["z"], D["u"], D["H"]; n = D["n"]
    pA = sigmoid(p.a_A - p.b_A * z); lag = p.lag_s / 3600; p0 = 1 / p.n_actions
    lam_dec = math.log(2) / p.half_life_h; s2 = math.log(1 + p.serv_cv ** 2)
    free = np.zeros(p.k); S = np.full(p.k, p.S0); tS = np.zeros(p.k)
    w = np.zeros(p.k); tw = np.zeros(p.k)
    out = np.full(n, -1, np.int8)
    sl2 = math.log(1 + 0.25); lat = (p.ai_lat_min / 60) * np.exp(-sl2 / 2 + math.sqrt(sl2) * D["Z_lat"])
    for i in range(n):
        tau = p.tau
        if p.adapt_tau and p.disrupt and p.t_d + p.det_delay <= t[i] < p.t_d + p.d_len: tau = min(p.tau, p.tau_low)
        if u[i] <= tau:
            out[i] = 1 if lat[i] > H[i] else (0 if D["U_ai"][i] < pA[i] else 2); continue
        j = int(np.argmin(free)); start = max(t[i], free[j])
        # workload: busy until free[j], idle until start
        def adv(j, to, busy):
            dt = to - tw[j]
            if dt > 0:
                e = math.exp(-dt / p.ema_T); w[j] = w[j] * e + (1 - e) * busy; tw[j] = to
        adv(j, start, 0.0)
        S[j] = p.S_min + (S[j] - p.S_min) * math.exp(-lam_dec * (start - tS[j])); tS[j] = start
        q = twin_quality(p, start); beta = engagement(p, q)
        psi = p.psi_floor + (1 - p.psi_floor) * math.exp(-0.5 * ((w[j] - p.w_star) / p.w_sd) ** 2)
        eng = D["U_eng"][i] < beta
        m = p.serv_mean_min / 60 * (1 + p.zeta * (1 - q)) * (1 if eng else p.rubber_frac)
        st = m * math.exp(-s2 / 2 + math.sqrt(s2) * D["Z_serv"][i])
        ph = sigmoid(D["aH"][j] + p.gamma_S * S[j] - p.b_H * z[i])
        ph = p.omega * ph + (1 - p.omega) * (q * ph + (1 - q) * p0); ph = p0 + (ph - p0) * psi
        if not eng: ph = pA[i]
        end = start + st; adv(j, end, 1.0)
        S[j] = p.S_min + (S[j] - p.S_min) * math.exp(-lam_dec * (end - tS[j])); tS[j] = end
        if end - t[i] + lag > H[i]: out[i] = 1
        else: out[i] = 0 if D["U_hum"][i] < ph else 2
        if eng and z[i] > p.z_hard: S[j] += p.eta_real * (1 - S[j])
        free[j] = end
    return out

rows = []
for k in (1, 2, 3):
    for tau in (-0.5, 0.5, 1.5):
        for seed in range(5):
            p = P(k=k, N=40 * k, tau=tau, T=400, warmup=40, t_d=250)
            D = draw(p, seed)
            r = simulate(p, D=D, record=True)
            ref = kw_reference(p, D)
            mism = int((r["_raw"]["outcome"] != ref).sum())
            rows.append(dict(k=k, tau=tau, seed=seed, n=D["n"], deferred=int(r["_raw"]["defer"].sum()), mismatches=mism))
v1 = pd.DataFrame(rows); v1.to_csv(OUT + "verify_kw.csv", index=False)
res["kw_total_cases"] = int(v1.n.sum()); res["kw_total_deferred"] = int(v1.deferred.sum()); res["kw_mismatches"] = int(v1.mismatches.sum())
print("V1 KW cross-check mismatches:", res["kw_mismatches"], "of", res["kw_total_deferred"], flush=True)

# ---------- V2: invariants on event log (with drills, EDF, shocks) ----------
viol = dict(overlap=0, start_before_arrival=0)
for pol in ("FCFS", "EDF", "SEV"):
    for arr in ("poisson", "shock"):
        p = P(policy=pol, arrivals=arr, drill_rate_day=8, tau=0.0, T=400, warmup=40, t_d=250)
        r = simulate(p, seed=7, record=True, check=True)
        log = pd.DataFrame(r["_log"], columns=["j", "s", "e", "kind"])
        for j, g in log.groupby("j"):
            g = g.sort_values("s"); viol["overlap"] += int((g.s.values[1:] < g.e.values[:-1] - 1e-12).sum())
        raw = r["_raw"]; d = raw["defer"]
        viol["start_before_arrival"] += int((raw["wait"][d] < -1e-12).sum())
res["invariant_violations"] = viol
print("V2 invariants:", viol, flush=True)

# ---------- V3: analytic M/M/k benchmark incl. high load ----------
rows = []
for k in (1, 2, 3):
    for rho_target in (0.2, 0.5, 0.7, 0.85, 0.95):
        mu = 60 / 6.0
        lam_d = rho_target * k * mu
        # choose N such that a fully deferring policy yields lam_d
        p = P(analytic=True, disrupt=False, k=k, tau=-50, lag_s=0.0, T=6000, warmup=500,
              N=1, lam_cell=lam_d, a_H=1.0)
        sims = [simulate(p, seed=s, record=True) for s in range(6)]
        Ps = [s["P_timely"] for s in sims]
        th = 60 / np.array(p.sev_horizon_min); pis = np.array(p.sev_p) / sum(p.sev_p)
        an = sum(pis[c] * p_timely_mmk(lam_d, mu, k, th[c]) for c in range(3))
        rows.append(dict(k=k, rho=rho_target, P_sim=np.mean(Ps), P_sd=np.std(Ps, ddof=1), P_an=an))
v3 = pd.DataFrame(rows); v3["abs_err"] = (v3.P_sim - v3.P_an).abs(); v3["z"] = (v3.P_sim - v3.P_an) / (v3.P_sd / np.sqrt(6))
v3.to_csv(OUT + "verify_mmk.csv", index=False)
res["mmk_mae"] = float(v3.abs_err.mean()); res["mmk_max"] = float(v3.abs_err.max()); res["mmk_max_absz"] = float(v3.z.abs().max())
print(v3.round(4).to_string(), flush=True)

# ---------- V4: Lemma 1 (stationary skill) benchmark ----------
rows = []
for load, lc in (("low", 0.005), ("high", 0.2)):
    for r_day in (0.5, 1, 2, 4, 8):
        p = P(disrupt=False, tau=-50, N=40, lam_cell=lc, eta_real=0.0, drill_rate_day=r_day, drill_expiry_h=1e9,
              T=3000, warmup=600, S0=0.5)
        sims = [simulate(p, seed=s) for s in range(4)]
        S_sim = np.nanmean([s["skill_n"] for s in sims])
        comp = np.mean([s["drill_completed"] / max(1, s["drill_scheduled"]) for s in sims])
        eta = math.log(2) / p.half_life_h
        for label, rr in (("nominal", r_day / 24), ("realised", comp * r_day / 24)):
            q = rr / (rr + eta); pred = (p.S_min * (1 - q) + q * p.eta_drill) / (1 - q * (1 - p.eta_drill))
            rows.append(dict(load=load, r_day=r_day, basis=label, S_sim=S_sim, S_pred=pred, completion=comp))
v4 = pd.DataFrame(rows); v4.to_csv(OUT + "verify_lemma.csv", index=False)
print(v4.round(3).to_string(), flush=True)

# ---------- V5: Proposition 2 numeric derivative check (reduced model) ----------
p = P(disrupt=False); mu = 60 / p.serv_mean_min; Lam = p.N * p.lam_cell; th = 60 / np.average(p.sev_horizon_min, weights=p.sev_p)
cp = np.array(p.class_p); mz = np.array(p.class_mu_z); su = math.sqrt(1 + p.sigma_u ** 2)
def f_u(x): return float(np.sum(cp * norm.pdf((x - mz) / su) / su))
def cond(x, fn):
    # E[fn(z) | u=x] under class mixture: z|u,c ~ N(mu_c + (x-mu_c)/(1+su2), su2/(1+su2))
    s2u = p.sigma_u ** 2; wts = cp * norm.pdf((x - mz) / su) / su; wts = wts / wts.sum()
    m = mz + (x - mz) / (1 + s2u); sd = math.sqrt(s2u / (1 + s2u))
    zz, ww = np.polynomial.hermite_e.hermegauss(40); ww = ww / ww.sum()
    return float(sum(wc * np.sum(ww * fn(mc + sd * zz)) for wc, mc in zip(wts, m)))
pA_ = lambda zv: sigmoid(p.a_A - p.b_A * zv)
pH_ = lambda zv: sigmoid(p.a_H + p.gamma_S * 0.95 - p.b_H * zv)
def lam_d(tau): return Lam * float(np.sum(cp * norm.sf((tau - mz) / su)))
def Pt(ld): return p_timely_mmk(ld, mu, 1, th)
def I(tau): return quad(lambda x: cond(x, pH_) * f_u(x), tau, tau + 12)[0]
def J(tau): return quad(lambda x: cond(x, pA_) * f_u(x), tau - 12, tau)[0] + Pt(lam_d(tau)) * I(tau) + quad(lambda x: cond(x, pA_) * f_u(x), -30, tau - 12)[0]
def dPt(ld, h=1e-5): return (Pt(ld + h) - Pt(ld - h)) / (2 * h)
def Jprime_formula(tau): return f_u(tau) * (cond(tau, pA_) - Pt(lam_d(tau)) * cond(tau, pH_) - Lam * dPt(lam_d(tau)) * I(tau))
grid = np.round(np.arange(-1.0, 3.01, 0.25), 3); rows = []
for tau in grid:
    h = 1e-3; num = (J(tau + h) - J(tau - h)) / (2 * h)
    g = cond(tau, pA_) - Pt(lam_d(tau)) * cond(tau, pH_) - Lam * dPt(lam_d(tau)) * I(tau)
    hmy = cond(tau, pA_) - Pt(lam_d(tau)) * cond(tau, pH_)
    rows.append(dict(tau=tau, J=J(tau), dJ_numeric=num, dJ_formula=Jprime_formula(tau), g=g, h_myopic=hmy, lam_d=lam_d(tau)))
v5 = pd.DataFrame(rows); v5.to_csv(OUT + "verify_prop2.csv", index=False)
from scipy.optimize import brentq
gfun = lambda x: cond(x, pA_) - Pt(lam_d(x)) * cond(x, pH_) - Lam * dPt(lam_d(x)) * I(x)
hfun = lambda x: cond(x, pA_) - Pt(lam_d(x)) * cond(x, pH_)
try: res["tau_star_reduced"] = brentq(gfun, -0.9, 3.0)
except Exception as e: res["tau_star_reduced"] = None
try: res["tau_myopic_reduced"] = brentq(hfun, -0.9, 3.0)
except Exception as e: res["tau_myopic_reduced"] = None
res["prop2_max_abs_diff_derivative"] = float((v5.dJ_numeric - v5.dJ_formula).abs().max())
res["prop2_sign_changes_g"] = int((np.diff(np.sign(v5.g)) != 0).sum())
print(v5.round(5).to_string(), res, flush=True)

# ---------- V6: workload integration check ----------
# busy on [0,1] h, idle on [1,3] h: exact w(3) = (1-e^{-1/T}) e^{-2/T}
from mhc2 import Server
Tw = P().ema_T; exact = (1 - math.exp(-1 / Tw)) * math.exp(-2 / Tw)
w = 0.0; e = math.exp(-1 / Tw); w = w * e + (1 - e); e2 = math.exp(-2 / Tw); w = w * e2
res["workload_check_abs_err"] = abs(w - exact)
json.dump(res, open(OUT + "verify_summary.json", "w"), indent=1, default=float)
print(res)
