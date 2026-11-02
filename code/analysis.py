import json, math
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = _os.path.join(ROOT, "results") + "/"; FIG = _os.path.join(ROOT, "figures") + "/"
import os; os.makedirs(FIG, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.8,
    "savefig.dpi": 600, "savefig.bbox": "tight", "axes.titlesize": 9, "axes.titleweight": "bold"})
S = {}   # numbers for the manuscript

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))

def paired(df, keycols, a, b, metric):
    """mean and 95% t-CI of paired difference b - a (dicts of filters) by seed."""
    da = df.loc[np.logical_and.reduce([df[k] == v for k, v in a.items()])].set_index("seed")[metric]
    db = df.loc[np.logical_and.reduce([df[k] == v for k, v in b.items()])].set_index("seed")[metric]
    d = (db - da).dropna()
    return float(d.mean()), float(ci(d)), d

# ---------------- verification figure ----------------
v3 = pd.read_csv(OUT + "verify_mmk.csv"); v4 = pd.read_csv(OUT + "verify_lemma.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axs[0]
for k, c, m in zip((1, 2, 3), (C1, C2, C3), ("o", "s", "D")):
    s = v3[v3.k == k]
    ax.errorbar(s.P_an, s.P_sim, yerr=stats.t.ppf(.975, 5) * s.P_sd / np.sqrt(6), fmt=m, ms=5, color=c, mec="white", mew=.8, elinewidth=.9, label=f"k = {k}")
ax.plot([0.25, 0.9], [0.25, 0.9], color=INK2, lw=.9, ls="--", zorder=0)
ax.set_xlabel("Analytical $P_t$ (Proposition 1)"); ax.set_ylabel("Simulated $P_t$")
ax.set_title("(a) Queue kernel, ρ = 0.20–0.95", loc="left"); ax.legend(loc="upper left")
ax = axs[1]
nom = v4[(v4.load == "low") & (v4.basis == "nominal")]
ax.plot(nom.r_day, nom.S_pred, "-", color=INK2, lw=1.2, label="Lemma 1, nominal rate")
real = v4[(v4.load == "high") & (v4.basis == "realised")]
ax.plot(real.r_day, real.S_pred, ":", color=C2, lw=1.6, label="Lemma 1, realised rate (high load)")
for load, c in (("low", C1), ("high", C2)):
    s = v4[(v4.load == load) & (v4.basis == "nominal")]
    ax.plot(s.r_day, s.S_sim, "o", color=c, ms=5, mec="white", label=f"Simulated, {load} load")
ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xlabel("Scheduled drills per supervisor per day"); ax.set_ylabel("Mean skill (time average)")
ax.set_title("(b) Stationary skill benchmark", loc="left"); ax.legend(loc="upper left", fontsize=6.8)
fig.tight_layout(); fig.savefig(FIG + "fig4_verification.png"); plt.close(fig)
vs = json.load(open(OUT + "verify_summary.json")); S["verify"] = vs
S["lemma_low_maxerr"] = float((v4[(v4.load == "low") & (v4.basis == "nominal")].eval("S_sim-S_pred")).abs().max())
S["lemma_high_nominal_maxerr"] = float((v4[(v4.load == "high") & (v4.basis == "nominal")].eval("S_sim-S_pred")).abs().max())
S["lemma_high_realised_maxerr"] = float((v4[(v4.load == "high") & (v4.basis == "realised")].eval("S_sim-S_pred")).abs().max())
S["lemma_high_completion"] = [float(v4[v4.load == "high"].completion.min()), float(v4[v4.load == "high"].completion.max())]
v5 = pd.read_csv(OUT + "verify_prop2.csv"); S["prop2_tab"] = v5[["tau", "dJ_numeric", "dJ_formula", "g", "h_myopic"]].round(5).to_dict("records")

# ---------------- X1 darkness curve (factorial) ----------------
x1 = pd.read_csv(OUT + "X1.csv")
g = x1.groupby(["N", "k", "tau"]).agg(A=("autonomy", "mean"), ERRn=("ERR_n", "mean"), ERRd=("ERR_d", "mean"), ERRd_ci=("ERR_d", ci),
                                       ECPd=("ECP_d", "mean"), ECPn=("ECP_n", "mean"), ARd=("AR_d", "mean"), skill=("skill_d", "mean"),
                                       PSd=("PS_d", "mean"), EAd=("EA_d", "mean"), utild=("util_d", "mean"), utiln=("util_n", "mean")).reset_index()
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True)
S["X1"] = {}
for ax, (N, k) in zip(axs.flat, [(40, 1), (80, 1), (40, 2), (80, 2)]):
    s = g[(g.N == N) & (g.k == k) & (g.tau < 8)].sort_values("A")
    ax.plot(s.A, s.ERRn, color=C1, marker="o", ms=3, label="ERR, nominal")
    ax.plot(s.A, s.ERRd, color=C2, marker="o", ms=3, label="ERR, during shift")
    ax.fill_between(s.A, s.ERRd - s.ERRd_ci, s.ERRd + s.ERRd_ci, color=C2, alpha=.15, lw=0)
    ax.plot(s.A, s.ECPd, color=C3, marker="s", ms=3, ls="--", label="ECP (command reach), shift")
    ax.set_title(f"N = {N} cells, k = {k} (N/k = {N // k})", loc="left"); ax.set_ylim(0, 1)
    i = s.ERRd.idxmax(); full = g[(g.N == N) & (g.k == k) & (g.tau == 8)].iloc[0]
    S["X1"][f"N{N}_k{k}"] = dict(A_opt_shift=float(s.loc[i, "A"]), ERRd_max=float(s.loc[i, "ERRd"]), tau_opt_shift=float(s.loc[i, "tau"]),
        ERRd_full=float(full.ERRd), ERRd_lowA=float(s.ERRd.iloc[0]), A_low=float(s.A.iloc[0]),
        A_opt_nom=float(s.loc[s.ERRn.idxmax(), "A"]), ERRn_max=float(s.ERRn.max()), ERRn_full=float(full.ERRn),
        ECPd_at_opt=float(s.loc[i, "ECPd"]), ECPd_full=float(full.ECPd), ECPd_max=float(s.ECPd.max()),
        skill_at_opt=float(s.loc[i, "skill"]), skill_A999=float(s[s.tau == 4.0].skill.iloc[0]), utild_at_lowA=float(s.utild.iloc[0]))
    ax.axvline(s.loc[i, "A"], color=C2, lw=.7, ls=":")
for ax in axs[1]: ax.set_xlabel("Exception-handling autonomy A (share not escalated)")
for ax in axs[:, 0]: ax.set_ylabel("Probability")
axs[0, 0].legend(loc="lower left", fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "fig5_reliability_command.png"); plt.close(fig)
# paired test of interior optimum at N40 k1
sub = x1[(x1.N == 40) & (x1.k == 1)]
topt = S["X1"]["N40_k1"]["tau_opt_shift"]
m1, c1_, _ = paired(sub, None, dict(tau=8.0), dict(tau=topt), "ERR_d"); m2, c2_, _ = paired(sub, None, dict(tau=-1.5), dict(tau=topt), "ERR_d")
S["X1"]["paired_opt_vs_full"] = [m1, c1_]; S["X1"]["paired_opt_vs_low"] = [m2, c2_]

x1b = pd.read_csv(OUT + "X1b.csv")
gb = x1b.groupby(["variant", "tau"]).agg(A=("autonomy", "mean"), ERRd=("ERR_d", "mean"), ERRn=("ERR_n", "mean")).reset_index()
S["X1b"] = {}
for v, s in gb.groupby("variant"):
    s = s.sort_values("tau"); i = s.ERRd.idxmax()
    S["X1b"][v] = dict(tau_opt=float(s.loc[i, "tau"]), A_opt=float(s.loc[i, "A"]), ERRd_max=float(s.ERRd.max()),
                       drop_to_full=float(s.ERRd.max() - s[s.tau == 8].ERRd.iloc[0]), drop_to_low=float(s.ERRd.max() - s[s.tau == -1.5].ERRd.iloc[0]))

# ---------------- X2 fine grid, argmax uncertainty ----------------
x2 = pd.read_csv(OUT + "X2.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
rng = np.random.default_rng(7); S["X2"] = {}
cols = {1: C1, 2: C2, 3: C3, 4: C4}
for k in (1, 2, 3, 4):
    s = x2[x2.k == k]; M = s.pivot(index="seed", columns="tau", values="ERR"); taus = M.columns.values
    mean = M.mean(); axs[0].plot(taus, mean, color=cols[k], label=f"k = {k}")
    boot = []
    for b in range(2000):
        idx = rng.integers(0, len(M), len(M)); boot.append(taus[np.argmax(M.values[idx].mean(0))])
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [5, 95]); am = taus[np.argmax(mean.values)]
    # split-sample winner's bias
    A_, B_ = M.iloc[:10], M.iloc[10:]
    sel = taus[np.argmax(A_.mean().values)]
    in_s = float(A_.mean().max()); out_s = float(B_[sel].mean()); best_B = float(B_.mean().max())
    # curvature from quadratic fit within +-0.5
    win = (taus >= am - 0.5) & (taus <= am + 0.5); cf = np.polyfit(taus[win], mean.values[win], 2)
    # paired difference between argmax and neighbours +-0.3
    d_lo = (M[am] - M[taus[max(0, np.argmax(mean.values) - 3)]]); d_hi = (M[am] - M[taus[min(len(taus) - 1, np.argmax(mean.values) + 3)]])
    S["X2"][f"k{k}"] = dict(argmax=float(am), ci90=[float(lo), float(hi)], ERRmax=float(mean.max()), curvature=float(2 * cf[0]),
        split_in=in_s, split_out=out_s, split_best_other=best_B, sel_tau=float(sel),
        d_vs_minus03=[float(d_lo.mean()), float(ci(d_lo))], d_vs_plus03=[float(d_hi.mean()), float(ci(d_hi))],
        ERR_at_m1=float(mean[-1.0]), mode_freq=float(np.mean(boot == am)))
    axs[1].errorbar(am, k, xerr=[[am - lo], [hi - am]], fmt="o", color=cols[k], ms=5, capsize=3, elinewidth=1.2)
axs[0].set_xlabel("Deferral threshold τ"); axs[0].set_ylabel("ERR over 30 days (incl. shift)"); axs[0].set_ylim(0.70, 0.9)
axs[0].set_title("(a) Mean ERR, N = 60, 20 paired replications", loc="left"); axs[0].legend(loc="lower right", fontsize=7)
axs[1].set_yticks([1, 2, 3, 4]); axs[1].set_yticklabels([f"k = {k}" for k in (1, 2, 3, 4)]); axs[1].set_xlabel("Optimal τ (point and 90% bootstrap interval)")
axs[1].set_title("(b) Uncertainty of the argmax", loc="left"); axs[1].set_ylim(0.5, 4.5)
fig.tight_layout(); fig.savefig(FIG + "fig6_threshold.png"); plt.close(fig)

# ---------------- X3 fidelity ----------------
x3 = pd.read_csv(OUT + "X3.csv"); S["X3"] = {}
q = x3[x3.part == "quality"].copy(); q["q"] = q.completeness * 0.95
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
lab = {"logistic": "Logistic β(q) (baseline)", "linear": "Linear β = q", "constant": "Constant β = 0.9", "forced": "Cognitive forcing β = 1"}
for mode, c in zip(("logistic", "linear", "constant", "forced"), (C1, C3, C4, C2)):
    s = q[q.eng_mode == mode].groupby("q").agg(RVO=("RVO_d", "mean"), RVOci=("RVO_d", ci), ECP=("ECP_d", "mean"), ISR=("ISR_d", "mean"), OL=("OL_min", "mean")).reset_index()
    axs[0].plot(s.q, s.RVO, color=c, marker="o", ms=3, label=lab[mode]); axs[0].fill_between(s.q, s.RVO - s.RVOci, s.RVO + s.RVOci, color=c, alpha=.12, lw=0)
    axs[1].plot(s.q, s.ECP, color=c, marker="o", ms=3)
    sign = np.sign(s.RVO.values); crossings = int((np.diff(sign[sign != 0]) != 0).sum())
    mono = bool(np.all(np.diff(s.RVO.values) > -0.003))
    S["X3"][mode] = dict(RVO_q095=float(s.RVO.iloc[-1]), RVO_q019=float(s.RVO.iloc[1]), RVO_q0095=float(s.RVO.iloc[0]), ECP_q095=float(s.ECP.iloc[-1]),
                          ECP_q019=float(s.ECP.iloc[1]), ISR_q019=float(s.ISR.iloc[1]), OL_q019=float(s.OL.iloc[1]), OL_q095=float(s.OL.iloc[-1]),
                          crossings=crossings, monotone=mono, RVO_min=float(s.RVO.min()))
axs[0].axhline(0, color=INK2, lw=.8)
axs[0].set_title("(a) RVO during shift", loc="left"); axs[0].set_ylabel("RVO during shift")
axs[1].set_title("(b) ECP during shift", loc="left"); axs[1].set_ylabel("ECP")
for ax in axs[:2]: ax.set_xlabel("Twin information quality q")
axs[0].legend(loc="upper left", fontsize=6.3)
l = x3[x3.part == "lag"].groupby("lag_s").agg(ECP=("ECP", "mean"), m2=("miss_s2", "mean"), m2ci=("miss_s2", ci), m0=("miss_s0", "mean")).reset_index()
axs[2].plot(l.lag_s, l.m2, color=C2, marker="o", ms=3, label="High severity (15 min)")
axs[2].plot(l.lag_s, l.m0, color=C1, marker="o", ms=3, label="Low severity (60 min)")
axs[2].set_xlabel("Synchronisation lag (s)"); axs[2].set_ylabel("Miss rate of escalations"); axs[2].set_title("(c) Synchronisation lag", loc="left")
axs[2].legend(loc="upper left", fontsize=6.5)
fig.tight_layout(); fig.savefig(FIG + "fig7_twin.png"); plt.close(fig)
S["X3"]["lag"] = l.round(4).to_dict("records")
me = x3[x3.part == "model_error"].groupby(["eps_shift", "recal_h"]).agg(ERRd=("ERR_d", "mean"), ECPd=("ECP_d", "mean"), qd=("q_d", "mean")).reset_index()
S["X3"]["model_error"] = me.round(4).to_dict("records")

# ---------------- X4 drills ----------------
x4 = pd.read_csv(OUT + "X4.csv"); S["X4"] = {}
rows = []
for tau in (2.5, 4.0):
    for ad in (False, True):
        base_ = x4[(x4.tau == tau) & (x4.adapt == ad) & (x4.drill == 0)].set_index("seed")
        for d in (0, 2, 4, 8):
            s = x4[(x4.tau == tau) & (x4.adapt == ad) & (x4.drill == d)].set_index("seed")
            dd = (s.ERR_d - base_.ERR_d); de = (s.ERR_e - base_.ERR_e)
            tost = None
            if d > 0:
                m, sd, n = dd.mean(), dd.std(ddof=1), len(dd); se = sd / math.sqrt(n); marg = 0.01
                p_lo = 1 - stats.t.cdf((m + marg) / se, n - 1); p_hi = stats.t.cdf((m - marg) / se, n - 1); tost = float(max(p_lo, p_hi))
            rows.append(dict(tau=tau, adapt=ad, drill=d, skill_d=s.skill_d.mean(), ERR_d=s.ERR_d.mean(), dERR_d=dd.mean(), dERR_d_ci=ci(dd),
                             dERR_e=de.mean(), dERR_e_ci=ci(de), ECP_d=s.ECP_d.mean(), sched=s.drill_scheduled.mean(), done=s.drill_completed.mean(),
                             interrupted=s.drill_interrupted.mean(), expired=s.drill_expired.mean(), tost_p=tost, OL=s.OL_min.mean()))
t5 = pd.DataFrame(rows); t5.to_csv(OUT + "T_drills.csv", index=False); S["X4"] = t5.round(4).to_dict("records")
# interaction contrast at tau=4, 4 drills
a = x4[(x4.tau == 4.0)]
def gs(ad, d): return a[(a.adapt == ad) & (a.drill == d)].set_index("seed").ERR_d
inter = (gs(True, 4) - gs(True, 0)) - (gs(False, 4) - gs(False, 0))
S["X4_interaction_tau4_d4"] = [float(inter.mean()), float(ci(inter)), float(stats.ttest_1samp(inter, 0).pvalue)]
ad_eff = gs(True, 0) - gs(False, 0); S["X4_adapt_effect_tau4"] = [float(ad_eff.mean()), float(ci(ad_eff))]
a25 = x4[(x4.tau == 2.5)]
ad25 = a25[(a25.adapt == True) & (a25.drill == 0)].set_index("seed").ERR_d - a25[(a25.adapt == False) & (a25.drill == 0)].set_index("seed").ERR_d
S["X4_adapt_effect_tau25"] = [float(ad25.mean()), float(ci(ad25))]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, tau in zip(axs, (2.5, 4.0)):
    for o, (ad, c, lb) in enumerate(((False, C1, "Static threshold"), (True, C2, "Shift-aware escalation"))):
        s = t5[(t5.tau == tau) & (t5.adapt == ad)]
        gg = x4[(x4.tau == tau) & (x4.adapt == ad)].groupby("drill").ERR_d
        x = np.arange(4) + (o - .5) * 0.39
        ax.bar(x, gg.mean().values, width=0.36, color=c, edgecolor="white", label=lb)
        ax.errorbar(x, gg.mean().values, yerr=gg.apply(ci).values, fmt="none", ecolor=INK, elinewidth=.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0", "2", "4", "8"]); ax.set_ylim(0.55, 0.86)
    ax.set_xlabel("Scheduled drills per supervisor per day"); ax.set_title(f"({'a' if tau == 2.5 else 'b'}) τ = {tau:g}", loc="left")
axs[0].set_ylabel("ERR during shift"); axs[0].legend(loc="upper left", fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(FIG + "fig8_drills.png"); plt.close(fig)

# ---------------- X5 scheduling x arrivals ----------------
x5 = pd.read_csv(OUT + "X5.csv")
g5 = x5.groupby(["policy", "arrivals", "tau"]).agg(m0=("miss_s0", "mean"), m1=("miss_s1", "mean"), m2=("miss_s2", "mean"), m2ci=("miss_s2", ci),
     p95=("Wq_p95_min", "mean"), ERRd=("ERR_d", "mean"), ERR=("ERR", "mean"), ECP=("ECP", "mean"), AR=("AR", "mean"), loss=("loss_h", "mean"), PS=("PS", "mean"), EA=("EA", "mean")).reset_index()
g5.to_csv(OUT + "T_sched.csv", index=False); S["X5"] = g5.round(4).to_dict("records")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
for ax, tau in zip(axs, (0.0, 1.0)):
    xs = np.arange(3); wdt = 0.2
    for o, (arr, pol) in enumerate([(a_, p_) for a_ in ("poisson", "shock") for p_ in ("FCFS", "EDF", "SEV")][:0]): pass
    for j, pol in enumerate(("FCFS", "EDF", "SEV")):
        for a_, hatch in (("poisson", None), ("shock", "///")):
            s = g5[(g5.policy == pol) & (g5.arrivals == a_) & (g5.tau == tau)].iloc[0]
            off = (j - 1) * 0.27 + (0 if a_ == "poisson" else 0.13) - 0.065
            ax.bar(xs + off, [s.m0, s.m1, s.m2], width=0.12, color=[C1, C2, C3][j], hatch=hatch, edgecolor="white", linewidth=.6)
    ax.set_xticks(xs); ax.set_xticklabels(["Low (60 min)", "Medium (30 min)", "High (15 min)"])
    ax.set_xlabel("Severity class (mean horizon)"); ax.set_title(f"({'a' if tau == 0 else 'b'}) τ = {tau:g}", loc="left")
axs[0].set_ylabel("Deadline-miss rate of escalations")
hand = [Patch(color=C1, label="FCFS"), Patch(color=C2, label="EDF"), Patch(color=C3, label="Severity priority"),
        Patch(facecolor="white", edgecolor=INK2, hatch="///", label="Common-cause shocks")]
axs[0].legend(handles=hand, loc="upper left", fontsize=6.8)
fig.tight_layout(); fig.savefig(FIG + "fig9_dispatch.png"); plt.close(fig)

# ---------------- X6 structural robustness ----------------
x6 = pd.read_csv(OUT + "X6.csv"); rows = []
for v, s in x6.groupby("variant", sort=False):
    p1 = s[s.block == "P1"]; M = p1.pivot(index="seed", columns="tau", values="ERR_d"); mu = M.mean()
    t_opt = mu.idxmax(); d_full = M[t_opt] - M[8.0]; d_low = M[t_opt] - M[-1.0]
    interior = (t_opt not in (-1.0, 8.0))
    sig = lambda d: d.mean() > 0 and stats.ttest_1samp(d, 0).pvalue < 0.05
    P1 = bool(interior and sig(d_full) and sig(d_low))
    p3 = s[s.block == "P3"]; lo = p3[p3.completeness == 0.2].set_index("seed"); hi = p3[p3.completeness == 0.9].set_index("seed")
    rv_lo, rv_hi = lo.RVO_d.mean(), hi.RVO_d.mean()
    P3 = bool(rv_lo < 0.5 * rv_hi and lo.ISR_d.mean() < 0.3)
    p4 = s[s.block == "P4"]
    def gg(ad, d): return p4[(p4.adapt == ad) & (p4.drill == d)].set_index("seed").ERR_d
    inter = (gg(True, 4) - gg(True, 0)) - (gg(False, 4) - gg(False, 0)); st_eff = gg(False, 4) - gg(False, 0)
    P4 = bool(inter.mean() > 0 and stats.ttest_1samp(inter, 0).pvalue < 0.10)
    rows.append(dict(variant=v, tau_opt=t_opt, A_opt=float(p1[p1.tau == t_opt].autonomy.mean()), drop_full=float(d_full.mean()), drop_low=float(d_low.mean()), P1=P1,
                     RVO_lo=float(rv_lo), RVO_hi=float(rv_hi), ISR_lo=float(lo.ISR_d.mean()), P3=P3,
                     inter=float(inter.mean()), inter_ci=float(ci(inter)), static_eff=float(st_eff.mean()), P4=P4))
t6 = pd.DataFrame(rows); t6.to_csv(OUT + "T_robust.csv", index=False); S["X6"] = t6.round(4).to_dict("records")

# ---------------- X7 global sensitivity ----------------
des = pd.read_csv(OUT + "X7_design.csv").set_index("scenario"); x7 = pd.read_csv(OUT + "X7.csv")
rows = []
for sc, s in x7.groupby("scenario"):
    p1 = s[s.block == "P1"]; M = p1.pivot(index="seed", columns="tau", values="ERR_d"); mu = M.mean(); t_opt = mu.idxmax()
    Mall = p1.pivot(index="seed", columns="tau", values="ERR").mean()
    d_full = M[t_opt] - M[8.0]; d_low = M[t_opt] - M[-0.5]
    interior = t_opt not in (-0.5, 8.0)
    P1 = bool(interior and d_full.mean() > 2 * d_full.std(ddof=1) / math.sqrt(len(d_full)) and d_low.mean() > 2 * d_low.std(ddof=1) / math.sqrt(len(d_low)))
    base1 = p1[p1.tau == 1.0]
    p3 = s[s.block == "P3"]
    rvo_lo = p3.RVO_d.mean(); rvo_hi = base1.RVO_d.mean()
    P3 = bool(rvo_lo < 0.5 * rvo_hi) if rvo_hi > 0.005 else np.nan
    p4 = s[s.block == "P4"]
    def gg(ad, d): return p4[(p4.adapt == ad) & (p4.drill == d)].set_index("seed").ERR_d
    inter = (gg(True, 4) - gg(True, 0)) - (gg(False, 4) - gg(False, 0))
    rows.append(dict(scenario=sc, P1=P1, P3=P3, P4=bool(inter.mean() > 0), inter=float(inter.mean()), tau_opt=float(t_opt), tau_opt_all=float(Mall.idxmax()),
                     ERRd_tau1=float(base1.ERR_d.mean()), ECPd_tau1=float(base1.ECP_d.mean()), margin=float(min(d_full.mean(), d_low.mean())), ERRd_max=float(mu.max())))
r7 = pd.DataFrame(rows).set_index("scenario").join(des)
r7.to_csv(OUT + "T_global.csv")
params = ["lam_cell", "serv_mean_min", "hscale", "half_life_h", "kappa", "sigma_u", "completeness", "z_shift", "a_A", "b_H", "rho_sz", "eps_shift", "k"]
def prcc(X, y):
    R = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])]); ry = stats.rankdata(y); out = []
    for j in range(X.shape[1]):
        o = np.delete(R, j, axis=1); o = np.column_stack([np.ones(len(o)), o])
        e1 = R[:, j] - o @ np.linalg.lstsq(o, R[:, j], rcond=None)[0]; e2 = ry - o @ np.linalg.lstsq(o, ry, rcond=None)[0]
        r = np.corrcoef(e1, e2)[0, 1]; n = len(y); tstat = r * math.sqrt((n - 2 - (X.shape[1] - 1)) / (1 - r ** 2))
        out.append((r, 2 * (1 - stats.t.cdf(abs(tstat), n - 2 - (X.shape[1] - 1)))))
    return out
X = r7[params].values.astype(float)
S["X7"] = dict(n=int(len(r7)), P1_share=float(r7.P1.mean()), P3_share=float(np.nanmean(r7.P3.astype(float))), P3_n=int(r7.P3.notna().sum()),
               P4_share=float(r7.P4.mean()), tau_opt_quartiles=[float(x) for x in np.percentile(r7.tau_opt_all, [25, 50, 75])])
pr = {}
for yname in ("margin", "tau_opt_all", "ERRd_tau1", "ECPd_tau1"):
    pr[yname] = {p_: dict(r=float(r), p=float(pv)) for p_, (r, pv) in zip(params, prcc(X, r7[yname].values))}
S["X7_prcc"] = pr
# logistic-free failure description: compare medians when P1 fails
S["X7_P1_fail_medians"] = r7[~r7.P1][params].median().round(3).to_dict(); S["X7_P1_hold_medians"] = r7[r7.P1][params].median().round(3).to_dict()
nice = dict(lam_cell="Exception rate λ", serv_mean_min="Mean service time", hscale="Horizon scale", half_life_h="Skill half-life", kappa="AI over-confidence κ",
            sigma_u="Uncertainty noise σᵤ", completeness="Twin completeness c", z_shift="Shift magnitude Δz", a_A="AI accuracy intercept a_A",
            b_H="Human difficulty slope b_H", rho_sz="Severity–difficulty corr. ρ", eps_shift="Twin model error in shift", k="Supervisors k")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
order = sorted(params, key=lambda p_: abs(pr["ERRd_tau1"][p_]["r"]))
for ax, yname, ttl in zip(axs, ("ERRd_tau1", "margin"), ("(a) ERR during shift at τ = 1", "(b) P1 margin (inverted-U depth)")):
    vals = [pr[yname][p_]["r"] for p_ in order]; sigs = [pr[yname][p_]["p"] < 0.01 for p_ in order]
    ax.barh(range(len(order)), vals, color=[C1 if v >= 0 else C2 for v in vals], alpha=1, height=.62, edgecolor="white")
    for i_, (v, sg) in enumerate(zip(vals, sigs)):
        if not sg: ax.barh(i_, v, color="white", height=.62, edgecolor=C1 if v >= 0 else C2, hatch="////", linewidth=.6)
    ax.axvline(0, color=INK2, lw=.8); ax.set_xlim(-1, 1); ax.set_title(ttl, loc="left"); ax.set_xlabel("PRCC"); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([nice[p_] for p_ in order])
axs[1].legend(handles=[Patch(color=C1, label="positive, p < 0.01"), Patch(color=C2, label="negative, p < 0.01"),
                       Patch(facecolor="white", edgecolor=INK2, hatch="////", label="p ≥ 0.01")], loc="upper center", bbox_to_anchor=(-0.05, -0.16), ncol=3, fontsize=7)
fig.tight_layout(); fig.savefig(FIG + "fig11_sensitivity.png"); plt.close(fig)

# ---------------- X9 recovery ----------------
x9 = pd.read_csv(OUT + "X9.csv"); ser = np.load(OUT + "X9_series.npy", allow_pickle=True)
edges = np.arange(72, 720 + 1e-9, 6.0); mid = edges[:-1] + 3
fig, ax = plt.subplots(figsize=(6.6, 2.9)); S["X9"] = {}
labs = {"tau_-0.5": ("τ = −0.5 (A ≈ 0.57)", C4), "tau_1": ("τ = 1 (A ≈ 0.89)", C1), "tau_4": ("τ = 4 (A ≈ 0.999)", C2),
        "tau_4_adaptive": ("τ = 4 + shift-aware escalation", C3), "tau_4_adaptive_drills": ("… + 4 drills/day", INK2)}
for c_, (lb, col) in labs.items():
    m = x9.config.values == c_; mean = np.nanmean(ser[m], axis=0)
    ax.plot(mid, mean, color=col, lw=1.4, label=lb, ls="--" if c_.endswith("drills") else "-")
    S["X9"][c_] = dict(max_degr=float(x9[m].max_degr.mean()), max_degr_ci=float(ci(x9[m].max_degr)), excess_loss=float(x9[m].excess_loss.mean()),
                       excess_loss_ci=float(ci(x9[m].excess_loss)), t_rec=float(np.nanmean(x9[m].t_recover_h)), t_rec_nan=int(x9[m].t_recover_h.isna().sum()),
                       ERR_n=float(x9[m].ERR_n.mean()))
ax.axvspan(480, 528, color=GRID, alpha=.7, lw=0); ax.text(486, 0.93, "shift", fontsize=7, color=INK2)
ax.set_xlim(400, 640); ax.set_ylim(0.5, 0.95); ax.set_xlabel("Time (h)"); ax.set_ylabel("ERR in 6-h windows")
ax.legend(loc="lower right", fontsize=6.6, ncol=2)
fig.savefig(FIG + "fig10_recovery.png"); plt.close(fig)

# ---------------- Pareto from X2 ----------------
g2 = x2.groupby(["k", "tau"]).agg(loss=("loss_h", "mean"), ECP=("ECP", "mean"), AR=("AR", "mean"), ERR=("ERR", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(4.6, 3.1))
for k in (1, 2, 3, 4):
    s = g2[g2.k == k]; ax.plot(s.ECP, s.loss, "o", ms=2.5, color=cols[k], label=f"k = {k}")
pts = g2[["ECP", "loss"]].values; eff = []
for i_, (e, l_) in enumerate(pts):
    if not np.any((pts[:, 0] >= e) & (pts[:, 1] <= l_) & ((pts[:, 0] > e) | (pts[:, 1] < l_))): eff.append(i_)
fr = g2.iloc[eff].sort_values("ECP"); ax.plot(fr.ECP, fr.loss, color=INK, lw=1, label="Pareto front (loss, ECP)")
ax.set_xlabel("Command reach ECP (30 days)"); ax.set_ylabel("Expected loss per hour (severity units)"); ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_ylim(top=min(60, g2.loss.max())); fig.savefig(FIG + "fig12_pareto.png"); plt.close(fig)
S["pareto"] = fr.round(4).to_dict("records")
for c_s in (1.0, 3.0):
    g2["total"] = g2.loss + c_s * g2.k; b = g2.loc[g2.total.idxmin()]
    S[f"opt_cost_cs{c_s}"] = dict(k=int(b.k), tau=float(b.tau), loss=float(b.loss), ECP=float(b.ECP), total=float(b.total))
    feas = g2[(g2.ECP >= 0.6)]
    if len(feas): b2 = feas.loc[feas.total.idxmin()]; S[f"opt_cost_cs{c_s}_ECP60"] = dict(k=int(b2.k), tau=float(b2.tau), loss=float(b2.loss), ECP=float(b2.ECP))

# MC precision summary
S["mc_precision"] = dict(X1_ERRd_halfwidth_median=float(g.ERRd_ci.median()), X1_ERRd_halfwidth_max=float(g.ERRd_ci.max()))
json.dump(S, open(OUT + "summary2.json", "w"), indent=1, default=float)
print("analysis done")
