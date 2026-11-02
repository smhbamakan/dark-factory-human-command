import json, math, os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results_ext") + "/"; O = os.path.join(ROOT, "results") + "/"
S = {}


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def se(d):
    return d.std(ddof=1) / math.sqrt(len(d))


# ---------- X1p: p_crit sensitivity and AR with halt time ----------
if os.path.exists(R + "X11.csv"):
    x = pd.read_csv(R + "X11.csv"); x1 = pd.read_csv(O + "X1.csv")
    cols = ["ERR", "ERR_d", "ERR_n", "ECP_d", "AR_d", "autonomy"]
    m = x.merge(x1, on=["N", "k", "tau", "seed"], suffixes=("", "_o"))
    S["X11_repro_maxdiff"] = float(max(np.nanmax(np.abs(m[c] - m[c + "_o"])) for c in cols))
    g = x.groupby(["N", "k", "tau"]).mean(numeric_only=True).reset_index()
    out = {}
    for (N, k), s in g.groupby(["N", "k"]):
        s = s[s.tau < 8]; i = s.ERR_d.idxmax()
        d = dict(tau_opt=float(s.loc[i, "tau"]), A_opt=float(s.loc[i, "autonomy"]))
        for pc in (80, 85, 90, 95):
            d[f"pc{pc}"] = dict(share=float(s[f"share_R_pc{pc}"].mean()), ECPd_at_ERRopt=float(s.loc[i, f"ECP_d_pc{pc}"]),
                                ECPd_max=float(s[f"ECP_d_pc{pc}"].max()), A_at_ECPmax=float(s.loc[s[f"ECP_d_pc{pc}"].idxmax(), "autonomy"]))
        for h in (0, 1, 2, 5):
            d[f"AR_d_h{h}"] = dict(at_opt=float(s.loc[i, f"AR_d_h{h}"]), min=float(s[f"AR_d_h{h}"].min()),
                                   at_lowA=float(s.sort_values("tau").iloc[0][f"AR_d_h{h}"]))
        out[f"N{N}_k{k}"] = d
    S["X11"] = out

# ---------- X6x: corrected abandon mode, difficulty-dependent decision time ----------
if os.path.exists(R + "X12.csv"):
    x6 = pd.read_csv(R + "X12.csv"); rows = []
    for v, s in x6.groupby("variant", sort=False):
        p1 = s[s.block == "P1"]; M = p1.pivot(index="seed", columns="tau", values="ERR_d"); mu = M.mean()
        t = mu.idxmax(); dfu = M[t] - M[8.0]; dlo = M[t] - M[-1.0]
        sig = lambda d: d.mean() > 0 and stats.ttest_1samp(d, 0).pvalue < 0.05
        P1 = bool(t not in (-1.0, 8.0) and sig(dfu) and sig(dlo))
        p3 = s[s.block == "P3"]; lo = p3[p3.completeness == 0.2]; hi = p3[p3.completeness == 0.9]
        P3 = bool(lo.RVO_d.mean() < 0.5 * hi.RVO_d.mean() and lo.ISR_d.mean() < 0.3)
        p4 = s[s.block == "P4"]
        gg = lambda a, d: p4[(p4.adapt == a) & (p4.drill == d)].set_index("seed").ERR_d
        inter = (gg(True, 4) - gg(True, 0)) - (gg(False, 4) - gg(False, 0))
        P4 = bool(inter.mean() > 0 and stats.ttest_1samp(inter, 0).pvalue < 0.10)
        rows.append(dict(variant=v, tau_opt=float(t), A_opt=float(p1[p1.tau == t].autonomy.mean()), drop_full=float(dfu.mean()),
                         drop_low=float(dlo.mean()), P1=P1, RVO_lo=float(lo.RVO_d.mean()), RVO_hi=float(hi.RVO_d.mean()), P3=P3,
                         inter=float(inter.mean()), P4=P4, OL=float(p1[p1.tau == 1.0].OL_min.mean())))
    S["X12"] = rows

# ---------- XSTOP: AI-initiated safe stop ----------
if os.path.exists(R + "X13.csv"):
    xs = pd.read_csv(R + "X13.csv"); xs["tau_stop"] = xs.tau_stop.astype(float)
    g = xs.groupby(["k", "tau", "tau_stop"]).agg(loss=("loss_h", "mean"), ECP=("ECP", "mean"), ERR=("ERR", "mean"),
                                                   PS=("PS", "mean"), EA=("EA", "mean")).reset_index()
    out = {}
    for k, s in g.groupby("k"):
        no = s[np.isinf(s.tau_stop)]; a = no.loc[no.loss.idxmin()]; b = s.loc[s.loss.idxmin()]
        e = no.loc[no.ERR.idxmax()]
        # paired CI for loss difference between best-with-stop and best-without
        da = xs[(xs.k == k) & (xs.tau == a.tau) & np.isinf(xs.tau_stop)].set_index("seed").loss_h
        db = xs[(xs.k == k) & (xs.tau == b.tau) & (xs.tau_stop == b.tau_stop)].set_index("seed").loss_h
        dd = db - da
        out[f"k{k}"] = dict(no_stop=dict(tau=float(a.tau), loss=float(a.loss), ECP=float(a.ECP), ERR=float(a.ERR)),
                            ERRopt_no_stop=dict(tau=float(e.tau), loss=float(e.loss), ECP=float(e.ECP), ERR=float(e.ERR)),
                            with_stop=dict(tau=float(b.tau), tau_stop=float(b.tau_stop), loss=float(b.loss), ECP=float(b.ECP), ERR=float(b.ERR)),
                            dloss=[float(dd.mean()), ci(dd)])
        # best loss per tau when stopping is allowed
        out[f"k{k}"]["curve"] = s.groupby("tau").loss.min().round(3).to_dict()
    S["X13"] = out

# ---------- X7b: global sensitivity with behavioural parameters ----------
if os.path.exists(R + "X14.csv"):
    des = pd.read_csv(R + "X14_design.csv").set_index("scenario"); x7 = pd.read_csv(R + "X14.csv")
    rows = []
    for sc, s in x7.groupby("scenario"):
        p1 = s[s.block == "P1"]; M = p1.pivot(index="seed", columns="tau", values="ERR_d"); mu = M.mean(); t = mu.idxmax()
        dfu = M[t] - M[8.0]; dlo = M[t] - M[-0.5]
        if t == 8.0: c = "boundary_full"
        elif t == -0.5: c = "boundary_low"
        elif dfu.mean() > 2 * se(dfu) and dlo.mean() > 2 * se(dlo): c = "pass"
        else: c = "interior_nonsig"
        base1 = p1[p1.tau == 1.0]; p3 = s[s.block == "P3"]
        P3 = bool(p3.RVO_d.mean() < 0.5 * base1.RVO_d.mean() and p3.ISR_d.mean() < 0.3) if base1.RVO_d.mean() > 0.005 else np.nan
        p4 = s[s.block == "P4"]
        gg = lambda a, d: p4[(p4.adapt == a) & (p4.drill == d)].set_index("seed").ERR_d
        inter = (gg(True, 4) - gg(True, 0)) - (gg(False, 4) - gg(False, 0))
        rows.append(dict(scenario=sc, cls=c, P1=c == "pass", P3=P3, P4pos=bool(inter.mean() > 0),
                         P4sig=bool(inter.mean() > 0 and stats.ttest_1samp(inter, 0).pvalue < 0.10),
                         margin=float(min(dfu.mean(), dlo.mean())), ERRd_tau1=float(base1.ERR_d.mean())))
    r = pd.DataFrame(rows).set_index("scenario").join(des)
    params = [c for c in des.columns]
    X = r[params].values.astype(float)

    def prcc(X, y):
        Rk = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])]); ry = stats.rankdata(y); res = []
        for j in range(X.shape[1]):
            o = np.column_stack([np.ones(len(Rk)), np.delete(Rk, j, axis=1)])
            e1 = Rk[:, j] - o @ np.linalg.lstsq(o, Rk[:, j], rcond=None)[0]; e2 = ry - o @ np.linalg.lstsq(o, ry, rcond=None)[0]
            rr = np.corrcoef(e1, e2)[0, 1]; df = len(y) - 2 - (X.shape[1] - 1); tt = rr * math.sqrt(df / (1 - rr ** 2))
            res.append((float(rr), float(2 * (1 - stats.t.cdf(abs(tt), df)))))
        return dict(zip(params, res))
    S["X14"] = dict(n=len(r), classes=r.cls.value_counts().to_dict(), P1_share=float(r.P1.mean()),
                    P3_share=float(np.nanmean(r.P3.astype(float))), P3_n=int(r.P3.notna().sum()),
                    P4pos_share=float(r.P4pos.mean()), P4sig_share=float(r.P4sig.mean()),
                    prcc_margin=prcc(X, r.margin.values), prcc_ERRd=prcc(X, r.ERRd_tau1.values))
    r.to_csv(R + "T_global_X14.csv")

# ---------- re-analysis of supplied X7 (criteria) and X3 (lag) ----------
x7 = pd.read_csv(O + "X7.csv"); cls = {}; p4s = 0; p4p = 0; p3 = []
for sc, s in x7.groupby("scenario"):
    p1 = s[s.block == "P1"]; M = p1.pivot(index="seed", columns="tau", values="ERR_d"); t = M.mean().idxmax()
    dfu = M[t] - M[8.0]; dlo = M[t] - M[-0.5]
    c = "boundary_full" if t == 8.0 else "boundary_low" if t == -0.5 else "pass" if (dfu.mean() > 2 * se(dfu) and dlo.mean() > 2 * se(dlo)) else "interior_nonsig"
    cls[c] = cls.get(c, 0) + 1
    p4 = s[s.block == "P4"]; gg = lambda a_, d_: p4[(p4.adapt == a_) & (p4.drill == d_)].set_index("seed").ERR_d
    inter = (gg(True, 4) - gg(True, 0)) - (gg(False, 4) - gg(False, 0))
    p4p += inter.mean() > 0; p4s += bool(inter.mean() > 0 and stats.ttest_1samp(inter, 0).pvalue < 0.10)
    b1 = p1[p1.tau == 1.0]; q3 = s[s.block == "P3"]
    if b1.RVO_d.mean() > 0.005: p3.append(bool(q3.RVO_d.mean() < 0.5 * b1.RVO_d.mean() and q3.ISR_d.mean() < 0.3))
S["X7_reanalysis"] = dict(P1_classes=cls, P4_positive=int(p4p), P4_sig10=int(p4s), P3_with_engagement=[int(sum(p3)), len(p3)])
x3 = pd.read_csv(O + "X3.csv")
S["X3_lag_all"] = x3[x3.part == "lag"].groupby("lag_s")[["ERR", "ERR_d", "ECP", "ECP_d", "miss_s2"]].mean().round(4).reset_index().to_dict("records")

json.dump(S, open(R + "summary_ext.json", "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float)[:6000])
