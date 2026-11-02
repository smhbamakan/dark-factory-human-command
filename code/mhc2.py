"""
MHC simulator v2 -- event-calendar discrete-event simulation of mediated human
command in a dark factory.

Changes relative to v1 (in response to peer review):
  * true event calendar (arrival, service start, service end, drill start/end,
    drill interruption, deadline expiry) with chronological state updates;
  * dispatch policies FCFS / EDF / SEV (severity priority);
  * Poisson or common-cause batch (shock) arrivals;
  * severity separated from difficulty (Gaussian copula, correlation rho_sz);
    consequence horizons depend on severity, not novelty;
  * vector digital-twin fidelity: synchronisation lag, state completeness,
    predictive model error (rises during shift until recalibration);
  * AI actuation latency;
  * outcome taxonomy: correct resolution (CR), protective stop (PS),
    erroneous action (EA); asymmetric loss;
  * constructs separated: ERR (reliability), ECP (informed command reach),
    AR (authority readiness), ISR (independent scrutiny), RVO (reliability
    value of oversight vs. matched AI-only counterfactual);
  * exact busy-fraction workload integration; learning at completion time;
  * interruptible idle-time drills with full accounting;
  * rival functional forms for engagement, vigilance, forgetting, operator
    heterogeneity and chance level.
All randomness is pre-drawn per exception, so the FCFS path can be checked
exactly against an independent Kiefer-Wolfowitz implementation (verify.py).
"""
from dataclasses import dataclass, replace, field
import heapq, math
import numpy as np
from scipy.stats import norm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class P:
    # plant / arrivals
    N: int = 40
    lam_cell: float = 0.25
    T: float = 720.0
    warmup: float = 72.0
    arrivals: str = "poisson"          # 'poisson' | 'shock'
    shock_rate: float = 0.25           # shocks per hour (mean-rate preserving)
    shock_batch: float = 8.0           # mean exceptions per shock
    shock_spread_min: float = 5.0
    shock_dz: float = 0.5              # extra difficulty of shock-induced cases
    # difficulty (novelty classes)
    class_p: tuple = (0.80, 0.17, 0.03)
    class_mu_z: tuple = (-1.0, 0.5, 2.0)
    # severity (separate from difficulty)
    sev_p: tuple = (0.60, 0.30, 0.10)
    sev_w: tuple = (1.0, 3.0, 10.0)
    sev_horizon_min: tuple = (60.0, 30.0, 15.0)
    horizon_cv: float = 0.6
    rho_sz: float = 0.3                # copula correlation severity~difficulty
    # AI
    a_A: float = 2.5
    b_A: float = 1.5
    sigma_u: float = 0.6
    kappa: float = 0.6
    ai_lat_min: float = 0.5
    tau: float = 1.0
    adapt_tau: bool = False
    tau_low: float = 0.5
    det_delay: float = 2.0
    # supervisors
    k: int = 1
    a_H: float = 1.0
    sd_aH: float = 0.0                 # operator heterogeneity
    gamma_S: float = 1.5
    b_H: float = 0.6
    serv_mean_min: float = 6.0
    serv_cv: float = 0.5
    rubber_frac: float = 0.35
    policy: str = "FCFS"               # FCFS | EDF | SEV
    expire_mode: str = "restart"       # restart | abandon
    n_actions: int = 2                 # chance level 1/M
    # digital twin (vector fidelity)
    lag_s: float = 10.0
    completeness: float = 0.9
    eps0: float = 0.05
    eps_shift: float = 0.15
    recal_h: float = 12.0              # twin recalibration delay after shift onset
    omega: float = 0.4
    zeta: float = 0.8
    # engagement
    eng_mode: str = "logistic"         # logistic | constant | linear | forced
    kF: float = 10.0
    F0: float = 0.55
    beta_const: float = 0.9
    # vigilance
    vig_mode: str = "invU"             # invU | none | monotone | fatigue
    w_star: float = 0.55
    w_sd: float = 0.35
    psi_floor: float = 0.6
    ema_T: float = 2.0
    fatigue_T: float = 8.0
    # skill
    S0: float = 0.9
    S_min: float = 0.15
    half_life_h: float = 240.0
    forget: str = "exp"                # exp | power
    eta_real: float = 0.03
    z_hard: float = 0.0
    # drills
    drill_rate_day: float = 0.0
    drill_min: float = 15.0
    eta_drill: float = 0.06
    drill_expiry_h: float = 24.0
    # shift
    disrupt: bool = True
    t_d: float = 480.0
    d_len: float = 48.0
    z_shift: float = 1.0
    novel_mult: float = 3.0
    # command-relevant set: AI accuracy below p_crit
    p_crit: float = 0.9
    # losses (per unit severity weight)
    loss_ps: float = 1.0
    loss_ea: float = 5.0
    # analytic benchmark mode
    analytic: bool = False
    window_h: float = 6.0


def draw(p: P, seed: int):
    """Pre-draw all exogenous randomness for one replication."""
    rng = np.random.default_rng(seed)
    Lam = p.N * p.lam_cell
    if p.arrivals == "shock":
        lam_b = max(Lam - p.shock_rate * p.shock_batch, 0.0)
        tb = rng.uniform(0, p.T, rng.poisson(lam_b * p.T))
        ns = rng.poisson(p.shock_rate * p.T)
        ts = rng.uniform(0, p.T, ns)
        bs = np.maximum(1, rng.poisson(p.shock_batch, ns))
        tsh = np.concatenate([t0 + rng.uniform(0, p.shock_spread_min / 60, b) for t0, b in zip(ts, bs)]) if ns else np.array([])
        t = np.concatenate([tb, tsh]); shock = np.concatenate([np.zeros(len(tb), bool), np.ones(len(tsh), bool)])
        o = np.argsort(t); t = t[o]; shock = shock[o]
    else:
        t = np.sort(rng.uniform(0, p.T, rng.poisson(Lam * p.T))); shock = np.zeros(len(t), bool)
    n = len(t)
    in_d = ((t >= p.t_d) & (t < p.t_d + p.d_len)) if p.disrupt else np.zeros(n, bool)
    cp = np.array(p.class_p, float); cp /= cp.sum()
    cls = rng.choice(3, size=n, p=cp)
    if p.disrupt and in_d.any():
        cpd = cp.copy(); cpd[2] *= p.novel_mult; cpd /= cpd.sum()
        cls[in_d] = rng.choice(3, size=in_d.sum(), p=cpd)
    eps_z = rng.normal(0, 1, n)
    z = np.array(p.class_mu_z)[cls] + eps_z + p.z_shift * in_d + p.shock_dz * shock
    u = z - p.kappa * p.z_shift * in_d + rng.normal(0, p.sigma_u, n)
    # severity via Gaussian copula with standardised difficulty innovation
    ys = p.rho_sz * eps_z + math.sqrt(max(0.0, 1 - p.rho_sz ** 2)) * rng.normal(0, 1, n)
    q = np.cumsum(p.sev_p)[:-1] / np.sum(p.sev_p)
    sev = np.searchsorted(norm.ppf(q), ys)
    hm = np.array(p.sev_horizon_min)[sev] / 60.0
    if p.analytic:
        H = rng.exponential(hm)
    else:
        s2 = math.log(1 + p.horizon_cv ** 2)
        H = hm * np.exp(rng.normal(-s2 / 2, math.sqrt(s2), n))
    return dict(t=t, n=n, in_d=in_d, cls=cls, z=z, u=u, sev=sev, H=H, shock=shock,
                w=np.array(p.sev_w)[sev],
                U_ai=rng.uniform(size=n), U_eng=rng.uniform(size=n), U_hum=rng.uniform(size=n),
                Z_serv=rng.normal(size=n), Z_lat=rng.normal(size=n), E_serv=rng.exponential(size=n),
                aH=p.a_H + p.sd_aH * rng.normal(size=p.k),
                drills=[np.sort(rng.uniform(0, p.T, rng.poisson(p.drill_rate_day / 24 * p.T))) for _ in range(p.k)])


def twin_quality(p, tt):
    """Information quality q = completeness * (1 - model error) at time tt."""
    eps = p.eps0
    if p.disrupt and p.t_d <= tt < p.t_d + min(p.recal_h, p.d_len):
        eps = p.eps0 + p.eps_shift
    return p.completeness * (1 - eps)


def engagement(p, q):
    if p.eng_mode == "forced": return 1.0
    if p.eng_mode == "constant": return p.beta_const
    if p.eng_mode == "linear": return float(np.clip(q, 0, 1))
    return float(sigmoid(p.kF * (q - p.F0)))


def tau_at(p, tt):
    if p.adapt_tau and p.disrupt and (p.t_d + p.det_delay) <= tt < (p.t_d + p.d_len):
        return min(p.tau, p.tau_low)
    return p.tau


class Server:
    __slots__ = ("S", "t_skill", "w", "wf", "t_w", "busy", "free_at", "task", "aH", "busy_hist")
    def __init__(self, S0, aH):
        self.S = S0; self.t_skill = 0.0; self.w = 0.0; self.wf = 0.0; self.t_w = 0.0
        self.busy = False; self.free_at = 0.0; self.task = None; self.aH = aH; self.busy_hist = []


def simulate(p: P, seed: int = 0, D=None, record=False, check=False):
    D = D if D is not None else draw(p, seed)
    t, n, z, u, H, w, sev, cls, in_d = D["t"], D["n"], D["z"], D["u"], D["H"], D["w"], D["sev"], D["cls"], D["in_d"]
    pA = sigmoid(p.a_A - p.b_A * z)
    lam_dec = math.log(2) / p.half_life_h
    p0 = 1.0 / p.n_actions
    smean = p.serv_mean_min / 60.0
    s2 = math.log(1 + p.serv_cv ** 2); ssd = math.sqrt(s2)
    lag = p.lag_s / 3600.0
    srv = [Server(p.S0, D["aH"][j]) for j in range(p.k)]

    # outputs
    defer = np.zeros(n, bool); eng = np.zeros(n, bool); outcome = np.zeros(n, np.int8)  # 0 CR,1 PS,2 EA
    wait = np.full(n, np.nan); soj = np.full(n, np.nan); S_at = np.full(n, np.nan); q_at = np.full(n, np.nan)
    ar_ok = np.zeros(n, bool); dec_time = np.full(n, np.nan)
    ai_out = np.zeros(n, np.int8)   # matched AI-only counterfactual
    drill_stats = dict(scheduled=0, completed=0, interrupted=0, expired=0)
    pending_drills = [[] for _ in range(p.k)]
    log_service = []  # (j, start, end) for invariant checks

    # ---- helpers ------------------------------------------------------
    def skill_to(sv, now):
        dt = now - sv.t_skill
        if dt > 0:
            if p.forget == "power":
                sv.S = p.S_min + (sv.S - p.S_min) * (1 + dt / (p.half_life_h / 1.0)) ** (-1.0)
            else:
                sv.S = p.S_min + (sv.S - p.S_min) * math.exp(-lam_dec * dt)
            sv.t_skill = now

    def work_to(sv, now):
        dt = now - sv.t_w
        if dt > 0:
            b = 1.0 if sv.busy else 0.0
            e = math.exp(-dt / p.ema_T); sv.w = sv.w * e + b * (1 - e)
            ef = math.exp(-dt / p.fatigue_T); sv.wf = sv.wf * ef + b * (1 - ef)
            sv.t_w = now

    def vigilance(sv):
        if p.vig_mode == "none": return 1.0
        if p.vig_mode == "monotone": return 1.0 - (1 - p.psi_floor) * sv.w
        if p.vig_mode == "fatigue": return 1.0 - (1 - p.psi_floor) * sv.wf
        return p.psi_floor + (1 - p.psi_floor) * math.exp(-0.5 * ((sv.w - p.w_star) / p.w_sd) ** 2)

    # AI-only counterfactual and AI path outcome
    if p.analytic:
        lat = np.zeros(n)
    else:
        sl2 = math.log(1 + 0.5 ** 2)
        lat = (p.ai_lat_min / 60.0) * np.exp(-sl2 / 2 + math.sqrt(sl2) * D["Z_lat"])
    ai_ok = D["U_ai"] < pA
    ai_out[:] = np.where(lat > H, 1, np.where(ai_ok, 0, 2))

    # event calendar: (time, order, seq, type, payload)
    ev = []; seq = 0
    def push(tt, order, typ, data):
        nonlocal seq; heapq.heappush(ev, (tt, order, seq, typ, data)); seq += 1
    for i in range(n):
        push(t[i], 2, "arr", i)
    for j in range(p.k):
        for td in D["drills"][j]:
            push(td, 3, "dsched", j); drill_stats["scheduled"] += 1
    queue = []   # heap of (key, seq, i)
    qseq = 0

    def key_of(i):
        if p.policy == "EDF": return (t[i] + H[i] - lag,)
        if p.policy == "SEV": return (-sev[i], t[i])
        return (t[i],)

    def start_service(j, i, now):
        sv = srv[j]
        work_to(sv, now); skill_to(sv, now)
        if sv.task is not None and sv.task[0] == "drill":   # interrupt drill
            drill_stats["interrupted"] += 1
        S_at[i] = sv.S
        q = 1.0 if p.analytic else twin_quality(p, now)
        q_at[i] = q
        if p.analytic:
            beta = 1.0; psi = 1.0
            st = smean * D["E_serv"][i]
            ph = sigmoid(sv.aH + p.gamma_S * sv.S - p.b_H * z[i])
            engaged = True
        else:
            beta = engagement(p, q); psi = vigilance(sv)
            engaged = D["U_eng"][i] < beta
            m = smean * (1 + p.zeta * (1 - q)) * (1.0 if engaged else p.rubber_frac)
            st = m * math.exp(-s2 / 2 + ssd * D["Z_serv"][i])
            ph = sigmoid(sv.aH + p.gamma_S * sv.S - p.b_H * z[i])
            ph = p.omega * ph + (1 - p.omega) * (q * ph + (1 - q) * p0)
            ph = p0 + (ph - p0) * psi
            if not engaged: ph = pA[i]
        eng[i] = engaged
        sv.busy = True; sv.task = ("case", i, ph); sv.free_at = now + st
        wait[i] = now - t[i]
        push(now + st, 1, "end", (j, i))
        log_service.append((j, now, now + st, "case"))

    def try_dispatch(now):
        for j in sorted(range(p.k), key=lambda jj: (srv[jj].free_at, jj)):
            sv = srv[j]
            if (not sv.busy or (sv.task and sv.task[0] == "drill")) and queue:
                if p.expire_mode == "abandon":
                    while queue and outcome[queue[0][2]] == 1 and now >= t[queue[0][2]] + H[queue[0][2]]:
                        heapq.heappop(queue)
                    if not queue: return
                _, _, i = heapq.heappop(queue)
                if sv.task and sv.task[0] == "drill":
                    work_to(sv, now); sv.busy = False
                start_service(j, i, now)

    def try_drill(j, now):
        sv = srv[j]
        if sv.busy or queue: return
        pd_ = pending_drills[j]
        while pd_ and now - pd_[0] > p.drill_expiry_h:
            pd_.pop(0); drill_stats["expired"] += 1
        if pd_:
            pd_.pop(0)
            work_to(sv, now); sv.busy = True; sv.task = ("drill", now); sv.free_at = now + p.drill_min / 60
            push(now + p.drill_min / 60, 1, "dend", (j, now))

    while ev:
        now, _, _, typ, data = heapq.heappop(ev)
        if typ == "arr":
            i = data
            # authority readiness at arrival: earliest residual among supervisors (drills interruptible)
            res = min(0.0 if (not s.busy or (s.task and s.task[0] == "drill")) else s.free_at - now for s in srv)
            ar_ok[i] = res + lag < H[i]
            if u[i] > tau_at(p, now):
                defer[i] = True
                heapq.heappush(queue, (key_of(i), qseq, i)); qseq += 1
                if p.expire_mode == "abandon":
                    push(t[i] + H[i], 0, "exp", i)
                try_dispatch(now)
            else:
                outcome[i] = ai_out[i]
        elif typ == "end":
            j, i = data; sv = srv[j]
            if sv.task is None or sv.task[0] != "case" or sv.task[1] != i:
                continue
            work_to(sv, now); skill_to(sv, now)
            ph = sv.task[2]
            soj[i] = now - t[i]
            if outcome[i] == 1 or soj[i] + lag > H[i]:
                outcome[i] = 1
            else:
                outcome[i] = 0 if D["U_hum"][i] < ph else 2
                dec_time[i] = now
            if eng[i] and z[i] > p.z_hard:
                sv.S = sv.S + p.eta_real * (1 - sv.S)
            sv.busy = False; sv.task = None; sv.free_at = now
            try_dispatch(now)
            if not sv.busy: try_drill(j, now)
        elif typ == "dsched":
            j = data; pending_drills[j].append(now); try_drill(j, now)
        elif typ == "dend":
            j, t0 = data; sv = srv[j]
            if sv.task is None or sv.task[0] != "drill" or sv.task[1] != t0:
                continue   # interrupted earlier
            work_to(sv, now); skill_to(sv, now)
            sv.S = sv.S + p.eta_drill * (1 - sv.S); drill_stats["completed"] += 1
            sv.busy = False; sv.task = None; sv.free_at = now
            log_service.append((j, t0, now, "drill"))
            try_dispatch(now)
            if not sv.busy: try_drill(j, now)
        elif typ == "exp":
            i = data
            if np.isnan(wait[i]):          # still queued: protective stop, abandon
                outcome[i] = 1
                soj[i] = H[i]

    # protective stop for queued cases at horizon (restart mode): already captured at service end
    drill_stats["pending_end"] = sum(len(x) for x in pending_drills)
    out = summarize(p, D, pA, defer, eng, outcome, ai_out, wait, soj, S_at, q_at, ar_ok, lag)
    out.update({f"drill_{k}": v for k, v in drill_stats.items()})
    if check:
        out["_log"] = log_service
    if record:
        out["_raw"] = dict(t=t, defer=defer, eng=eng, outcome=outcome, ai_out=ai_out, wait=wait, soj=soj,
                           S_at=S_at, w=w, sev=sev, cls=cls, in_d=in_d, z=z, H=H, pA=pA, ar_ok=ar_ok)
    return out


def summarize(p, D, pA, defer, eng, outcome, ai_out, wait, soj, S_at, q_at, ar_ok, lag):
    t, w, in_d, z, sev = D["t"], D["w"], D["in_d"], D["z"], D["sev"]
    keep = t >= p.warmup
    base = keep & ~in_d; dm = keep & in_d
    early = dm & (t < p.t_d + 12.0)
    R = pA < p.p_crit                     # command-relevant: AI unreliable
    def wmean(x, m):
        return float((w[m] * x[m]).sum() / w[m].sum()) if m.any() else np.nan
    CR = outcome == 0; PS = outcome == 1; EA = outcome == 2
    cmd = defer & eng & (outcome != 1)
    loss = w * (p.loss_ps * PS + p.loss_ea * EA)
    loss_ai = w * (p.loss_ps * (ai_out == 1) + p.loss_ea * (ai_out == 2))
    span = p.T - p.warmup
    o = {}
    for tag, m in (("", keep), ("_n", base), ("_d", dm), ("_e", early)):
        o["ERR" + tag] = wmean(CR, m)
        o["ERRai" + tag] = wmean(ai_out == 0, m)
        o["ECP" + tag] = wmean(cmd, m & R)
        o["AR" + tag] = wmean(ar_ok, m & R)
        o["PS" + tag] = wmean(PS, m); o["EA" + tag] = wmean(EA, m)
    o["RVO_n"] = o["ERR_n"] - o["ERRai_n"]; o["RVO_d"] = o["ERR_d"] - o["ERRai_d"] if dm.any() else np.nan
    dk = defer & keep
    o["autonomy"] = 1 - float(defer[base].mean())
    o["ISR"] = float(eng[dk].mean()) if dk.any() else np.nan
    o["ISR_d"] = float(eng[dk & in_d].mean()) if (dk & in_d).any() else np.nan
    o["OL_min"] = float(np.nanmean(soj[dk]) * 60) if dk.any() else np.nan
    o["Wq_p95_min"] = float(np.nanpercentile(wait[dk], 95) * 60) if dk.any() else np.nan
    o["MTBHI_min"] = float(span * 60 / max(1, dk.sum()))
    o["P_timely"] = float(1 - PS[dk].mean()) if dk.any() else np.nan
    for s in range(3):
        m = dk & (sev == s)
        o[f"miss_s{s}"] = float(PS[m].mean()) if m.any() else np.nan
    o["skill_n"] = float(np.nanmean(S_at[dk & ~in_d])) if (dk & ~in_d).any() else np.nan
    o["skill_d"] = float(np.nanmean(S_at[dk & in_d])) if (dk & in_d).any() else np.nan
    o["q_d"] = float(np.nanmean(q_at[dk & in_d])) if (dk & in_d).any() else np.nan
    o["loss_h"] = float(loss[keep].sum() / span)
    o["loss_ai_h"] = float(loss_ai[keep].sum() / span)
    o["loss_d"] = float(loss[dm].sum()) if dm.any() else np.nan
    o["util"] = float(np.nansum((soj - wait)[dk]) / (p.k * span)) if dk.any() else 0.0
    o["util_d"] = float(np.nansum((soj - wait)[dk & in_d]) / (p.k * p.d_len)) if (dk & in_d).any() else 0.0
    o["util_n"] = float(np.nansum((soj - wait)[dk & ~in_d]) / (p.k * (span - (p.d_len if p.disrupt else 0)))) if (dk & ~in_d).any() else 0.0
    # recovery metrics on 6-h windows
    if p.disrupt:
        edges = np.arange(p.warmup, p.T + 1e-9, p.window_h)
        idx = np.digitize(t, edges) - 1
        ser = np.array([wmean(CR, keep & (idx == b)) for b in range(len(edges) - 1)])
        ref = o["ERR_n"]
        o["max_degr"] = float(ref - np.nanmin(ser))
        lossw = np.array([loss[keep & (idx == b)].sum() for b in range(len(edges) - 1)])
        base_rate = float(loss[base].sum() / (base.sum() and (span - p.d_len)))
        after = edges[:-1] >= p.t_d
        o["excess_loss"] = float((lossw[after] - base_rate * p.window_h).sum())
        end_d = p.t_d + p.d_len
        rec = np.nan
        for b in range(len(edges) - 1):
            if edges[b] >= end_d and np.all(ser[b:b + 2] >= ref - 0.02):
                rec = edges[b] - end_d; break
        o["t_recover_h"] = rec
        o["_series"] = ser
    return o


# ---------------- analytic results ----------------
def erlang_c(k, a):
    if a >= k: return 1.0
    s = sum(a ** m / math.factorial(m) for m in range(k))
    last = a ** k / math.factorial(k) * k / (k - a)
    return last / (s + last)

def p_timely_mmk(lam_d, mu, k, theta):
    a = lam_d / mu
    if a >= k: return 0.0
    C = erlang_c(k, a); r = k * mu - lam_d
    return ((1 - C) + C * r / (r + theta)) * mu / (mu + theta)


def run(p, reps, seed0=1000, keys=None):
    rs = [simulate(p, seed=seed0 + r) for r in range(reps)]
    keys = keys or [k for k in rs[0] if not k.startswith("_")]
    return {k: np.array([r[k] for r in rs], float) for k in keys}
