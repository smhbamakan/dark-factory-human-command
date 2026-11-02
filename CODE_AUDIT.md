# Code audit and additional experiments (manuscript revision v0.8)

## 1. Reproduction

All supplied scripts were re-run from a clean copy (Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, 2 cores).

| Step | Result |
|---|---|
| `verify.py` | `verify_summary.json` identical; all `verify_*.csv` identical |
| `experiments.py X1 X1b X2 X3 X4 X5 X6 X7 X9 X10` | every CSV identical to the supplied one (maximum absolute difference 0.0) |
| `analysis.py` | `summary2.json` identical |

The numbers reported in the manuscript therefore follow from the code as supplied.

## 2. Consistency of code and manuscript

Checked against Sections 4–6 and Algorithm 1: arrival and shock process, copula, horizons, AI accuracy and latency, uncertainty signal, shift-aware threshold, twin quality, engagement, vigilance, decision time, skill decay and learning, drills, timeliness rule, outcome taxonomy, ERR/ECP/AR/ISR/RVO, warm-up and windows. They agree, with the points below.

Answers to the open questions in the manuscript (now written into the text):

* Dispatch (`try_dispatch`) orders supervisors by `free_at`, so idle supervisors are served before drilling ones and, among idle ones, the one free longest goes first (ties by index).
* Drilling sets `busy = True`, so drills count as busy time in the workload filter (Eq. 16).
* Learning occurs at completion whenever the case was engaged and `z > z_hard`, also for cases that expired in restart mode.
* The lag experiment reports ECP over the 30-day window at τ = 1 (`ECP`); the model-error experiment reports the shift window (`ECP_d`). Table 6 reports 30-day ERR, ECP and loss.
* The recovery time is the mean over replications of the start of the first pair of 6-h windows within 0.02 of nominal ERR (8.4–17.1 h).
* V1 covers k ∈ {1, 2, 3}; the reference implementation assigns each case to the supervisor with the earliest free time.

Issues found:

1. **Abandon mode (minor bug).** Only expired cases at the head of the queue were removed; an expired case behind a live one was still served. Fixed in `mhc3.py` (`abandon_fix=True`). Effect on the only affected result (Table 7, abandon variant): none to four decimals (X12).
2. **Global-analysis criteria differ from the structural ones.** In X7, P1 requires the optimum to beat both extremes by two standard errors (not a t-test), P4 only requires a positive interaction (126 of 160 are also significant at p < 0.10), and P3 omits the engagement condition (adding it gives the same 130 of 134). The manuscript now states this.
3. **Fig. 7 axis.** The horizontal axis is the nominal twin quality c(1 − ε0); during the first 12 h of the shift quality is lower. Now stated in the caption.
4. Cosmetic: stale docstrings (`exp2.py`, `out2/`) corrected; README called the 15 alternatives plus baseline "16 structural variants".

## 3. Additional experiments (`code/mhc3.py`, `code/exp3.py`, `code/analysis_ext.py`)

`mhc3.py` extends `mhc2.py`; with default settings it reproduces `mhc2.py` exactly (`exp3.py CHECK`: 0 mismatches) except for the abandon-mode fix. Outputs are in `results_ext/`, summaries in `summary_ext.json`.

| ID | Question | Main result |
|---|---|---|
| X11 | Does the ERR/ECP divergence depend on p_crit? How informative is AR with a halt-execution time? | Baseline ECP during the shift at the ERR-optimal threshold: 0.58 / 0.52 / 0.45 / 0.37 for p_crit = 0.8 / 0.85 / 0.9 / 0.95 (maxima 0.67 / 0.63 / 0.61 / 0.55). Divergence holds in every cell and criterion. With a 5-min halt time AR falls to 0.94 at the optimum and 0.87 under heaviest escalation (0.85 for N = 80, k = 1). X1 outputs reproduced exactly. |
| X12 | Corrected abandon mode; difficulty-dependent decision time | Abandon: unchanged. Difficulty-dependent time: A_opt 0.81, P1/P3/P4 hold; gain over full autonomy 0.066 → 0.049; RVO at baseline quality 0.056 → 0.040. |
| X13 | Is the loss advantage of heavy escalation an artefact of having no safe stop? | Yes. With an AI-initiated stop (τ_stop = 1, no escalation) loss is 15.78/h vs 16.79 for the best policy without stop (k = 1; paired difference −1.01 ± 0.32), and ECP = 0. Under c_PS = 1, c_EA = 5 expected loss gives human command no value. |
| X14 | Global sensitivity with eight behavioural parameters added | P1 64% (103/160; 48 interior but not resolved, 9 optimum at most-escalating grid point), P3 98% (129/131), P4 97% (71% significant). Strongest driver of the P1 margin: engagement midpoint F0 (PRCC −0.41), then twin completeness (+0.36) and staffing (−0.30). |

Re-analysis of supplied X7: of 50 P1 failures, 10 have the optimum at τ = −0.5 and 40 an interior optimum not resolved against one or both extremes; in 18 the gain over full autonomy (median 0.009) is too small to resolve with six replications. Re-analysis of supplied X3 (lag 0 → 300 s): ERR 0.890 → 0.871, ERR_d 0.772 → 0.748, ECP 0.410 → 0.334, ECP_d 0.319 → 0.264.

Run time of all additional experiments: about 8 min on 2 cores:

```bash
python3 code/exp3.py CHECK X11 X12 X13 X14
python3 code/analysis_ext.py
```
