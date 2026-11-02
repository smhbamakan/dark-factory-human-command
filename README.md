# Separating Reliability from Human Command in Dark Factories

[![reproduce](https://github.com/smhbamakan/dark-factory-human-command/actions/workflows/reproduce.yml/badge.svg)](https://github.com/smhbamakan/dark-factory-human-command/actions/workflows/reproduce.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)

Code, raw outputs and figures for the manuscript

> Hosseini Bamakan, S. M., & Mirzabagheri, M. *Separating Reliability from Human Command in Dark Factories: A Queueing and Digital-Twin Model of AI Escalation.* Manuscript submitted for publication.

In a dark factory, a remote supervisor sees an exception only if an AI decision system escalates it. The model separates three quantities that a single oversight score mixes up: **exception-resolution reliability (ERR)**, the **effective command probability (ECP)**, i.e. the share of command-relevant exceptions on which a person makes an engaged and timely decision, and **authority readiness (AR)**. They are computed in an event-calendar discrete-event simulation with severity-dependent deadlines, alternative dispatch rules, common-cause disturbances, a digital twin with lag, completeness and model error, and supervisors whose skill, workload and engagement change over time.

All results are reproducible bit-for-bit from fixed seeds.

## Repository layout

| Path | Content |
|---|---|
| `code/mhc2.py` | Simulator used for all main results (parameter class `P`, pre-drawn randomness `draw`, `simulate`), plus the analytical benchmarks (Erlang C, Proposition 1). Algorithm 1 in the manuscript. |
| `code/verify.py` | Verification checks V1–V6 (Sections 5.3 and 6.1). |
| `code/experiments.py` | Main experiments X1–X10 (Table 4). |
| `code/analysis.py` | Statistics, tables, `results/summary2.json` and Figs. 4–12. |
| `code/mhc3.py` | Extension of `mhc2.py` (AI-initiated safe stop, difficulty-dependent decision time, corrected abandon mode, ECP for several criteria, AR with halt time). With default settings it reproduces `mhc2.py` exactly. |
| `code/exp3.py`, `code/analysis_ext.py` | Additional experiments X11–X14 and their analysis. |
| `results/` | Raw outputs of V1–V6 and X1–X10 (one row per replication, with its seed) and summary files. |
| `results_ext/` | Raw outputs and summary (`summary_ext.json`) of X11–X14. |
| `figures/` | Figures 1–12 at 600 dpi. |
| `diagrams/` | Graphviz sources of Figs. 1–3. |
| [`CODE_AUDIT.md`](CODE_AUDIT.md) | Independent re-run of the package, code–manuscript consistency checks and the additional experiments. |

## Where each result comes from

| Manuscript | Script | Output |
|---|---|---|
| Sections 5.3, 6.1; Fig. 4 (V1–V6) | `verify.py` | `results/verify_*.csv`, `verify_summary.json` |
| Sections 6.2–6.3; Fig. 5 (X1, X1b) | `experiments.py X1 X1b` | `results/X1.csv`, `X1b.csv` |
| Sections 6.4, 6.9; Figs. 6, 12 (X2) | `experiments.py X2` | `results/X2.csv` |
| Section 6.5; Fig. 7 (X3) | `experiments.py X3` | `results/X3.csv` |
| Section 6.6; Table 5; Fig. 8 (X4) | `experiments.py X4` | `results/X4.csv`, `T_drills.csv` |
| Section 6.7; Table 6; Fig. 9 (X5) | `experiments.py X5` | `results/X5.csv`, `T_sched.csv` |
| Section 6.8; Table 7 (X6) | `experiments.py X6` | `results/X6.csv`, `T_robust.csv` |
| Section 6.8; Fig. 11 (X7) | `experiments.py X7` | `results/X7.csv`, `X7_design.csv`, `T_global.csv` |
| Section 6.7; Fig. 10 (X9) | `experiments.py X9` | `results/X9.csv`, `X9_series.npy` |
| Section 6.9 (X10) | `experiments.py X10` | `results/X10.csv` |
| Section 6.2 (X11: p_crit, AR with halt time) | `exp3.py X11` | `results_ext/X11.csv` |
| Section 6.8 (X12: abandon mode, decision time) | `exp3.py X12` | `results_ext/X12.csv` |
| Section 6.9 (X13: AI safe stop) | `exp3.py X13` | `results_ext/X13.csv` |
| Section 6.8 (X14: behavioural parameters) | `exp3.py X14` | `results_ext/X14.csv`, `X14_design.csv`, `T_global_X14.csv` |
| All numbers quoted in Section 6 | `analysis.py`, `analysis_ext.py` | `results/summary2.json`, `results_ext/summary_ext.json` |

Experiment IDs follow the manuscript (X8 is not used).

## Reproducing the results

Python 3.13 with the pinned packages in `requirements.txt` (or `environment.yml`). Graphviz is needed only for Figs. 1–3.

```bash
pip install -r requirements.txt
make smoke        # ~3 min: mhc3 == mhc2 check and verification suite V1–V6
make all          # ~45 min on 2 cores: everything below
```

Individual steps:

```bash
python3 code/verify.py                                        # ~3 min
python3 code/experiments.py X1 X1b X2 X3 X4 X5 X6 X7 X9 X10   # ~30 min on 2 cores
python3 code/analysis.py                                      # tables, summary2.json, Figs. 4-12
python3 code/exp3.py CHECK X11 X12 X13 X14                    # ~8 min on 2 cores
python3 code/analysis_ext.py                                  # summary_ext.json
bash diagrams/render.sh                                       # Figs. 1-3
```

Running the scripts overwrites the files in `results/`, `results_ext/` and `figures/` with identical values; `git status` should show no change afterwards (figures may differ at byte level across matplotlib versions).

Tested on Linux with Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5 and matplotlib 3.11.2.

## Conventions

* **Seeds and common random numbers.** Replication *i* uses seed `1000 + i`, and the same seeds are used in every scenario, so contrasts are computed on matched differences.
* **Pre-drawn randomness.** All exogenous random quantities are drawn before a run, per exception and per supervisor. Changing a policy never changes the exception stream, which is what allows the event calendar to be checked case by case against the Kiefer–Wolfowitz recursion (V1).
* **Log-normal variables** are parameterised by mean *m* and coefficient of variation *cv*: `X = m·exp(σZ − σ²/2)`, `σ² = ln(1 + cv²)`.
* **Inference.** 95% t-intervals over replications, paired differences for contrasts, bootstrap (2,000 resamples) for the argmax, TOST with margin ±0.01 for equivalence, PRCC for global sensitivity.

## How to cite

Please cite the article (see [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button uses it). Until it is published:

```
Hosseini Bamakan, S. M., & Mirzabagheri, M. (2026). Separating Reliability from Human Command in
Dark Factories: A Queueing and Digital-Twin Model of AI Escalation [Code and data, v0.8.0].
https://github.com/smhbamakan/dark-factory-human-command
```

## License

Code (`code/`, `diagrams/`, `Makefile`): [MIT](LICENSE).
Data, results and figures (`results/`, `results_ext/`, `figures/`): [CC BY 4.0](DATA_LICENSE.md).

## Funding

This work was supported by the Marie Skłodowska-Curie OP JAK fellowship project "MSCAfellow7_MUNI" (registration number CZ.02.01.01/00/22_010/0008854) at Masaryk University.

## Contact

Seyed Mojtaba Hosseini Bamakan, Department of Business Management, Faculty of Economics and Administration, Masaryk University, Brno, Czech Republic — smhosseini@muni.cz
