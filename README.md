# STRATA — Real-time statistical supervision of an AI trading agent

> **[Coherence note, 2026-06-17]** The thesis case study is **SMCI** (a fair benchmark, B&H ≈ 0.484), where a
> deployable walk-forward meta-learner beats the agent, the rule and buy&hold nominally in directional accuracy
> (0.552 vs 0.484). SPY is the *mechanism* case (where the rescue is statistically significant). Canonical
> one-page summary: **`memoria/MANUAL.md`**.

[![CI](https://github.com/RaquelGarciah/Real-time-statistical-supervision-of-an-AI-trading-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/RaquelGarciah/Real-time-statistical-supervision-of-an-AI-trading-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active%20research-success.svg)

> **STRATA** (*Statistical Trading Real-time Audit*) is a **statistical supervision layer** that
> audits and corrects, decision by decision, what an LLM-based trading agent does. It turns a
> money-losing black-box agent into a **disciplined, interpretable, statistically validated**
> system — without becoming another black box.

Bachelor's Thesis ·  Degree in **Mathematics and Data Science**, Complutense University
of Madrid · Author: **Raquel García**.

---

## The problem

LLM-based trading agents are marketed as the new frontier of automated investing, but they are
**unreliable black boxes**. The agent we study — *AI Hedge Fund*, an open-source system with five
investor personalities (Buffett, Wood, Druckenmiller, Burry, Ackman) — trading the **SPY** ETF, out
of sample (Oct 2024 – Jun 2026, 401 sessions):

- **Loses money**: €1,000 → **€903**.
- **Predicts market direction less than 50 % of the time**: 38.4 % of days (*sign test* p < 0.001).
  Worse than a coin flip.

The research question: **how do you make a losing LLM agent usable — without replacing one black box
with another?**

---

## The solution

STRATA **does not predict the market**. It is a **deterministic function** that sits between the
agent and the market and uses only information available *today*:

```
f : (agent decision,  market state today)  ⟶  supervised position  w ∈ [−1, +1]
```

Three **classical, orthogonal statistical detectors** audit every daily decision:

| Detector | Axis it watches | Underlying model | Question |
|---|---|---|---|
| **RAM** | Market regime | 3-state Gaussian HMM | Is the agent's direction coherent with the regime (calm/stress/crisis)? |
| **PSA** | Agent consistency | BOCPD (Adams & MacKay, 2007) | Did the agent just change its mind anomalously? |
| **GSO** | Market volatility | GARCH(1,1)-t | Is the bet size compatible with current volatility? |

When the regime confidently contradicts the agent, STRATA **flips** the position toward the
regime's direction, sized by volatility. Three intervention modes (`warn` / `reduce` / `override`)
span from passive logging to active correction.

---

## Key results

On SPY, out of sample (Oct 2024 – Jun 2026, 401 sessions), under **strictly causal evaluation**
(today's position earns tomorrow's return, `signal_lag = 1`):

| Strategy | Directional accuracy | Sharpe | €1,000 → |
|---|:--:|:--:|:--:|
| LLM agent alone (unsupervised) | 38.4 % | −1.82 | €903 |
| **STRATA (statistical supervision)** | **43.6 %** | **+0.67** | **€1,069** |
| XGBoost meta-learner (ML benchmark) | 53.9 % | +0.64 | €1,035 |
| Buy & Hold (passive market) | 56.9 % | +1.09 | €1,317 |

**Three findings, each backed by a statistical test:**

1. **STRATA rescues the agent.** Directional accuracy rises from 38.4 % to 43.6 % and the account
   goes from losing to recovering. The improvement is **significant in the paired test that matters**:
   *McNemar* STRATA vs agent, **p ≈ 0.07** (of the 121 days where they differ, STRATA fixes 71 and
   breaks 50). Honestly stated: significant at α = 0.10, *borderline* at α = 0.05.

2. **An "everything-in" meta-learner does not beat it.** An XGBoost validated with *Combinatorial
   Purged CV* on 22 features (the 5 personalities + the 3 detectors + 4 regime features) **matches**
   the hand-built rule but does not surpass it (*Diebold-Mariano* p = 0.61). And **SHAP** shows the
   informative features are exactly STRATA's and the regime's — not the agent's: **the ML
   rediscovers the rule rather than improving on it.**

3. **Scientific honesty.** No system beats the passive market (Buy & Hold, €1,317). The contribution
   is **not "beating the market"**: it is **rescuing a losing agent with a defensible statistical
   protocol.**

---

## The contribution — why this project matters

- **Rigor over a pretty equity curve.** No figure is reported without its test: paired contrasts
  (*McNemar*, *Diebold-Mariano*), *Deflated Sharpe Ratio*, stationary *bootstrap*, CPCV validation
  **with no temporal leakage**, and **pre-registration** of every experiment (hypothesis and success
  criterion fixed *before* looking at results) as a shield against *p-hacking*.

- **Interpretability over black box.** STRATA is built from classical statistics (HMM, GARCH, BOCPD)
  that can be **explained and defended** before a committee — not an opaque model. Every
  intervention is traceable step by step.

- **Statistical discipline > ML complexity.** The central result — that a well-founded hand-built
  rule is *statistically indistinguishable* from a universal XGBoost — is a counterintuitive,
  valuable lesson: in a problem with weak signal and a small sample, **complexity buys no edge;
  discipline does.**

- **Falsifiable science.** The project explicitly documents **when it does NOT work** (the
  *prior-flip* rule: if the calibrated sign of the regime disagrees with the out-of-sample sign, it
  is reported as a failure). Reporting the limits is part of the result.

---

## How it works — one concrete day

The backtest engine is pure accounting: `P&L = position · next-day return`. The only thing that
changes between strategies is **how the position is computed**. An intervention day:

| Step | Computation | Result |
|---|---|---|
| 1. Agent decides | long, *size* = +0.30 | agent tuple |
| 2. HMM + GARCH | regime = **Crisis** (P = 0.80), σ = 23 % | market state |
| 3. RAM flags mismatch | long in Crisis ⇒ score 0.80 (*high*) | triggers |
| 4. *override* toward regime | regime_sign · vol_band = −1 · 0.43 | **position = −0.43** |

The agent wanted to buy in the middle of a crisis; STRATA reorients it to short. Over 401 days, that
kind of correction is what turns the loss into a recovery.

---

## Tech stack

**Models:** Gaussian HMM (regimes) · GARCH(1,1)-Student-t (volatility) · BOCPD (change points) ·
XGBoost + SHAP (ML benchmark).
**Inference:** McNemar · Diebold-Mariano · *sign test* · Deflated Sharpe · stationary *bootstrap*
(Politis-Romano) · Combinatorial Purged CV (López de Prado).
**Engineering:** Python 3.11, `numpy` / `pandas` / `scipy` / `scikit-learn` / `hmmlearn` / `arch` /
`xgboost`; tests with `pytest`; CI on GitHub Actions; reproducibility via fixed random seed.

---

## Repository layout

```
core/         Tested mathematical primitives (HMM, GARCH, BOCPD, CPCV, metrics, statistical tests)
strata/       The three detectors (RAM/PSA/GSO) + the intervention layer
experiments/  Reproducible experiments, each with its pre-registration and JSON output
notebooks/    Canonical thesis notebook (strata_canonical) + experiments notebook
tests/        Test suite (includes a look-ahead / leakage check)
cache/        Calibrated models (HMM/GARCH/thresholds) and the agent's per-asset decisions
BITACORA.md   Lab notebook: methodological decisions, findings and experiment pre-registrations
```

---

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # test suite (incl. no-leakage check)
jupyter notebook notebooks/strata_canonical.ipynb   # full thesis analysis
```

Model calibration: 2000–2024 (once). Out-of-sample evaluation: Oct 2024 onward, starting after the
LLM's knowledge cutoff to rule out look-ahead contamination.

---

## Scope and limitations (stated)

- **Central case: SPY.** It works because in aggregate indices the *leverage effect* (Black 1976;
  Christie 1982) makes high volatility coincide with drawdowns, so the regime acts as a directional
  *proxy*. The assumption weakens for individual stocks — a documented limitation, with a 10-asset
  robustness panel as an appendix.
- **A single out-of-sample window** (bullish). Multi-window / walk-forward validation is ongoing
  work.
- **It does not beat the passive market.** The goal is to supervise the agent, not to beat Buy &
  Hold.

---

*STRATA — Statistical Trading Real-time Audit. Raquel García, Complutense University of Madrid.*
