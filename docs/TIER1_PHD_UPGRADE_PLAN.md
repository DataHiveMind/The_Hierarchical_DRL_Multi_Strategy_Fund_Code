# Tier-1 Hedge Fund & PhD Upgrade Plan (6 Weeks)

## Objective

Elevate this project from strong prototype to conference-grade quantitative research artifact with reproducible, leakage-safe, statistically defensible claims.

## Success Criteria

1. Single source of truth for experiment assumptions and split dates.
2. No doc-code-report inconsistencies for algorithms, data windows, and metrics.
3. Leakage-safe evaluation with purged walk-forward and embargo.
4. Statistical significance reporting (confidence intervals, paired tests).
5. Clear narrative for methodology, ablations, and failure modes.

---

## Week 1 — Research Governance & Reproducibility (P0)

- Lock canonical experiment spec in `configs/experiment_spec.yaml`.
- Add consistency gate script (`scripts/validate_research_consistency.py`).
- Standardize run seeds and deterministic switches across notebooks and scripts.
- Deliverable: passing consistency check and reproducibility checklist.

## Week 2 — Leakage-Safe Validation Layer (P1)

- Integrate `PurgedWalkForwardSplitter` into specialist and master evaluation loops.
- Add purge and embargo sensitivity analysis.
- Add strict chronological assertions in backtest pipelines.
- Deliverable: walk-forward result tables with fold-level metrics.

## Week 3 — Market Realism & Execution Modeling (P1)

- Replace static slippage/cost assumptions with volatility- and turnover-dependent model.
- Add capacity proxy (ADV participation limits, impact penalty).
- Add latency and quote-staleness stress tests.
- Deliverable: friction stress plots and robustness report.

## Week 4 — RL Research Depth (P1/P2)

- Run structured ablations:
  - reward terms,
  - action space parameterization,
  - observation blocks,
  - entropy / exploration schedules.
- Add stronger baselines (risk parity variants, contextual bandit, non-RL hierarchical allocator).
- Deliverable: ablation matrix and baseline comparison with significance.

## Week 5 — Statistical Rigor & Uncertainty Quantification (P1)

- Add bootstrap CIs for Sharpe/Sortino/Calmar and drawdown statistics.
- Add paired comparisons and multiple-testing correction where relevant.
- Add regime-conditional attribution (bull/bear/high-vol) and turnover attribution.
- Deliverable: confidence intervals and hypothesis test appendix.

## Week 6 — Conference Packaging (P0/P2)

- Consolidate methodology and experiment protocol into a paper-style technical report.
- Auto-generate README metrics and final report from canonical artifact.
- Add explicit limitations/failure modes and reproducibility instructions.
- Deliverable: conference-ready deck + reproducibility bundle.

---

## Immediate Commands

Run consistency gate:

```bash
python scripts/validate_research_consistency.py
```

Run leakage-split tests:

```bash
pytest tests/test_validation_splitter.py -q
```

---

## Notes for Presentation Quality

- Always present uncertainty bars with headline metrics.
- Avoid single backtest narratives; show fold distribution and regime breakdown.
- Explicitly disclose assumptions (data source, costs, leverage limits, rebalance cadence).
- Include one slide titled "Failure Modes & Where Model Breaks".
