# Business memo - using a default-probability model for decision support

> Illustrative memo for a portfolio project. Numbers must come from `artifacts/policy_comparison.csv` and `artifacts/threshold_sensitivity.csv` after real-data training; none are claimed here.

## Question
Given a calibrated probability of default (PD) for each applicant, how should an organisation route applications, and what does it trade off?

## What the model provides
A calibrated PD, an illustrative risk band (Low < 5%, Medium < 15%, High otherwise - configurable placeholders), and the main model-attributed drivers for review. It does **not** provide an approval decision.

## Policy scenarios (configurable)
| Policy | Behaviour | Typical effect |
|---|---|---|
| Conservative | low flag threshold | catches more defaulters (higher recall) but flags many good applicants (more false positives, lost business) |
| Balanced | middle threshold | compromise |
| Higher acceptance | high flag threshold | fewer good applicants flagged, but more defaulters pass (more false negatives, higher credit losses) |

The right point depends on loss given default, margin, funding cost, capacity for manual review and regulatory/fair-treatment constraints. The default cost ratio (5 : 1 FN to FP) is an arbitrary placeholder; replace it with finance estimates before reading anything into the cost curve.

## Suggested use of outputs
1. Use PD for ranking and for thresholds; check calibration on recent data before using PD as an absolute estimate.
2. Route "High Risk" to manual review or additional verification rather than automatic decline.
3. Review the share of applicants and the observed default rate per tier regularly (`risk_tiers.csv`).
4. Track false-positive cost (declined/reviewed good customers) alongside credit losses.

## Risks
Selection bias (only past approved loans), proxies of sensitive attributes, population drift, delayed labels, and over-reliance on a single score. See `reports/limitations.md`.

## Monitoring and retraining
Watch input drift, score drift and, as labels mature, performance. Drift triggers investigation; retraining is recommended only with labelled evidence of degradation and always needs human approval.
