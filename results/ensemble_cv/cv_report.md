# Ensemble Cross-Validation Results

**Generated:** 2026-01-28 05:21:04

## Summary

- **Mean F1 Score:** 0.7577 ± 0.0285
- **Voting Strategy:** Hierarchical (SVC=1, LR=1, [HGB+MNB+CNB]=1)
- **Number of Folds:** 5
- **Data Sorting:** Strict Timestamp

## Fold Scores

| Fold | F1 Score |
|------|----------|
| 1 | 0.7023 |
| 2 | 0.7641 |
| 3 | 0.7660 |
| 4 | 0.7834 |
| 5 | 0.7730 |

## Configuration

- **Preprocessing:** DatasetCleaner + DatasetDeduplicator + FeatureExtractor + TimeExtractor + **TimeSort**
- **Structure:**
  - **Tier 1 (Final Vote):** LinearSVC, LogisticRegression, WeakConsensus
  - **Tier 2 (WeakConsensus):** HistGradientBoosting, MultinomialNB, ComplementNB
- **Feature Distribution:**
  - HGB: Text + Density + Time + PageRank + Source
  - SVC/LR: Text + Density + PageRank + Source (No Time)
  - MNB/CNB: Text + Source (No Time, No PageRank, No Density)

## Notes

Results saved to `results/ensemble_cv/fold_scores.json`.
