# Model card - House Price Estimator

**Task:** regression - estimate median house value (USD) of a California census block group.
**Candidates:** Linear Regression (scaled), Random Forest, Hist Gradient Boosting; mean baseline for reference. Selected by lowest 5-fold CV RMSE on the training split.
**Data:** California Housing (1990 census). Row counts/quality report in `models/metrics.json`.
**Evaluation:** held-out 20% test split; MAE, RMSE, R² in USD; actual-vs-predicted and residual plots in the app.

**Intended use:** learning / portfolio demonstration.
**Not intended for:** property valuation, lending, pricing, tax or any decision affecting people.

**Known issues:** 1990 prices; district-level rows; capped target (~$500k); outliers retained; permutation importance and linear coefficients answer different questions.
**Demo mode:** `--dataset synthetic` trains on formula-generated data and is labelled demo in the app.
**Reproducibility:** `python -m src.train --seed 42`; seed/split/CV settings stored in `metrics.json`.
**Security:** the model is a joblib file; load only files you generated yourself.
