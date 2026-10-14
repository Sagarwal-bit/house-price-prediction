"""Streamlit UI for the house price estimator.

Run:  streamlit run app.py
The model is trained separately (`python -m src.train`) and only loaded here. The input
form is generated from the trained model's own feature list, so it always matches the model.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.evaluation import plot_actual_vs_predicted, plot_bars, plot_residuals, r2_interpretation
from src.predict import ModelNotAvailableError, load_predictor, load_test_predictions

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return load_predictor("models")


@st.cache_data(show_spinner=False)
def get_test_predictions():
    return load_test_predictions("models")


def usd(x: float) -> str:
    return f"${x:,.0f}"


def _num_args(f: dict) -> dict:
    """Streamlit number_input arguments from the training range of one feature."""
    lo, hi, med = f["train_min"], f["train_max"], f["train_median"]
    if f["integer"]:
        return dict(min_value=int(math.floor(lo)), max_value=int(math.ceil(hi)), value=int(round(med)), step=1)
    rng = max(hi - lo, 1e-9)
    step = max(10 ** math.floor(math.log10(rng / 100)), 0.001)
    return dict(min_value=float(lo), max_value=float(hi), value=float(med), step=float(step))


st.title("House Price Estimator")

try:
    predictor = get_predictor()
except ModelNotAvailableError as exc:
    st.error(f"Model not available: {exc}")
    st.code("python -m src.train", language="bash")
    st.stop()

M = predictor.metrics
ds, selected = M["dataset"], predictor.model_name
test = M["models"][selected]["test"]

if ds["is_demo"]:
    st.warning(
        "**Demo mode:** this model was trained on *synthetic* data generated from a formula. "
        "Its metrics say nothing about real-world accuracy and its output is not a property valuation."
    )
st.caption(
    f"Dataset: **{ds['title']}**. Estimates are statistical predictions for educational purposes, "
    "not certified property valuations."
)

tab_pred, tab_perf, tab_cmp, tab_feat, tab_about = st.tabs(
    ["Estimate price", "Model performance", "Model comparison", "Feature analysis", "About & limitations"]
)

# ------------------------------------------------------------------ predict
with tab_pred:
    st.subheader("Describe the property")
    st.caption(ds["notes"])
    with st.form("property_form"):
        cols = st.columns(2)
        values = {}
        for i, f in enumerate(predictor.features):
            with cols[i % 2]:
                values[f["name"]] = st.number_input(f["label"], help=f["help"], key=f"in_{f['name']}", **_num_args(f))
        submitted = st.form_submit_button("Estimate price", type="primary")

    if submitted:
        try:
            est = predictor.predict(values)
        except ValueError as exc:
            st.warning(str(exc))
        except Exception:
            st.error("Something went wrong while estimating the price. Please try again.")
        else:
            st.success("Estimate ready")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Estimated price ({ds['currency']})", usd(est.price))
            c2.metric("Typical error (MAE, test set)", usd(test["mae"]))
            c3.metric("Model", selected)
            st.caption(
                f"Target: {ds['target_description']}. The typical error is the *average* miss on held-out data; "
                "individual estimates can be much further off."
            )
            for w in est.warnings:
                st.warning(w)

# -------------------------------------------------------------- performance
with tab_perf:
    st.subheader(f"Selected model: {selected}")
    st.caption(f"Computed on a held-out test set of {test['n']:,} rows that was not used for training or model selection.")
    a, b, c = st.columns(3)
    a.metric("MAE", usd(test["mae"]), help="Mean absolute error: average size of the miss.")
    b.metric("RMSE", usd(test["rmse"]), help="Root mean squared error: like MAE but penalises big misses more.")
    c.metric("R²", f"{test['r2']:.3f}", help="Share of price variation explained; 0 = no better than predicting the average.")
    st.info(r2_interpretation(test["r2"]))
    preds = get_test_predictions()
    left, right = st.columns([2, 3])
    with left:
        st.pyplot(plot_actual_vs_predicted(preds["actual"], preds["predicted"]))
    with right:
        st.pyplot(plot_residuals(preds["actual"], preds["predicted"]))
    rep = ds["report"]
    if rep["share_at_target_max"] > 0.01:
        st.caption(
            f"{rep['share_at_target_max'] * 100:.1f}% of rows sit exactly at the maximum target value "
            f"({usd(rep['target_max'])}); the data is capped there, so the model cannot learn higher prices."
        )

# --------------------------------------------------------------- comparison
with tab_cmp:
    st.subheader("Model comparison")
    rows = []
    for name, r in M["models"].items():
        rows.append({"Model": name, "CV RMSE (train)": r["cv"]["rmse_mean"], "Test MAE": r["test"]["mae"],
                     "Test RMSE": r["test"]["rmse"], "Test R²": r["test"]["r2"]})
    comp = pd.DataFrame(rows)
    st.dataframe(comp.style.format({"CV RMSE (train)": "{:,.0f}", "Test MAE": "{:,.0f}",
                                    "Test RMSE": "{:,.0f}", "Test R²": "{:.3f}"}), hide_index=True)
    st.bar_chart(comp.set_index("Model")[["Test MAE", "Test RMSE"]])
    st.info(
        f"**{selected}** was selected by the lowest cross-validated RMSE on the training set "
        "(the mean baseline is only a reference). Test numbers are reported, not used for selection."
    )

# ------------------------------------------------------------------ features
with tab_feat:
    st.subheader("Feature analysis")
    st.markdown("Two *different* analyses are shown. They answer different questions and should not be compared directly.")
    col1, col2 = st.columns(2)
    labels = {f["name"]: f["label"] for f in predictor.features}
    with col1:
        st.markdown("**Linear Regression coefficients**")
        coefs = M["linear_analysis"]["coefficients"]
        st.pyplot(plot_bars([labels[k] for k in coefs], list(coefs.values()), "Standardised coefficients", "USD per +1 std dev"))
        st.caption(M["linear_analysis"]["note"] + " Correlated features share effects, so read signs with care.")
    with col2:
        imp = M["importance_analysis"]
        st.markdown(f"**Permutation importance - {imp['model']}**")
        st.pyplot(plot_bars([labels[k] for k in imp["rmse_increase_mean"]], list(imp["rmse_increase_mean"].values()),
                            "Permutation importance", "RMSE increase (USD)"))
        st.caption(imp["note"] + " It shows how much the model relies on a feature, not the direction of the effect.")

# -------------------------------------------------------------------- about
with tab_about:
    st.subheader("About")
    rep = ds["report"]
    st.markdown(
        f"""
**Dataset:** {ds['title']}. Target: {ds['target_description']}.
{rep['n_rows_raw']:,} raw rows, {rep['n_rows_used']:,} used after cleaning
({rep['n_duplicate_rows_removed']:,} duplicates and {rep['n_missing_target_removed'] + rep['n_nonpositive_target_removed']:,}
invalid-target rows removed). Split: {M['split']['n_train']:,} train / {M['split']['n_test']:,} test (seed {M['split']['seed']}).

**Method:** imputation (+ scaling for the linear model) -> regressor, all inside one scikit-learn pipeline fitted on training data only.
Models compared with {M['split']['cv_folds']}-fold cross-validation, then evaluated once on the held-out test set.

**Limitations**
- Not a certified valuation; do not use for buying, selling, lending or tax decisions.
- Training data reflects its own time and place (see dataset above); prices elsewhere or today will differ.
- The model only knows the listed features. It does not see condition, renovations, schools or the exact property.
- Inputs outside the training range produce unreliable extrapolations.
        """
    )
    st.caption(f"scikit-learn {M['environment']['scikit_learn']} - trained {M['created_utc']}")
