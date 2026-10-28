import json
from pathlib import Path
import numpy as np, pandas as pd, joblib, streamlit as st
from common import CLASSES, CAT, FEATURES, R2L_BOOST, engineer, build_matrix

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")

BASE = Path(__file__).resolve().parent   # always resolve files relative to app.py, not the working directory

def show_img(rel, caption=None):
    """Show a saved figure if it exists; otherwise render supported charts from results.json."""
    p = BASE / rel
    if p.exists():
        st.image(str(p), caption=caption)
        return True
    st.warning(f"Saved figure not found: {rel}. Showing a generated fallback where possible.")
    return False


def show_confusion_matrices():
    """Generate confusion matrices from results.json if the PNG was not deployed."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(axes, ["before", "after"], ["Original model", "Improved model"]):
        cm = np.asarray(R[key]["cm"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(ax=ax, values_format="d", xticks_rotation=45, colorbar=False)
        ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_before_after():
    """Generate per-class recall/F1 chart from results.json if the PNG was not deployed."""
    import matplotlib.pyplot as plt

    x = np.arange(len(CLASSES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ("Recall before", "before", "recall", -1.5 * width),
        ("Recall after", "after", "recall", -0.5 * width),
        ("F1 before", "before", "f1", 0.5 * width),
        ("F1 after", "after", "f1", 1.5 * width),
    ]
    for label, version, metric, offset in series:
        vals = [R[version][metric][c] for c in CLASSES]
        ax.bar(x + offset, vals, width, label=label)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class recall and F1 before and after")
    ax.legend(ncol=2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

@st.cache_resource
def load():
    b = joblib.load(BASE / "nsl_kdd_random_forest.joblib")
    return b["model"], b["r2l_model"], float(b["r2l_threshold"]), b["selected_columns"], pd.Index(b["dummy_cols"])
model, r2l_model, r2l_threshold, selected, dummy_cols = load()
R = json.load(open(BASE / "results.json"))

def predict(df):
    df = df[FEATURES].copy()
    for c in CAT: df[c] = df[c].astype(str).str.strip().str.lower()
    X = build_matrix(engineer(df), dummy_cols).reindex(columns=selected, fill_value=0)
    p = model.predict_proba(X)
    # Retain the existing R2L probability boost, then let the specialist add
    # R2L overrides when its independently tuned threshold is met.
    p = p * np.where(np.asarray(model.classes_) == "R2L", R2L_BOOST, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(p, columns=model.classes_)[CLASSES]
    base_pred = model.classes_[p.argmax(axis=1)]
    r2l_positive_index = list(r2l_model.classes_).index(1)
    r2l_prob = r2l_model.predict_proba(X)[:, r2l_positive_index]
    r2l_override = r2l_prob >= r2l_threshold
    pred = np.where(r2l_override, "R2L", base_pred)
    confidence = p.max(axis=1).copy()
    # For records overridden by the specialist, show its binary probability.
    confidence[r2l_override] = r2l_prob[r2l_override]
    out.insert(0, "Predicted class", pred)
    out.insert(1, "Confidence", confidence)
    out.insert(2, "R2L specialist probability", r2l_prob)
    return out

st.title("🛡️ Network Intrusion Detection")
st.caption("NSL-KDD Random Forest with engineered features, class balancing, an R2L probability boost, and a validation-tuned specialist")
t1, t2, t3, t4 = st.tabs(["Try a sample connection", "Upload connections (CSV)", "Model performance", "Features each class needs"])

with t1:
    sample = pd.read_csv(BASE / "sample_connections.csv")
    pick = st.selectbox("Actual class to browse", CLASSES, index=3)
    sub = sample[sample.actual_class == pick].reset_index(drop=True)
    i = st.selectbox("Sample connection", sub.index, format_func=lambda k: f"#{k}: {sub.protocol_type[k]} / {sub.service[k]} / {sub.flag[k]} (src_bytes={sub.src_bytes[k]})")
    row = sub.iloc[[i]]
    res = predict(row).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", res["Predicted class"]); c2.metric("Confidence", f"{res['Confidence']*100:.1f}%"); c3.metric("Actual class", pick)
    (st.success if res["Predicted class"] == pick else st.warning)("Correct prediction" if res["Predicted class"] == pick else "Incorrect prediction (R2L is still often missed; see Model performance)")
    st.bar_chart(res[CLASSES].astype(float))
    with st.expander("Show the 41 feature values"): st.dataframe(row[FEATURES].T.astype(str))

with t2:
    st.write("Upload a CSV with the 41 NSL-KDD feature columns (header row with the standard feature names).")
    st.download_button("Download a sample CSV", pd.read_csv(BASE / "sample_connections.csv").to_csv(index=False), "sample_connections.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}")
        else:
            out = pd.concat([predict(df), df.reset_index(drop=True)], axis=1)
            st.dataframe(out.head(500)); st.bar_chart(out["Predicted class"].value_counts())

with t3:
    A, B = R["after"], R["before"]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{A['accuracy']*100:.2f}%", f"{(A['accuracy']-B['accuracy'])*100:+.2f} pts")
    b.metric("Macro F1", f"{A['macro_f1']*100:.2f}%", f"{(A['macro_f1']-B['macro_f1'])*100:+.2f} pts")
    c.metric("Attacks detected", f"{A['attacks_detected']*100:.2f}%", f"{(A['attacks_detected']-B['attacks_detected'])*100:+.2f} pts")
    d.metric("Normal flagged as attack", f"{A['normal_flagged_as_attack']*100:.2f}%", f"{(A['normal_flagged_as_attack']-B['normal_flagged_as_attack'])*100:+.2f} pts", delta_color="inverse")
    st.table(pd.DataFrame({f"{m} {w}": [f"{R[w.lower()][k][cl]*100:.2f}%" for cl in CLASSES]
                           for k, m in [("recall", "Recall"), ("f1", "F1")] for w in ["Before", "After"]}, index=CLASSES))
    if not show_img("figures/confusion_matrices.png", caption="Confusion matrices on KDDTest+"):
        show_confusion_matrices()
    if not show_img("figures/per_class_before_after.png", caption="Per-class recall and F1 before and after"):
        show_before_after()
    bp = R["best_params"]
    st.info(f"Grid search (macro-F1, 3-fold): rare-class target {int(bp['rare_n'])}, large-class cap {int(bp['max_n'])}, min_samples_leaf {int(bp['min_samples_leaf'])}. "
            f"R2L probability boost: {R2L_BOOST:g}x; specialist threshold: {r2l_threshold:.2f}. R2L/U2R remain difficult, especially for attack types absent from training.")

with t4:
    imp_path = BASE / "figures" / "per_class_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)
        missing_classes = [c for c in CLASSES if c not in imp.columns]
        if missing_classes:
            st.error(f"The feature-importance CSV is missing class columns: {missing_classes}. Please upload the complete figures/per_class_importance.csv generated by train_model.py.")
        else:
            cls = st.selectbox("Class", CLASSES, index=3)
            st.subheader(f"Most important features for {cls}")
            st.bar_chart(imp[cls].nlargest(12))
            show_img("figures/per_class_feature_importance.png", caption="Top features per class (one-vs-rest forests)")
            st.dataframe(imp.loc[imp.max(axis=1).nlargest(25).index].round(4))
    else:
        st.error("Missing figures/per_class_importance.csv. Upload this file from the project's figures/ folder to GitHub. It contains the class-specific importance values used by this tab.")
        st.info("A generic importance ranking from the main five-class model would not be equivalent to the per-class one-vs-rest analysis reported in your paper, so this tab does not substitute a misleading chart.")
