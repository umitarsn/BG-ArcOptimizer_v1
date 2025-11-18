import io
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="BG-EAF Arc Optimizer",
    layout="wide",
)

@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def split_features_target(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y, feature_cols

def train_rf_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return {"model": model, "metrics": metrics}

def ensure_session_keys():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None
    if "feature_cols" not in st.session_state:
        st.session_state["feature_cols"] = None
    if "target_col" not in st.session_state:
        st.session_state["target_col"] = None

def set_model(model, feature_cols, target_col, metrics):
    st.session_state["trained_model"] = model
    st.session_state["feature_cols"] = feature_cols
    st.session_state["target_col"] = target_col
    st.session_state["model_metrics"] = metrics

def get_model():
    return (
        st.session_state.get("trained_model"),
        st.session_state.get("feature_cols"),
        st.session_state.get("target_col"),
        st.session_state.get("model_metrics"),
    )


ensure_session_keys()

st.sidebar.title("BG-EAF Arc Optimizer")
page = st.sidebar.radio(
    "Menü",
    (
        "1) Veri Yükleme & Keşif",
        "2) Model Eğitimi",
        "3) Tek Heat Tahmini",
        "4) Batch Tahmin (CSV)",
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "EAF verisi üzerinden enerji / süre / kalite tahmini için demo Arc Optimizer."
)

if page.startswith("1"):
    st.title("1) Veri Yükleme & Keşif")
    uploaded = st.file_uploader("CSV yükle", type=["csv"])
    if uploaded:
        df = load_csv(uploaded.getvalue())
        st.session_state["raw_df"] = df
        st.success(f"{uploaded.name} yüklendi ({df.shape[0]} satır).")
        st.subheader("İlk 50 satır")
        st.dataframe(df.head(50))
        st.subheader("İstatistikler")
        st.dataframe(df.describe().transpose())
    else:
        st.info("CSV yükle.")

elif page.startswith("2"):
    st.title("2) Model Eğitimi")
    df = st.session_state.get("raw_df")
    if df is None:
        st.warning("Önce CSV yükle.")
    else:
        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if not numeric_cols:
            st.error("Sayısal kolon yok.")
        else:
            target_col = st.selectbox("Hedef kolon:", numeric_cols)
            feature_cols = st.multiselect(
                "Feature kolonlar:", numeric_cols, default=[c for c in numeric_cols if c != target_col]
            )
            test_size = st.slider("Test oranı", 0.1, 0.4, 0.2)
            n_estimators = st.slider("Ağaç sayısı", 50, 500, 200)

            if feature_cols and st.button("Modeli Eğit"):
                model_df = df[feature_cols + [target_col]].dropna()
                X, y, _ = split_features_target(model_df, target_col)
                result = train_rf_model(X, y, test_size, n_estimators=n_estimators)
                set_model(result["model"], feature_cols, target_col, result["metrics"])
                st.success("Model eğitildi.")
                st.subheader("Metrikler")
                st.json(result["metrics"])
            elif not feature_cols:
                st.error("En az bir feature seç.")

elif page.startswith("3"):
    st.title("3) Tek Heat Tahmini")
    model, feature_cols, target_col, metrics = get_model()
    if model is None:
        st.warning("Model yok. Önce eğit.")
    else:
        vals = []
        for f in feature_cols:
            vals.append(st.number_input(f, 0.0))
        if st.button("Tahmini Hesapla"):
            pred = float(model.predict(np.array(vals).reshape(1,-1))[0])
            st.success(f"{target_col} tahmini: {pred:.3f}")

elif page.startswith("4"):
    st.title("4) Batch Tahmin")
    model, feature_cols, target_col, metrics = get_model()
    if model is None:
        st.warning("Önce model eğit.")
    else:
        uploaded = st.file_uploader("Batch CSV yükle", type=["csv"])
        if uploaded:
            df = load_csv(uploaded.getvalue())
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error("Eksik kolonlar: " + ", ".join(missing))
            else:
                preds = model.predict(df[feature_cols].values)
                df[f"pred_{target_col}"] = preds
                st.subheader("Sonuç (ilk 50)")
                st.dataframe(df.head(50))
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button("Sonuçları indir", buf.getvalue(), "batch_predictions.csv")
