from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# App configuration
# ============================================================
CURRENT_YEAR = 2026
DATA_FILE = Path(__file__).with_name("car details v4 (1).csv")

st.set_page_config(
    page_title="AutoValue | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained visual layer: the app remains usable if custom CSS is disabled.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
      .hero {
        padding: 1.45rem 1.7rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(115deg, #101f35 0%, #173b58 58%, #167d83 100%);
        color: #fff;
      }
      .hero h1 { color: white; font-size: 2rem; margin: 0 0 .35rem 0; }
      .hero p { color: #d9e8f0; margin: 0; font-size: .98rem; }
      .section-label { color: #16858a; font-weight: 700; letter-spacing: .08em;
                       text-transform: uppercase; font-size: .72rem; }
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.15);
        padding:  .85rem 1rem; border-radius: 14px;
      }
      .result-box {
        padding: 1.2rem 1.4rem; border-radius: 16px;
        border: 1px solid rgba(22,133,138,.35);
        background: rgba(22,133,138,.08);
      }
      .result-price { font-size: 2.1rem; font-weight: 700; color: #16858a; }
      .muted { color: #7b8794; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading and preparation
# ============================================================
@st.cache_data(show_spinner="Loading car dataset...")
def load_dataset(file_path: str, modified_time: float) -> pd.DataFrame:
    """Read the CSV. modified_time is part of the cache key."""
    df = pd.read_csv(file_path)
    return df


def extract_max_power(series: pd.Series) -> pd.Series:
    """Extract the first numeric value from strings such as '100 bhp'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def remove_iqr_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove rows outside 1.5 * IQR for the selected columns."""
    result = df.copy()
    for column in columns:
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result = result[result[column].between(lower, upper)]
    return result.copy()


def prepare_model_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Apply the same transformations used in the supplied notebook.
    Returns X, log(y), and the cleaned dataframe for EDA.
    """
    df = raw_df.copy()

    required = {"Price", "Year", "Max Power", "Length", "Width", "Kilometer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df["Max Power"] = extract_max_power(df["Max Power"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Kilometer"] = pd.to_numeric(df["Kilometer"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Width"] = pd.to_numeric(df["Width"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Keep the notebook's year-to-age transformation.
    df["Year"] = CURRENT_YEAR - df["Year"]

    # The notebook explicitly drops records with missing Max Power.
    df = df.dropna(subset=["Max Power"]).copy()

    df["Power Area Product"] = df["Length"] * df["Width"] * df["Max Power"]

    # Log transformations require strictly positive values.
    df = df.dropna(subset=["Price", "Kilometer", "Power Area Product"]).copy()
    df = df[
        (df["Price"] > 0)
        & (df["Kilometer"] >= 0)
        & (df["Power Area Product"] > 0)
    ].copy()

    # Preserve a pre-IQR copy for EDA so the boxplots can reveal the outliers.
    eda_df = df.copy()

    # IQR filtering as specified in the notebook; this filtered copy trains the model.
    df = remove_iqr_outliers(df, ["Kilometer", "Power Area Product"])

    y = np.log(df["Price"].astype(float))

    drop_cols = [
        "Price", "Width", "Length", "Max Power",
        "Max Torque", "Engine", "Model",
    ]
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    X["Kilometer"] = np.log1p(X["Kilometer"].astype(float))
    X["Power Area Product"] = np.log(X["Power Area Product"].astype(float))

    return X, y, eda_df


# ============================================================
# Model pipeline
# ============================================================
@st.cache_resource(show_spinner="Training the price prediction model...")
def train_model(X: pd.DataFrame, y: pd.Series):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)

    predicted_log = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, predicted_log),
        "mae_log": mean_absolute_error(y_test, predicted_log),
        "rmse_log": float(np.sqrt(mean_squared_error(y_test, predicted_log))),
        "test_size": len(X_test),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return model, metrics


def build_prediction_row(
    form_values: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create a one-row dataframe with exactly the model's expected columns."""
    row = dict(form_values)

    # Inputs are collected in original units; reproduce notebook transformations.
    if "Year" in row:
        row["Year"] = CURRENT_YEAR - float(row["Year"])
    if "Power Area Product" in feature_columns:
        row["Power Area Product"] = (
            float(row.pop("Length"))
            * float(row.pop("Width"))
            * float(row.pop("Max Power"))
        )
        row["Power Area Product"] = np.log(row["Power Area Product"])
    if "Kilometer" in row:
        row["Kilometer"] = np.log1p(float(row["Kilometer"]))

    # Remove raw engineering inputs if they are not model features.
    row = {key: value for key, value in row.items() if key in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def money(value: float) -> str:
    return f"{value:,.0f}"


# ============================================================
# Load and train
# ============================================================
st.markdown(
    """
    <div class="hero">
      <div class="section-label" style="color:#8ce0d8;">AUTOVALUE · MACHINE LEARNING</div>
      <h1>Car Price Intelligence</h1>
      <p>Explore the market data and estimate a car's price using a trained Linear Regression pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_FILE.exists():
    st.error(
        f"Dataset not found: `{DATA_FILE.name}`. "
        "Place the CSV file in the same folder as `app.py`, then restart the app."
    )
    st.stop()

try:
    raw_data = load_dataset(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    X, y, eda_data = prepare_model_data(raw_data)
    if X.empty:
        st.error("No usable rows remain after data cleaning. Please check the CSV values.")
        st.stop()
    model, model_metrics = train_model(X, y)
except Exception as exc:
    st.exception(exc)
    st.stop()


# ============================================================
# Sidebar prediction form
# ============================================================
st.sidebar.markdown("## 🚘 Configure a car")
st.sidebar.caption("Enter the vehicle details to estimate its market price.")

categorical_features = model_metrics["categorical_features"]
numeric_features = model_metrics["numeric_features"]

# Raw columns needed to calculate the engineered feature, although they are not
# passed directly to the model.
engineering_inputs = ["Length", "Width", "Max Power"]
form_values: dict[str, Any] = {}

with st.sidebar.form("prediction_form"):
    st.markdown("### Vehicle information")

    # The model uses car age, but the user enters the manufacturing year.
    if "Year" in X.columns:
        min_year = int(max(1900, min(CURRENT_YEAR, pd.to_numeric(raw_data["Year"], errors="coerce").min())))
        max_year = int(min(CURRENT_YEAR, pd.to_numeric(raw_data["Year"], errors="coerce").max()))
        if min_year > max_year:
            min_year, max_year = 1990, CURRENT_YEAR
        form_values["Year"] = st.number_input(
            "Manufacturing year",
            min_value=min_year,
            max_value=max_year,
            value=max(min_year, min(max_year, 2020)),
            step=1,
            help="The app converts this to vehicle age, matching the notebook.",
        )

    if "Kilometer" in X.columns:
        km_median = float(pd.to_numeric(raw_data["Kilometer"], errors="coerce").median())
        form_values["Kilometer"] = st.number_input(
            "Kilometers driven",
            min_value=0.0,
            value=max(0.0, km_median if np.isfinite(km_median) else 30000.0),
            step=1000.0,
        )

    # Collect categorical values using values observed in the dataset.
    for column in categorical_features:
        if column not in raw_data.columns:
            continue
        options = raw_data[column].dropna().astype(str).value_counts().index.tolist()
        if not options:
            options = ["Unknown"]
        form_values[column] = st.selectbox(column, options)

    # Collect remaining numeric model features, excluding derived fields above.
    handled = {"Year", "Kilometer", "Power Area Product"}
    for column in numeric_features:
        if column in handled:
            continue
        if column not in raw_data.columns:
            continue
        values = pd.to_numeric(raw_data[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        minimum = float(values.min()) if values.notna().any() else 0.0
        maximum = float(values.max()) if values.notna().any() else max(1.0, median * 2)
        step = 1.0 if column.lower() in {"seats", "owner"} else max((maximum - minimum) / 100, 0.1)
        form_values[column] = st.number_input(
            column,
            min_value=float(minimum),
            max_value=float(maximum) if maximum > minimum else float(minimum + 1),
            value=float(np.clip(median, minimum, maximum)) if maximum > minimum else float(minimum),
            step=float(step),
        )

    st.markdown("### Power-area details")
    for column in engineering_inputs:
        numeric_values = pd.to_numeric(
            extract_max_power(raw_data[column]) if column == "Max Power" else raw_data[column],
            errors="coerce",
        )
        default = float(numeric_values.median()) if numeric_values.notna().any() else 1.0
        minimum = 0.0 if column == "Max Power" else max(0.0, float(numeric_values.min()))
        form_values[column] = st.number_input(
            column,
            min_value=float(minimum),
            value=max(float(minimum), default),
            step=1.0,
            help="Used to calculate Power Area Product.",
        )

    submitted = st.form_submit_button(
        "✨ Predict Price",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# Main tabs
# ============================================================
prediction_tab, eda_tab = st.tabs(["💰 Price Prediction", "📊 Exploratory Data Analysis"])

with prediction_tab:
    st.markdown("### Model overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Linear Regression")
    c2.metric("R² score", f"{model_metrics['r2']:.4f}")
    c3.metric("Training records", f"{len(X) - model_metrics['test_size']:,}")
    c4.metric("Test records", f"{model_metrics['test_size']:,}")

    st.caption(
        "R² is calculated on the held-out test set using log-transformed prices, "
        "as in the notebook. The score is recalculated from the CSV when the data changes."
    )

    if submitted:
        try:
            # Add the engineered feature and transform inputs exactly as during training.
            prediction_input = build_prediction_row(form_values, X.columns.tolist())
            predicted_log_price = float(model.predict(prediction_input)[0])
            predicted_price = float(np.exp(predicted_log_price))

            if not np.isfinite(predicted_price):
                raise ValueError("The model returned a non-finite prediction.")

            st.markdown(
                f"""
                <div class="result-box">
                  <div class="section-label">ESTIMATED MARKET PRICE</div>
                  <div class="result-price">{money(predicted_price)}</div>
                  <div class="muted">Price estimate in the same currency/unit as the dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("Prediction completed successfully.")
            with st.expander("View submitted model features"):
                st.dataframe(prediction_input, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")
    else:
        st.info("Complete the vehicle details in the sidebar, then select **Predict Price**.")

    st.markdown("### Model performance")
    left, right = st.columns(2)
    with left:
        st.metric("MAE · log-price scale", f"{model_metrics['mae_log']:.4f}")
    with right:
        st.metric("RMSE · log-price scale", f"{model_metrics['rmse_log']:.4f}")

    st.caption(
        "These error metrics are reported on the log-price scale. "
        "They should not be interpreted as currency amounts."
    )


with eda_tab:
    st.markdown("### Dataset at a glance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw records", f"{len(raw_data):,}")
    m2.metric("Records after cleaning", f"{len(eda_data):,}")
    m3.metric("Columns", f"{eda_data.shape[1]:,}")

    st.markdown("### Price distribution")
    price_fig = px.histogram(
        eda_data,
        x="Price",
        nbins=35,
        marginal="box",
        color_discrete_sequence=["#16858a"],
        title="Distribution of car prices",
    )
    price_fig.update_layout(
        xaxis_title="Price",
        yaxis_title="Number of cars",
        bargap=0.08,
        template="plotly_white",
    )
    st.plotly_chart(price_fig, use_container_width=True)

    st.markdown("### Outlier overview")
    box_columns = ["Price", "Kilometer", "Power Area Product"]
    box_cols = st.columns(3)
    for container, column in zip(box_cols, box_columns):
        fig = px.box(
            eda_data,
            y=column,
            points="outliers",
            color_discrete_sequence=["#16858a"],
            title=column,
        )
        fig.update_layout(template="plotly_white", yaxis_title=column, showlegend=False)
        container.plotly_chart(fig, use_container_width=True)

    st.markdown("### Numerical feature correlations")
    numeric_eda = eda_data.select_dtypes(include=np.number)
    corr = numeric_eda.corr(numeric_only=True)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar_title="Correlation",
        )
    )
    heatmap.update_layout(
        title="Correlation matrix",
        template="plotly_white",
        height=max(520, 35 * len(corr.columns)),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(heatmap, use_container_width=True)

    with st.expander("Preview cleaned dataset"):
        st.dataframe(eda_data.head(100), use_container_width=True)

st.divider()
st.caption("AutoValue · Educational machine-learning demo · Built with Streamlit and scikit-learn")
