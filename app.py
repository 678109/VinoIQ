import json
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "wine_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "winequality-red.csv"
PREDICTIONS_PATH = ROOT / "models" / "test_predictions.csv"

st.set_page_config(
    page_title="VinoIQ | Wine Quality Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

BG = "#100D13"
SURFACE = "#1E1722"
BORDER = "#493044"
WINE = "#963E60"
GOLD = "#E9C48D"
TEXT = "#F7EFE7"
MUTED = "#C8B9C3"
BLUE = "#83C7F7"

st.markdown(
    """
    <style>
    :root {
        --background: #100D13;
        --surface: #1E1722;
        --surface-light: #2A1D2B;
        --border: #493044;
        --wine: #963E60;
        --gold: #E9C48D;
        --text: #F7EFE7;
        --muted: #C8B9C3;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
    }

    h1 {
        font-family: Georgia, serif;
        font-size: 3rem !important;
    }

    h2 {
        font-family: Georgia, serif;
    }

    p {
        color: var(--muted);
    }

    section[data-testid="stSidebar"] {
        background: #19121C;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] h1 {
        color: var(--gold) !important;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #84324F, #A64B6C);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background: #B95678;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--gold);
    }

    .hero-card {
        background: linear-gradient(135deg, #251727, #422337);
        border: 1px solid #70445C;
        border-radius: 24px;
        padding: 48px;
        margin-bottom: 30px;
    }

    .hero-label {
        color: var(--gold);
        font-size: 0.78rem;
        letter-spacing: 3px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    .hero-title {
        font-family: Georgia, serif;
        font-size: clamp(2.1rem, 4vw, 3.8rem);
        line-height: 1.15;
        color: #FFF1E4;
        margin-bottom: 22px;
    }

    .hero-description {
        color: #E0CDD6;
        font-size: 1.08rem;
        line-height: 1.8;
        max-width: 700px;
    }

    .section-label {
        color: var(--gold);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 14px;
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 28px;
        min-height: 180px;
        margin-bottom: 15px;
    }

    .info-card h3 {
        color: var(--gold) !important;
        margin-top: 0;
        font-size: 1.35rem;
    }

    .info-card p {
        line-height: 1.7;
    }

    .result-card {
        background: linear-gradient(135deg, #321B2F, #572B43);
        border: 1px solid #A56B85;
        border-radius: 24px;
        padding: 42px 25px;
        text-align: center;
        margin: 28px 0;
    }

    .result-label {
        color: var(--gold);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }

    .result-score {
        font-family: Georgia, serif;
        color: #FFF2E5;
        font-size: clamp(3.5rem, 7vw, 5.5rem);
        font-weight: 700;
        line-height: 1.2;
    }

    .result-max {
        font-size: 1.8rem;
        color: #D9B5C5;
        font-weight: 400;
    }

    .result-description {
        color: #EADAE2;
        font-size: 1.08rem;
        margin-top: 15px;
    }

    .quality-track {
        width: 100%;
        height: 16px;
        background: #352735;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid #60465A;
        margin-top: 15px;
    }

    .quality-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #84324F,
            #C06B83,
            #E9C48D
        );
        border-radius: 20px;
    }

    .quality-labels {
        display: flex;
        justify-content: space-between;
        color: #BDAAB6;
        font-size: 0.8rem;
        margin-top: 10px;
    }

    .note-card {
        background: #201A27;
        border-left: 3px solid var(--gold);
        border-radius: 8px;
        padding: 18px 22px;
        color: #D8C8D1;
        line-height: 1.7;
        margin: 20px 0;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 700px) {
        .hero-card {
            padding: 25px;
        }

        .result-card {
            padding: 30px 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html):
    """
    Remove indentation before rendering HTML.

    This prevents Streamlit Markdown from displaying
    indented HTML as a code block.
    """
    cleaned = textwrap.dedent(html).strip()

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_data
def load_test_predictions():
    if not PREDICTIONS_PATH.exists():
        return None

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "actual",
        "predicted",
        "absolute_error",
    }

    if not required_columns.issubset(predictions.columns):
        return None

    return predictions


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "Model files are missing. Run "
        "`python -m src.train_model` first."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "The dataset is missing. Run the training script first."
    )
    st.stop()


bundle = load_model()

model = bundle["model"]
features = bundle["features"]

metrics = load_metrics()
df = load_dataset()

test_predictions = load_test_predictions()


# ============================================================
# FEATURE INFORMATION
# ============================================================

FEATURE_INFO = {
    "fixed acidity": (
        "Fixed Acidity",
        "Non-volatile acids in the wine.",
    ),
    "volatile acidity": (
        "Volatile Acidity",
        "Volatile acids that may influence aroma and flavor.",
    ),
    "citric acid": (
        "Citric Acid",
        "The amount of citric acid in the wine.",
    ),
    "residual sugar": (
        "Residual Sugar",
        "Sugar remaining after fermentation.",
    ),
    "chlorides": (
        "Chlorides",
        "The chloride content of the wine.",
    ),
    "free sulfur dioxide": (
        "Free Sulfur Dioxide",
        "Free sulfur dioxide that helps preserve the wine.",
    ),
    "total sulfur dioxide": (
        "Total Sulfur Dioxide",
        "The total amount of sulfur dioxide.",
    ),
    "density": (
        "Density",
        "The density of the wine.",
    ),
    "pH": (
        "pH",
        "A measure of acidity.",
    ),
    "sulphates": (
        "Sulphates",
        "A sulphate-related chemical measurement.",
    ),
    "alcohol": (
        "Alcohol",
        "The alcohol content of the wine.",
    ),
}


# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:

    for feature in features:
        st.session_state[feature] = float(
            metrics["feature_means"][feature]
        )

    st.session_state["initialized"] = True


if "prediction" not in st.session_state:
    st.session_state["prediction"] = None


if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = None


if "example_values" not in st.session_state:
    st.session_state["example_values"] = None


if "predicted_values" not in st.session_state:
    st.session_state["predicted_values"] = None


# ============================================================
# ANALYZER FUNCTIONS
# ============================================================

def reset_values():

    for feature in features:
        st.session_state[feature] = float(
            metrics["feature_means"][feature]
        )

    st.session_state["prediction"] = None
    st.session_state["selected_example"] = None
    st.session_state["example_values"] = None
    st.session_state["predicted_values"] = None


def load_example(quality):

    examples = df[df["quality"] == quality]

    if examples.empty:
        return

    example = examples.iloc[0]

    values = {}

    for feature in features:

        value = float(example[feature])

        st.session_state[feature] = value
        values[feature] = value

    st.session_state["selected_example"] = int(quality)
    st.session_state["example_values"] = values
    st.session_state["prediction"] = None
    st.session_state["predicted_values"] = None


def predict_quality():

    values = {
        feature: float(st.session_state[feature])
        for feature in features
    }

    input_df = pd.DataFrame(
        [values],
        columns=features,
    )

    prediction = model.predict(input_df)[0]

    st.session_state["prediction"] = float(prediction)
    st.session_state["predicted_values"] = values


def quality_description(score):

    if score < 5:
        return (
            "The model estimates a relatively low "
            "wine quality score."
        )

    if score < 6.5:
        return (
            "The model estimates a moderate "
            "wine quality score."
        )

    return (
        "The model estimates a relatively high "
        "wine quality score."
    )


def example_is_unchanged():

    original = st.session_state["example_values"]

    if original is None:
        return False

    return all(
        abs(
            float(st.session_state[feature])
            - original[feature]
        ) < 0.000001
        for feature in features
    )


def prediction_matches_sliders():

    predicted_values = st.session_state["predicted_values"]

    if predicted_values is None:
        return False

    return all(
        abs(
            float(st.session_state[feature])
            - predicted_values[feature]
        ) < 0.000001
        for feature in features
    )


# ============================================================
# CHART STYLING
# ============================================================

def style_chart(fig, height=420):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor="#15121B",
        font=dict(
            color=TEXT,
            size=13,
        ),
        height=height,
        margin=dict(
            l=30,
            r=30,
            t=50,
            b=50,
        ),
        xaxis=dict(
            gridcolor="#39303E",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#39303E",
            zeroline=False,
        ),
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍷 VinoIQ")

    st.caption("WINE QUALITY INTELLIGENCE")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Wine Analyzer",
            "📊 Model Insights",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 🧠 ML Engine")

    st.write(
        f"**Active Model:** {metrics['model_selected']}"
    )

    st.write(
        f"**Dataset:** {metrics['dataset_rows']} wines"
    )

    st.write(
        f"**Test MAE:** {metrics['test']['MAE']:.3f}"
    )

    st.divider()

    st.caption("DAT158 | Machine Learning Project")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    render_html("""
    <div class="hero-card">
        <div class="hero-label">
            MACHINE LEARNING MEETS WINE
        </div>
        <div class="hero-title">
            Discover the Science<br>
            Behind Wine Quality.
        </div>
        <div class="hero-description">
            VinoIQ uses machine learning to analyze
            chemical properties and estimate wine quality.
            Explore the data, experiment with different
            wine profiles, and discover how machine
            learning can support predictive analysis.
        </div>
    </div>
    """)

    render_html("""
    <div class="section-label">
        PLATFORM OVERVIEW
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Wines in Dataset",
        f"{metrics['dataset_rows']:,}",
    )

    col2.metric(
        "Chemical Features",
        len(features),
    )

    col3.metric(
        "Test R²",
        f"{metrics['test']['R2']:.3f}",
    )

    st.write("")

    st.markdown("## Explore VinoIQ")

    col1, col2 = st.columns(2)

    with col1:

        render_html("""
        <div class="info-card">
            <h3>🔮 Wine Analyzer</h3>
            <p>
                Analyze chemical properties and
                receive a machine learning prediction
                of wine quality.
            </p>
        </div>
        """)

    with col2:

        render_html("""
        <div class="info-card">
            <h3>📊 Model Insights</h3>
            <p>
                Explore model performance, prediction
                ranges, feature importance, and
                the underlying dataset.
            </p>
        </div>
        """)

    render_html("""
    <div class="note-card">
        Select <strong>Wine Analyzer</strong> from the
        sidebar to analyze a wine profile.
        Choose a real example wine or adjust
        the chemical measurements manually.
    </div>
    """)


# ============================================================
# WINE ANALYZER
# ============================================================

elif page == "🔮 Wine Analyzer":

    render_html("""
    <div class="section-label">
        PREDICTIVE ANALYSIS
    </div>
    """)

    st.title("Wine Quality Analyzer")

    st.write(
        "Analyze chemical measurements using a "
        "machine learning model trained on Portuguese red wine."
    )

    st.divider()

    # --------------------------------------------------------
    # EXAMPLE WINES
    # --------------------------------------------------------

    st.subheader("🍇 Choose an Example Wine")

    st.write(
        "Load a real observation from the dataset "
        "to see how the model performs."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.button(
            "Quality 3/10",
            use_container_width=True,
            on_click=load_example,
            args=(3,),
        )

    with col2:

        st.button(
            "Quality 5/10",
            use_container_width=True,
            on_click=load_example,
            args=(5,),
        )

    with col3:

        st.button(
            "Quality 7/10",
            use_container_width=True,
            on_click=load_example,
            args=(7,),
        )

    if st.session_state["selected_example"] is not None:

        st.info(
            "Example wine loaded. "
            "Recorded quality: "
            f"{st.session_state['selected_example']}/10."
        )

    st.divider()

    # --------------------------------------------------------
    # CHEMICAL PROFILE
    # --------------------------------------------------------

    st.subheader("⚗️ Chemical Profile")

    st.write(
        "Adjust the measurements to create your own "
        "wine profile. Hover over the help icons "
        "to learn more about each feature."
    )

    col1, col2 = st.columns(2)

    for index, feature in enumerate(features):

        name, description = FEATURE_INFO.get(
            feature,
            (feature.title(), ""),
        )

        minimum = float(
            metrics["feature_min"][feature]
        )

        maximum = float(
            metrics["feature_max"][feature]
        )

        target = col1 if index % 2 == 0 else col2

        with target:

            st.slider(
                name,
                min_value=minimum,
                max_value=maximum,
                key=feature,
                help=description,
            )

    st.divider()

    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

    col1, col2 = st.columns([3, 1])

    with col1:

        st.button(
            "🔮 Analyze Wine Quality",
            type="primary",
            use_container_width=True,
            on_click=predict_quality,
        )

    with col2:

        st.button(
            "↺ Reset",
            use_container_width=True,
            on_click=reset_values,
        )

    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    prediction = st.session_state["prediction"]

    if prediction is not None:

        if not prediction_matches_sliders():

            st.info(
                "The chemical profile has changed. "
                "Click Analyze Wine Quality to update the prediction."
            )

        else:

            description = quality_description(prediction)

            render_html(f"""
            <div class="result-card">
                <div class="result-label">
                    PREDICTED WINE QUALITY
                </div>
                <div class="result-score">
                    {prediction:.2f}
                    <span class="result-max">/ 10</span>
                </div>
                <div class="result-description">
                    {description}
                </div>
            </div>
            """)

            st.markdown("### Quality Scale")

            percentage = max(
                0.0,
                min(prediction * 10, 100.0),
            )

            render_html(f"""
            <div class="quality-track">
                <div
                    class="quality-fill"
                    style="width: {percentage:.2f}%;">
                </div>
            </div>
            <div class="quality-labels">
                <span>0 · Lower</span>
                <span>5 · Moderate</span>
                <span>10 · Higher</span>
            </div>
            """)

            render_html("""
            <div class="note-card">
                This result is a statistical estimate,
                not an objective judgment of taste.
                The model may make mistakes, especially
                for unusual wine profiles.
            </div>
            """)

            # ------------------------------------------------
            # EXAMPLE COMPARISON
            # ------------------------------------------------

            if (
                st.session_state["selected_example"] is not None
                and example_is_unchanged()
            ):

                actual = st.session_state["selected_example"]

                difference = abs(
                    prediction - actual
                )

                st.markdown("### Example Comparison")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Predicted Quality",
                    f"{prediction:.2f}",
                )

                c2.metric(
                    "Recorded Quality",
                    str(actual),
                )

                c3.metric(
                    "Absolute Error",
                    f"{difference:.2f}",
                )

                st.caption(
                    "Examples are selected from the full dataset. "
                    "Some may belong to the training data, so "
                    "these comparisons should not be treated "
                    "as independent model evaluation."
                )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "📊 Model Insights":

    render_html("""
    <div class="section-label">
        MODEL PERFORMANCE
    </div>
    """)

    st.title("Model Insights")

    st.write(
        "Explore model performance, prediction behavior, "
        "the dataset, and the most important features."
    )

    st.divider()

    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    st.subheader("🏆 Selected Model")

    st.success(
        f"Selected Model: {metrics['model_selected']}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test MAE",
        f"{metrics['test']['MAE']:.3f}",
    )

    c2.metric(
        "Test RMSE",
        f"{metrics['test']['RMSE']:.3f}",
    )

    c3.metric(
        "Test R²",
        f"{metrics['test']['R2']:.3f}",
    )

    render_html("""
    <div class="note-card">
        <strong>MAE:</strong> Mean Absolute Error,
        the average absolute prediction error.<br>
        <strong>RMSE:</strong> Root Mean Squared Error,
        which gives greater weight to larger errors.<br>
        <strong>R²:</strong> Measures how much variation
        the model explains compared with predicting
        a constant mean.
    </div>
    """)

    st.divider()

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    comparison = pd.DataFrame(
        metrics["validation"]
    ).T

    st.dataframe(
        comparison,
        use_container_width=True,
    )

    st.markdown("#### Validation MAE")

    comparison_chart = comparison.reset_index()

    comparison_chart = comparison_chart.rename(
        columns={"index": "Model"}
    )

    fig = px.bar(
        comparison_chart,
        x="Model",
        y="MAE",
        color="Model",
        color_discrete_sequence=[
            BLUE,
            WINE,
            GOLD,
        ],
    )

    fig.update_layout(
        showlegend=False,
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )

    st.caption(
        "The model with the lowest validation MAE "
        "was selected as the final model."
    )

    st.divider()

    # ========================================================
    # PREDICTION RANGE ANALYSIS
    # ========================================================

    st.subheader("🎯 Prediction Range Analysis")

    st.write(
        "Why does the model rarely predict "
        "wine quality scores above 7?"
    )

    st.write(
        "Random Forest combines predictions from many "
        "decision trees. This often produces predictions "
        "that are closer to the middle of the quality distribution."
    )

    if test_predictions is not None:

        actual = test_predictions["actual"]
        predicted = test_predictions["predicted"]

        # ----------------------------------------------------
        # PREDICTION STATISTICS
        # ----------------------------------------------------

        minimum_prediction = float(
            predicted.min()
        )

        maximum_prediction = float(
            predicted.max()
        )

        average_prediction = float(
            predicted.mean()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Lowest Prediction",
            f"{minimum_prediction:.2f}",
        )

        c2.metric(
            "Highest Prediction",
            f"{maximum_prediction:.2f}",
        )

        c3.metric(
            "Average Prediction",
            f"{average_prediction:.2f}",
        )

        st.caption(
            "These values are calculated from the independent "
            "test set, not from every possible wine profile."
        )

        st.write("")

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Actual vs. Predicted Quality")

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=actual,
                name="Actual Quality",
                marker_color=BLUE,
                opacity=0.65,
                xbins=dict(
                    start=0,
                    end=10,
                    size=0.25,
                ),
            )
        )

        fig.add_trace(
            go.Histogram(
                x=predicted,
                name="Predicted Quality",
                marker_color=GOLD,
                opacity=0.65,
                xbins=dict(
                    start=0,
                    end=10,
                    size=0.25,
                ),
            )
        )

        fig.update_layout(
            barmode="overlay",
            xaxis_title="Wine Quality",
            yaxis_title="Number of Wines",
            legend=dict(
                orientation="h",
                y=1.15,
            ),
        )

        fig.update_xaxes(
            range=[0, 10],
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        render_html("""
        <div class="note-card">
            This chart compares actual quality scores
            with model predictions on the test set.
            Predictions are often more concentrated
            around the average than the recorded scores.
        </div>
        """)

        # ----------------------------------------------------
        # PREDICTED VS ACTUAL SCATTER PLOT
        # ----------------------------------------------------

        st.subheader("Predicted vs. Actual")

        fig = px.scatter(
            test_predictions,
            x="actual",
            y="predicted",
            color="absolute_error",
            color_continuous_scale=[
                WINE,
                "#D48D9B",
                GOLD,
            ],
            labels={
                "actual": "Actual Quality",
                "predicted": "Predicted Quality",
                "absolute_error": "Absolute Error",
            },
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 10],
                y=[0, 10],
                mode="lines",
                name="Perfect Prediction",
                line=dict(
                    color=GOLD,
                    dash="dash",
                ),
            )
        )

        fig.update_xaxes(
            range=[0, 10],
        )

        fig.update_yaxes(
            range=[0, 10],
        )

        st.plotly_chart(
            style_chart(fig, height=500),
            use_container_width=True,
        )

        st.caption(
            "The dashed line represents perfect predictions. "
            "Points closer to the line have smaller prediction errors."
        )

        # ----------------------------------------------------
        # ERROR BY QUALITY
        # ----------------------------------------------------

        st.subheader("Prediction Error by Quality")

        error_by_quality = (
            test_predictions
            .groupby("actual")
            .agg(
                observations=(
                    "absolute_error",
                    "size",
                ),
                mean_absolute_error=(
                    "absolute_error",
                    "mean",
                ),
                mean_prediction=(
                    "predicted",
                    "mean",
                ),
            )
            .reset_index()
        )

        fig = px.bar(
            error_by_quality,
            x="actual",
            y="mean_absolute_error",
            color_discrete_sequence=[WINE],
            labels={
                "actual": "Actual Quality",
                "mean_absolute_error": "Mean Absolute Error",
            },
            title="Mean Absolute Error by Quality",
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
        )

        display_table = error_by_quality.rename(
            columns={
                "actual": "Actual Quality",
                "observations": "Number of Wines",
                "mean_absolute_error": "MAE",
                "mean_prediction": "Average Prediction",
            }
        )

        st.dataframe(
            display_table.round(3),
            use_container_width=True,
            hide_index=True,
        )

        render_html("""
        <div class="note-card">
            The model may struggle more with very low
            and very high quality scores. One possible
            reason is that these scores are less common
            in the training data.
        </div>
        """)

        # ----------------------------------------------------
        # HIGH QUALITY WINES
        # ----------------------------------------------------

        st.subheader("High Quality Wines")

        high_quality = test_predictions[
            test_predictions["actual"] >= 7
        ]

        if not high_quality.empty:

            high_actual_mean = float(
                high_quality["actual"].mean()
            )

            high_predicted_mean = float(
                high_quality["predicted"].mean()
            )

            high_mae = float(
                high_quality["absolute_error"].mean()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Actual Average",
                f"{high_actual_mean:.2f}",
            )

            c2.metric(
                "Predicted Average",
                f"{high_predicted_mean:.2f}",
            )

            c3.metric(
                "MAE for Quality ≥ 7",
                f"{high_mae:.2f}",
            )

            st.write(
                f"The test set contains **{len(high_quality)} wines** "
                "with recorded quality scores of 7 or higher."
            )

            if high_predicted_mean < high_actual_mean:

                render_html("""
                <div class="note-card">
                    For wines with actual quality scores
                    of 7 or higher, the average predicted
                    score is lower than the actual average.
                    This indicates that the model tends
                    to underestimate high-quality wines
                    in the test set.
                </div>
                """)

            elif high_predicted_mean > high_actual_mean:

                render_html("""
                <div class="note-card">
                    For wines with actual quality scores
                    of 7 or higher, the average predicted
                    score is higher than the actual average.
                    This indicates an average overestimation
                    for this group in the test set.
                </div>
                """)

            else:

                st.info(
                    "The average prediction matches "
                    "the actual average for this group."
                )

        else:

            st.info(
                "No wines with quality scores of 7 or higher "
                "were found in the test set."
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.subheader("Why Does This Happen?")

        st.markdown(
            """
            **1. Imbalanced Dataset**

            Most wines have quality scores of 5 or 6.
            Very high and very low scores are less common.

            **2. Random Forest Uses Averages**

            Each decision tree predicts the average target
            value of the training observations in a leaf.
            Random Forest then averages predictions
            across multiple trees.

            **3. Extreme Scores Are Harder to Predict**

            Averaging can pull predictions toward
            the middle of the quality distribution.

            **4. Chemical Features Do Not Explain Everything**

            Wine quality also depends on factors
            that are not included in the dataset.
            """
        )

        render_html("""
        <div class="note-card">
            <strong>Important:</strong>
            A Random Forest regressor cannot predict
            beyond the minimum and maximum target
            values in its training data.
            However, the averaging process may
            produce a much narrower prediction range.
            The highest test-set prediction is not
            necessarily the model's theoretical maximum.
        </div>
        """)

    else:

        st.warning(
            "Test predictions were not found. "
            "Run the updated training script to generate "
            "`models/test_predictions.csv`."
        )

    st.divider()

    # --------------------------------------------------------
    # QUALITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Wine Quality Distribution")

    quality_counts = (
        df["quality"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    quality_counts.columns = [
        "Quality",
        "Count",
    ]

    fig = px.bar(
        quality_counts,
        x="Quality",
        y="Count",
        color_discrete_sequence=[BLUE],
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
    )

    st.caption(
        "Distribution of recorded wine quality "
        "scores in the complete dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        st.subheader("Feature Importance")

        importance = pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_,
            }
        )

        importance = importance.sort_values(
            "Importance",
            ascending=False,
        )

        fig = px.bar(
            importance,
            x="Feature",
            y="Importance",
            color_discrete_sequence=[BLUE],
        )

        fig.update_xaxes(
            tickangle=-45,
        )

        st.plotly_chart(
            style_chart(fig, height=480),
            use_container_width=True,
        )

        render_html("""
        <div class="note-card">
            Feature importance indicates which
            variables the model relies on most.
            These values do not prove that a
            feature causes higher or lower wine quality.
        </div>
        """)

    st.divider()

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Observations",
        len(df),
    )

    c2.metric(
        "Input Features",
        len(features),
    )

    c3.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )

    with st.expander("View Dataset"):

        st.dataframe(
            df.head(20),
            use_container_width=True,
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_html("""
    <div class="section-label">
        ABOUT THE PROJECT
    </div>
    """)

    st.title("About VinoIQ")

    st.markdown(
        """
        ## Wine Quality Intelligence

        VinoIQ is a machine learning web application
        developed as part of the DAT158 course.

        The project investigates how physicochemical
        measurements can be used to estimate
        sensory wine quality.

        ---

        ## 🧠 Machine Learning

        The project compares three models:

        - Dummy Regressor (baseline)
        - Ridge Regression
        - Random Forest Regressor

        Model performance is evaluated using:

        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - R² Score

        Model selection is based on a separate
        validation set.

        The selected model is then retrained using
        the combined training and validation data.

        Finally, the model is evaluated on an
        independent test set.

        ---

        ## 📊 Dataset

        The project uses the Wine Quality dataset
        from the UCI Machine Learning Repository.

        The dataset contains 1,599 observations
        of Portuguese red wine.

        Each observation includes 11 chemical
        measurements and one recorded quality score.

        ---

        ## 🎯 Prediction Range Analysis

        The application investigates how the
        selected model behaves across different
        wine quality levels.

        Predictions are analyzed using an
        independent test set.

        The analysis includes:

        - Minimum and maximum predictions
        - Actual versus predicted quality
        - Prediction errors by quality score
        - Performance on high-quality wines

        These results help explain why the model
        may underestimate unusually high-quality wines.

        ---

        ## ⚠️ Limitations

        The model estimates sensory quality
        but cannot replace professional tasting.

        It is trained on a relatively small dataset,
        and predictions may be less reliable for
        other wine types or unusual chemical profiles.

        Random Forest regression can also
        underestimate extreme quality scores.

        Example wines are included for demonstration
        purposes and are not necessarily part of
        the independent test set.

        ---

        ## 📚 References

        **Dataset**

        [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

        **Machine Learning**

        [Scikit-learn](https://scikit-learn.org/stable/)

        **Web Application**

        [Streamlit](https://streamlit.io/)

        **Data Visualization**

        [Plotly](https://plotly.com/python/)

        ---

        **VinoIQ — DAT158 Machine Learning Project**
        """
    )