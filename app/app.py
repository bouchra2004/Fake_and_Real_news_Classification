"""
Streamlit Application — Fake News Detection

A professional, clean web interface for classifying news articles
as Fake or Real using the trained ML pipeline.
"""

import os
import sys
import json

import streamlit as st
import pandas as pd

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_text, load_pipeline
from src.preprocessing import validate_text_input


# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #FFEBEE;
        border: 2px solid #EF5350;
    }
    .real-result {
        background-color: #E8F5E9;
        border: 2px solid #66BB6A;
    }
    .disclaimer-box {
        background-color: #FFF3E0;
        border: 1px solid #FFB74D;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline(model_filename: str = "best_model.joblib"):
    """Load a trained pipeline once and cache it."""
    try:
        pipeline = load_pipeline(model_filename)
        return pipeline
    except FileNotFoundError:
        return None


def load_model_metrics():
    """Load saved model metrics from reports/results/."""
    metrics_path = os.path.join(PROJECT_ROOT, "reports", "results", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/newspaper.png", width=80)
    st.title("📰 Fake News Detection")
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Enter or paste a news article
    2. Click **Analyze**
    3. View the classification result

    **Classification Classes:**
    - 🔴 **Fake News** (label 0)
    - 🟢 **Real News** (label 1)
    """)

    st.markdown("---")
    st.markdown("""
    **NLP Pipeline:**
    - Text cleaning & preprocessing
    - TF-IDF vectorization
    - Machine Learning classification
    """)

    st.markdown("---")
    MODEL_OPTIONS = {
        "LinearSVC (Best — F1: 0.9971)": "best_model.joblib",
        "LogisticRegression (F1: 0.9912)": "logisticregression.joblib",
        "MultinomialNB (F1: 0.9612)": "multinomialnb.joblib",
        "DummyClassifier (Baseline)": "dummyclassifier.joblib",
    }
    selected_model_label = st.selectbox(
        "Model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Switch between trained classifiers to compare predictions."
    )
    selected_model_file = MODEL_OPTIONS[selected_model_label]

    st.markdown("---")
    st.caption("Academic Project — AI & Its Applications")
    st.caption("Bouchra OULMOUK & Houda JABAL")


# ── Main Content ──────────────────────────────────────────────────────────

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📰 Fake News Detection")
st.markdown(
    "Classify news articles as **Fake** or **Real** using Natural Language Processing "
    "and Machine Learning."
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_about = st.tabs([
    "🔍 Analyze Article", "📊 Model Performance", "ℹ️ About"
])


# ── Tab 1: Prediction ────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Enter a News Article")

    # Demo examples
    demo_examples = {
        "— Select a demo article —": "",
        "Demo: Real news (political report)": (
            "WASHINGTON (Reuters) - President Donald Trump said on Monday he plans to announce "
            "a major tax reform plan in the coming weeks as part of an effort to cut regulations "
            "and boost U.S. economic growth. Speaking during a meeting with business leaders at "
            "the White House, Trump said his administration would direct federal agencies to "
            "review existing rules affecting manufacturing and energy companies. Lawmakers from "
            "both parties said they were reviewing the proposal ahead of congressional hearings."
        ),
        "Demo: Fake news (sensationalist)": (
            "BREAKING: Scientists discover that drinking coffee makes you immortal! "
            "A new study that nobody has ever seen proves that coffee drinkers live forever. "
            "Big pharma doesn't want you to know this secret! Share this before it gets deleted!"
        ),
        "Demo: Out-of-domain / short text (limitation example)": (
            "WASHINGTON (Reuters) - The U.S. Senate voted on Thursday to approve a bipartisan "
            "infrastructure bill worth approximately $1 trillion, marking a significant legislative "
            "achievement after months of negotiations between Democrats and Republicans. The bill "
            "includes funding for roads, bridges, broadband internet, and water systems across "
            "the country. President Biden praised the vote, calling it a historic step forward "
            "for the American economy."
        ),
    }

    if "article_input" not in st.session_state:
        st.session_state["article_input"] = ""

    def on_demo_change():
        choice = st.session_state.get("demo_select", "— Select a demo article —")
        st.session_state["article_input"] = demo_examples.get(choice, "")

    selected_demo = st.selectbox(
        "Try a demo article:",
        options=list(demo_examples.keys()),
        key="demo_select",
        on_change=on_demo_change,
    )

    # Text area
    article_text = st.text_area(
        "Paste or type a news article below:",
        height=200,
        placeholder="Enter the full text of a news article here...",
        key="article_input"
    )

    # Word count display
    if article_text.strip():
        word_count = len(article_text.split())
        st.caption(f"📝 Word count: {word_count}")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_clicked = st.button(
            "🔍 Analyze Article",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["article_input"] = ""
            st.session_state["demo_select"] = "— Select a demo article —"
            st.rerun()

    # ── Prediction Result ─────────────────────────────────────────────────
    if analyze_clicked:
        if not article_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            pipeline = get_pipeline(selected_model_file)
            if pipeline is None:
                st.error(
                    "❌ **Model not found.** The prediction pipeline has not been trained yet.\n\n"
                    "Please train the models first:\n"
                    "```bash\n"
                    "python run_training.py\n"
                    "```"
                )
            else:
                with st.spinner("Analyzing article..."):
                    result = predict_text(article_text, pipeline=pipeline)

                if not result["input_valid"]:
                    st.warning(f"⚠️ {result['validation_message']}")
                else:
                    st.markdown("---")
                    st.subheader("Classification Result")

                    if result["label"] == 0:
                        st.markdown(
                            '<div class="result-box fake-result">'
                            '<h2>🔴 Fake News</h2>'
                            '<p>The model classifies this article as <strong>Fake News</strong> '
                            'based on the patterns learned from the training data.</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="result-box real-result">'
                            '<h2>🟢 Real News</h2>'
                            '<p>The model classifies this article as <strong>Real News</strong> '
                            'based on the patterns learned from the training data.</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words in input", result["word_count"])
                    with col2:
                        st.metric("Predicted class", result["label_name"])
                    with col3:
                        if result.get("confidence") is not None:
                            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                        elif result.get("decision_score") is not None:
                            st.metric("Decision margin", f"{result['decision_score']:+.3f}")
                        else:
                            st.metric("Model", selected_model_label.split(" ")[0])

                    score = result.get("decision_score")
                    if score is not None and abs(score) < 0.35:
                        st.info(
                            "ℹ️ **Borderline prediction:** The decision margin is close to 0 "
                            f"({score:+.3f}). Short texts or articles from outside the 2015–2017 "
                            "training period may lack familiar vocabulary and produce less reliable predictions."
                        )
                    elif result["word_count"] < 80:
                        st.info(
                            "ℹ️ **Short input notice:** Training articles average ~385–420 words. "
                            "Short excerpts provide fewer TF-IDF features and may be misclassified "
                            "if they fall outside the training corpus vocabulary."
                        )

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer-box">'
        '⚠️ <strong>Important Disclaimer</strong><br>'
        'This application classifies articles based on textual patterns learned from '
        'a training dataset. A prediction of <strong>Fake News</strong> does not constitute '
        'proof that the information is false, and a prediction of <strong>Real News</strong> '
        'does not guarantee that the information is true. The model can make errors and has '
        'limitations related to the training data, including potential biases in source, '
        'style, subject matter, and time period. '
        'This tool is not a fact-checking system.'
        '</div>',
        unsafe_allow_html=True
    )


# ── Tab 2: Model Performance ─────────────────────────────────────────────
with tab_metrics:
    st.subheader("📊 Model Performance Comparison")

    metrics = load_model_metrics()

    if metrics is None:
        st.info(
            "📋 Model evaluation results are not yet available.\n\n"
            "Train the models first to generate performance metrics."
        )
    else:
        # Build comparison table
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1_score']:.4f}",
            })
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Show confusion matrix images if available
        st.markdown("### Confusion Matrices")
        figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
        cm_files = [
            f for f in os.listdir(figures_dir)
            if f.startswith("confusion_matrix_") and f.endswith(".png")
        ] if os.path.exists(figures_dir) else []

        if cm_files:
            cols = st.columns(min(len(cm_files), 2))
            for i, cm_file in enumerate(sorted(cm_files)):
                with cols[i % 2]:
                    st.image(
                        os.path.join(figures_dir, cm_file),
                        caption=cm_file.replace("confusion_matrix_", "").replace(".png", "").replace("_", " ").title(),
                        use_container_width=True
                    )
        else:
            st.info("Confusion matrix figures have not been generated yet.")

        # Show comparison chart if available
        comparison_path = os.path.join(figures_dir, "model_comparison.png")
        if os.path.exists(comparison_path):
            st.markdown("### Performance Comparison Chart")
            st.image(comparison_path, use_container_width=True)


# ── Tab 3: About ──────────────────────────────────────────────────────────
with tab_about:
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### Fake News Detection — NLP-Based Classification

    This academic project implements a complete NLP pipeline to classify
    news articles as **Fake** or **Real** based on their textual content.

    #### Methodology
    1. **Data Exploration**: Analyze the dataset's structure, class distribution, and text characteristics.
    2. **Text Preprocessing**: Clean texts by removing noise (HTML, URLs, special characters)
       while preserving meaningful content.
    3. **TF-IDF Vectorization**: Convert cleaned text into numerical feature vectors
       using Term Frequency–Inverse Document Frequency.
    4. **Model Training**: Train and compare multiple classifiers:
       - DummyClassifier (baseline)
       - Multinomial Naive Bayes
       - Logistic Regression
       - Linear SVC
    5. **Evaluation**: Measure Accuracy, Precision, Recall, F1-Score, and analyze errors.

    #### Dataset
    - **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
    - **Language**: English
    - **Classes**: Fake (0) and Real (1)

    #### Known Limitations
    - The model is trained on a specific dataset and may not generalize to all news sources.
    - The dataset contains articles from a particular time period and set of sources.
    - The presence of source markers (e.g., Reuters) can create classification shortcuts.
    - This is a statistical classifier, not a fact-checking system.
    - Predictions should be interpreted with caution and critical thinking.

    #### Authors
    **Bouchra OULMOUK & Houda JABAL**

    Academic Project — Artificial Intelligence and Its Applications
    """)

