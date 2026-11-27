# Fake and Real News Classification Using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## Project Overview

This project implements a complete NLP pipeline to automatically classify news articles as **Fake News** or **Real News** based on their textual content. It uses classical Machine Learning algorithms with TF-IDF text representation.

> **⚠️ Important:** This is a statistical text classifier, not a fact-checking system. Predictions reflect patterns learned from the training dataset and should not be taken as proof of an article's factual accuracy.

## Problem Statement

**How can we use NLP and supervised Machine Learning to classify news articles as Fake or Real based on textual patterns in a labeled dataset?**

This is a **binary classification** problem:
- **Class 0 — Fake News**: Articles labeled as fake in the dataset
- **Class 1 — Real News**: Articles labeled as real in the dataset

## Objectives

1. Explore and analyze a labeled news corpus
2. Preprocess and clean textual data
3. Build TF-IDF text representations
4. Train and compare 4 supervised ML models
5. Evaluate using classification metrics (Accuracy, Precision, Recall, F1-Score)
6. Analyze classification errors (false positives and false negatives)
7. Discuss biases and limitations
8. Build an interactive Streamlit web application

## Dataset

- **Source**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- **Files**: `Fake.csv` (label 0) + `True.csv` (label 1)
- **Language**: English
- **Columns**: `title`, `text`, `subject`, `date`

> **Note**: The CSV files are not included in the repository due to their size. Download them from the Kaggle link above and place them in `data/raw/`.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Main programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning & TF-IDF |
| Joblib | Model serialization |
| Jupyter Notebook | Interactive analysis |
| Streamlit | Web application |
| Pytest | Testing |

## Project Structure

```
Fake_and_Real_news_Classification/
│
├── data/
│   ├── raw/                    # Raw CSV files (Fake.csv, True.csv)
│   └── processed/              # Processed data files
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and dataset analysis
│   ├── 02_preprocessing.ipynb       # Text cleaning pipeline
│   ├── 03_tfidf_representation.ipynb # TF-IDF explanation
│   ├── 04_model_training.ipynb      # Model training
│   └── 05_model_evaluation.ipynb    # Evaluation and error analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and validation
│   ├── preprocessing.py        # Text cleaning functions
│   ├── train.py                # Model training pipelines
│   ├── evaluate.py             # Evaluation and visualization
│   ├── predict.py              # Prediction on new text
│   └── utils.py                # Utility functions
│
├── models/                     # Saved model pipelines (.joblib)
├── reports/
│   ├── figures/                # Generated visualizations
│   ├── results/                # Metrics and analysis (JSON/CSV)
│   ├── project_report.md       # Academic report
│   └── presentation_outline.md # Presentation slides outline
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── tests/
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   ├── test_prediction.py      # Prediction pipeline tests
│   └── test_data_validation.py # Data loading tests
│
├── run_training.py             # Full training pipeline script
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## NLP Workflow

```
Raw Text → Cleaning → TF-IDF Vectorization → ML Classification → Prediction
```

1. **Text Cleaning**: Remove HTML, URLs, emails, source markers (Reuters), special characters
2. **Feature Combination**: Merge article title + body text
3. **TF-IDF**: Convert cleaned text to numerical vectors (unigrams + bigrams, 50K features)
4. **Classification**: Predict Fake (0) or Real (1)

### TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF measures term importance by combining:
- **TF**: How often a word appears in a document
- **IDF**: How rare the word is across all documents (penalizes common words)

Key configuration:
- `ngram_range=(1, 2)` — captures word pairs like "fake news"
- `max_features=50000` — limits vocabulary to most informative terms
- `sublinear_tf=True` — dampens the effect of very high frequencies
- TF-IDF is integrated in an sklearn `Pipeline` to prevent data leakage

## Models Implemented

| Model | Description |
|---|---|
| **DummyClassifier** | Baseline — always predicts most frequent class |
| **MultinomialNB** | Naive Bayes classifier suited for text |
| **LogisticRegression** | Linear classifier with regularization |
| **LinearSVC** | Support Vector Classifier for text |

All models use the same train/test split (80/20, stratified) and the same TF-IDF configuration inside sklearn Pipelines.

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Of predicted Real, how many are actually Real |
| **Recall** | Of actual Real, how many are correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Breakdown of TP, TN, FP, FN |

## Installation

### Prerequisites
- Python 3.9+ (tested with 3.11 and 3.13)
- pip

### Setup

**Windows:**
```bash
git clone https://github.com/bouchra2004/Fake_and_Real_news_Classification.git
cd Fake_and_Real_news_Classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
git clone https://github.com/bouchra2004/Fake_and_Real_news_Classification.git
cd Fake_and_Real_news_Classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Dataset

1. Go to [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download `Fake.csv` and `True.csv`
3. Place them in `data/raw/`

## How to Run

### Train Models

```bash
python run_training.py
```

This will:
1. Load and validate the dataset
2. Remove duplicates
3. Preprocess text
4. Split data (80/20 stratified)
5. Train all 4 models
6. Evaluate and save results
7. Save the best model pipeline

### Launch Streamlit App

```bash
streamlit run app/app.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

## Results

Results from training on the dataset (44,898 articles → 39,100 after removing 5,793 duplicates):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **LinearSVC** | **0.9968** | **0.9955** | **0.9986** | **0.9971** |
| LogisticRegression | 0.9904 | 0.9867 | 0.9958 | 0.9912 |
| MultinomialNB | 0.9579 | 0.9611 | 0.9613 | 0.9612 |
| DummyClassifier | 0.5421 | 0.5421 | 1.0000 | 0.7030 |

**Best model**: LinearSVC (F1-Score: 0.9971, only 25 errors out of 7,820 test articles)

**Error analysis** (LinearSVC):
- False Positives (Real → Fake): 6 articles
- False Negatives (Fake → Real): 19 articles

> ⚠️ **Caution**: The very high performance likely reflects the model learning source-specific patterns (writing style, vocabulary) rather than truly detecting misinformation. See the report for a detailed discussion of biases and limitations.

## Known Limitations

1. **Dataset-specific**: The model is trained on articles from specific sources and time periods
2. **Source bias**: The dataset's Real articles predominantly come from Reuters — the model may learn source-specific patterns rather than content features
3. **English only**: The model works exclusively with English text
4. **Not fact-checking**: Predictions are based on textual patterns, not factual verification
5. **Temporal drift**: Performance may degrade on articles from different time periods
6. **Subject correlation**: Some topics correlate with class labels, creating shortcuts

## Future Improvements

- Cross-validation for more robust performance estimates
- Feature importance analysis to understand what the model learns
- Out-of-distribution testing with articles from other sources
- Comparison with and without Reuters marker removal
- Title-only vs. text-only vs. combined feature comparison

## Authors

**Bouchra OULMOUK & Houda JABAL**

Academic Project — Artificial Intelligence and Its Applications

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
