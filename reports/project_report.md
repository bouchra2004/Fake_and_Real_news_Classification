# Fake News Detection — Project Report

## Artificial Intelligence and Its Applications

**Authors**: Bouchra OULMOUK & Houda JABAL

**Academic Year**: 2025–2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Presentation](#4-dataset-presentation)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Text Preprocessing](#6-text-preprocessing)
7. [TF-IDF Representation](#7-tf-idf-representation)
8. [Train/Test Split Methodology](#8-traintest-split-methodology)
9. [Models Studied](#9-models-studied)
10. [Experimental Protocol](#10-experimental-protocol)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Results](#12-results)
13. [Model Comparison](#13-model-comparison)
14. [Error Analysis](#14-error-analysis)
15. [Biases and Limitations](#15-biases-and-limitations)
16. [Streamlit Application](#16-streamlit-application)
17. [Conclusion](#17-conclusion)
18. [Future Work](#18-future-work)
19. [References](#19-references)

---

## 1. Introduction

The proliferation of digital media and social networks has made the spread of misinformation a significant challenge. Fake news — articles that contain false or misleading information — can influence public opinion and decision-making. This project explores whether Natural Language Processing (NLP) and Machine Learning (ML) techniques can be used to automatically classify news articles as "Fake" or "Real" based on their textual content.

It is important to clarify from the outset that this project implements a **statistical text classifier**, not a fact-checking system. The model learns patterns from a labeled dataset and classifies new texts based on those patterns. The predictions reflect statistical similarities to the training data, not the factual accuracy of the information.

## 2. Problem Statement

**How can we use NLP and supervised Machine Learning to automatically classify news articles as Fake or Real based on the textual patterns present in a labeled dataset?**

This is formulated as a **binary classification** problem:
- **Class 0 (Fake)**: Articles labeled as fake news in the dataset
- **Class 1 (Real)**: Articles labeled as real news in the dataset

### Target Users

- Students exploring text classification techniques
- Readers interested in understanding how ML approaches fake news detection
- Users seeking an educational demonstration of NLP pipelines

This system does **not** replace professional fact-checkers, journalists, or editorial processes.

### Cost of Errors

- **False Positive** (Real → Fake): A legitimate article is incorrectly flagged as fake, potentially undermining trust in credible sources.
- **False Negative** (Fake → Real): A fake article is classified as real, failing to flag potentially misleading content.

Both error types are consequential, which is why model evaluation should not rely solely on accuracy but also on precision, recall, and F1-score.

## 3. Objectives

1. Explore and analyze a labeled news corpus
2. Preprocess and clean textual data for NLP
3. Build numerical text representations using TF-IDF
4. Train and compare four supervised ML models
5. Evaluate performance using multiple classification metrics
6. Analyze classification errors (false positives and false negatives)
7. Discuss biases and limitations of the approach
8. Develop an interactive Streamlit web application
9. Document the methodology for reproducibility

## 4. Dataset Presentation

### Source

The dataset used is the **Fake and Real News Dataset** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Structure

The dataset consists of two CSV files:
- **Fake.csv**: Articles labeled as fake news
- **True.csv**: Articles labeled as real news

Each file contains the following columns:
- `title`: The headline of the article
- `text`: The body text of the article
- `subject`: The topic category
- `date`: The publication date

### Labeling

Labels are assigned during data loading:
- Fake.csv → label = 0
- True.csv → label = 1

### Language

The articles are written in **English**. The model is designed for English text only.

### Dataset Statistics

- **Total number of articles**: 44,898 (raw), 39,100 (after duplicate removal)
- **Number of Fake articles**: 23,481 (raw)
- **Number of Real articles**: 21,417 (raw)
- **Class balance**: Approximately balanced (52.3% Fake, 47.7% Real)
- **Number of duplicates found**: 5,793 (removed based on title + text)
- **Missing values**: None across all columns
- **Articles removed after cleaning**: 5 (empty text after preprocessing)

### License

The dataset is available on Kaggle. Users should verify the dataset's license terms on the Kaggle page. We do not version the CSV files in the repository due to their size.

## 5. Exploratory Data Analysis

### Dataset Dimensions

- **Fake.csv**: 23,481 rows × 4 columns (`title`, `text`, `subject`, `date`)
- **True.csv**: 21,417 rows × 4 columns (`title`, `text`, `subject`, `date`)
- **Merged raw dataset**: 44,898 rows × 5 columns (including `label`)

### Class Distribution

- **Fake (0)**: 23,481 articles (**52.3%**)
- **Real (1)**: 21,417 articles (**47.7%**)
- The raw dataset is approximately balanced (visualized in `reports/figures/class_distribution.png`). After removing duplicates and empty texts (39,100 articles remaining), the split remains nearly balanced (~45.8% Fake, ~54.2% Real).

### Text Length Analysis

- **Body text character length** (`reports/figures/text_length_distribution.png`):
  - Fake: mean **2,547** chars (median **2,166**, std 2,533, max 51,794)
  - Real: mean **2,383** chars (median **2,222**, std 1,685, max 29,781)
- **Body word count**:
  - Fake: mean **423** words (median **363**, std 408)
  - Real: mean **386** words (median **359**, std 274)
- **Title character length** (`reports/figures/title_length_boxplot.png`):
  - Fake: mean **94.2** chars (median **90**, range 8–286)
  - Real: mean **64.7** chars (median **64**, range 26–133)
  - Fake titles are substantially longer and more variable than concise Reuters headlines.

### Missing Values

- **NaN values**: 0 missing (`null`) entries across all columns (`title`, `text`, `subject`, `date`).
- **Empty strings**: 631 rows in `text` (from `Fake.csv`) contain whitespace-only body text; all titles are non-empty. Combining `title` + `text` and filtering empty cleaned texts ensures clean inputs.

### Duplicate Analysis

- **Exact full-row duplicates**: 209
- **Duplicate `text` values**: 6,252
- **Duplicate `(title, text)` pairs**: 5,793
- **Cross-class duplicates**: 0 (no article appears in both Fake and Real).
- Removing the 5,793 `(title, text)` duplicates before splitting prevents identical articles from appearing in both train and test sets.

### Subject Analysis

The `subject` column exhibits **zero overlap** between classes (`reports/figures/subject_distribution.png`):
- **Fake (0)**: `News` (9,050), `politics` (6,841), `left-news` (4,459), `Government News` (1,570), `US_News` (783), `Middle-east` (778)
- **Real (1)**: `politicsNews` (11,272), `worldnews` (10,145)

Because `subject` directly encodes the source label, it is **excluded** from model features to prevent trivial data leakage.

### Key Observations

1. **Strong source marker leakage risk**: **99.2%** of Real articles (`21,247 / 21,417`) contain `(Reuters)` vs. only **0.04%** of Fake articles (`9 / 23,481`). We strip the `CITY (Reuters) -` dateline prefix during preprocessing to mitigate this shortcut.
2. **Formatting and headline style differ strongly**: Fake headlines are ~30 characters longer on average and often use clickbait/all-caps phrasing.
3. **Deduplication is mandatory**: 5,793 duplicate `(title, text)` records must be dropped prior to the train/test split.

## 6. Text Preprocessing

### Preprocessing Pipeline

The following cleaning steps are applied in order:

1. **HTML tag removal**: Strips any HTML markup from the text
2. **URL removal**: Removes HTTP/HTTPS links and www addresses
3. **Email address removal**: Removes email addresses
4. **Reuters marker removal**: Removes the "CITY (Reuters) -" dateline pattern to reduce source-based bias (documented as a deliberate choice)
5. **Special character removal**: Removes characters that don't contribute to meaning while preserving apostrophes (for contractions like "don't") and hyphens (for compounds like "fact-checking")
6. **Whitespace normalization**: Collapses multiple spaces into single spaces

### Design Decisions

- **Lowercase conversion** is delegated to TfidfVectorizer (lowercase=True) to keep preprocessing and vectorization concerns separate
- **Stop word removal** is handled by TfidfVectorizer (stop_words='english')
- **Negation words** are preserved (not, never, no) as they carry meaning
- **Numbers** are preserved as they can be part of meaningful content
- **Reuters markers** are removed to mitigate source-based classification shortcuts

### Combined Features

We combine the article's `title` and `text` into a single string before cleaning. This provides the model with more contextual information, as titles often capture the article's main claim or framing.

## 7. TF-IDF Representation

### Why TF-IDF?

Machine learning models require numerical inputs, but news articles are raw text. TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors by measuring the importance of each term relative to the document and the corpus.

### How It Works

- **TF (Term Frequency)**: Measures how often a term appears in a document. Higher frequency → higher importance within that document.
- **IDF (Inverse Document Frequency)**: Penalizes terms that appear across many documents (common words like "the", "is"). Terms appearing in fewer documents get higher IDF scores.
- **TF-IDF = TF × IDF**: Balances local importance (within a document) and global rarity (across the corpus).

### Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `lowercase` | True | Normalizes case |
| `ngram_range` | (1, 2) | Captures bigrams like "fake news", "white house" |
| `max_features` | 50,000 | Limits vocabulary to most relevant terms |
| `min_df` | 3 | Ignores very rare terms (likely noise) |
| `max_df` | 0.95 | Ignores terms in >95% of documents |
| `sublinear_tf` | True | Applies log dampening for long documents |
| `stop_words` | 'english' | Removes common English stop words |

### Data Leakage Prevention

TF-IDF is integrated into an sklearn `Pipeline`:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(...)),
    ('classifier', Model(...))
])
```

The vectorizer learns vocabulary and IDF values **only from the training set**. When making predictions on test data, it uses the vocabulary learned during training — it does not refit.

### Sparse vs. Dense Representations

TF-IDF produces **sparse matrices** where most values are zero (a document only contains a small fraction of the total vocabulary). This is memory-efficient compared to dense representations.

### TF-IDF vs. Modern Embeddings

TF-IDF is a bag-of-words approach that ignores word order and semantic similarity. Modern approaches like Word2Vec, GloVe, or transformer-based embeddings (BERT) capture semantic relationships, but TF-IDF remains effective for supervised classification tasks and is well-suited for this academic project's scope.

## 8. Train/Test Split Methodology

- **Split ratio**: 80% train / 20% test
- **Stratification**: `stratify=y` ensures both sets maintain the original class distribution
- **Random seed**: `random_state=42` for reproducibility
- **Duplicate handling**: Duplicates are removed before splitting to prevent data leakage between train and test sets

The test set is reserved exclusively for final evaluation. It is never used for vocabulary learning, IDF computation, or model selection.

## 9. Models Studied

### 1. DummyClassifier (Baseline)

Predicts the most frequent class regardless of input. Establishes the minimum performance level — any useful model must significantly outperform this baseline.

**Strategy**: `most_frequent` — always predicts the majority class.

### 2. Multinomial Naive Bayes

A probabilistic classifier based on Bayes' theorem with the "naive" assumption that features (terms) are conditionally independent given the class. Well-suited for text classification with TF-IDF features.

### 3. Logistic Regression

A linear model that estimates the probability of class membership. Despite its name, it performs classification by applying a logistic function to a linear combination of features. Known for strong performance on high-dimensional text data.

**Configuration**: max_iter=1000, solver='lbfgs', C=1.0

### 4. LinearSVC

A Support Vector Classifier that finds the optimal hyperplane separating the two classes in the TF-IDF feature space. Particularly effective for high-dimensional, sparse text data.

**Configuration**: max_iter=2000, C=1.0

## 10. Experimental Protocol

All models are evaluated under the same conditions:
1. Same preprocessing pipeline
2. Same train/test split (80/20, stratified, random_state=42)
3. TF-IDF learned only on training data (via sklearn Pipeline)
4. Evaluation on the same held-out test set
5. Same metrics computed for all models

## 11. Evaluation Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Accuracy** | Proportion of correct predictions | Overall performance, but can be misleading with imbalanced classes |
| **Precision** | TP / (TP + FP) | How many predicted "Real" articles are actually real |
| **Recall** | TP / (TP + FN) | How many actual "Real" articles are correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall | Balances precision and recall |
| **Confusion Matrix** | 2×2 matrix of TP, TN, FP, FN | Visualizes error distribution |

We report metrics for the positive class (Real = 1) and use F1-Score as the primary metric for model selection, as it balances precision and recall.

## 12. Results

### Performance Table

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LinearSVC** | **0.9968** | **0.9955** | **0.9986** | **0.9971** |
| LogisticRegression | 0.9904 | 0.9867 | 0.9958 | 0.9912 |
| MultinomialNB | 0.9579 | 0.9611 | 0.9613 | 0.9612 |
| DummyClassifier | 0.5421 | 0.5421 | 1.0000 | 0.7030 |

### Confusion Matrices

Confusion matrices for all models are saved in `reports/figures/`. On the 7,820 test articles (3,581 Fake, 4,239 Real), the best model (**LinearSVC**) achieved a confusion matrix of `[[3562, 19], [6, 4233]]`:
- **Fake correctly classified (true=0, pred=0)**: 3,562 (99.47% of Fake test set)
- **Real correctly classified (true=1, pred=1)**: 4,233 (99.86% of Real test set)
- **Fake misclassified as Real (true=0, pred=1)**: 19 (0.53% of Fake test set)
- **Real misclassified as Fake (true=1, pred=0)**: 6 (0.14% of Real test set)

### Model Comparison Chart

The model comparison chart is saved in `reports/figures/model_comparison.png`.

## 13. Model Comparison

All three real classifiers significantly outperformed the DummyClassifier baseline (F1=0.70), confirming that they learn meaningful patterns from the text.

**LinearSVC** achieved the best performance (F1=0.9971), followed by LogisticRegression (F1=0.9912) and MultinomialNB (F1=0.9612). Both linear models (LinearSVC and LogisticRegression) performed exceptionally well, which is expected given TF-IDF's high-dimensional sparse feature space where linear decision boundaries are effective.

MultinomialNB, while still strong (96% accuracy), had more errors (329 vs. 25 for LinearSVC). This is consistent with the naive independence assumption being a stronger simplification for this dataset.

The very high performance across all models should be interpreted with caution — it likely reflects the dataset containing strong source-specific markers and stylistic differences between the two classes, rather than the models truly understanding factual accuracy.

## 14. Error Analysis

*(Note: In fake-news error analysis below, we track Real articles falsely flagged as Fake (`Real → Fake`) and Fake articles missed as Real (`Fake → Real`), matching `reports/results/error_analysis.json`.)*

### Real Articles Misclassified as Fake (`Real → Fake`) — 6 cases

Real articles incorrectly classified as Fake. Examples include:
- "Tiger tips Clinton, bear backs Trump in Siberian zoo's mock vote" — An unusual, lighthearted Reuters article about a zoo prediction. Its non-standard topic may resemble sensationalist fake news styles.
- "Trump's Washington foreign policy speech" — A Reuters article summarizing Trump's speech. The political content and direct quotes may overlap with vocabulary typical of fake articles in the dataset.
- "A Picture and its Story: Reaching out to rescue a Rohingya woman" — A human-interest story with emotional content.

**Pattern**: These false positives tend to be atypical Real articles — either unusually informal, emotionally charged, or covering topics that overlap with common fake news subjects.

### Fake Articles Misclassified as Real (`Fake → Real`) — 19 cases

Fake articles incorrectly classified as Real. Examples include:
- "Irma Update: Latest Look as Hurricane Strengthens to Cat 5" — Written in a factual, news-report style about a real event.
- "Court Slams Down Trump's Order Blocking Funding To Sanctuary Cities" — Written in formal journalistic style with factual-sounding claims.
- "BREAKING: Activist Appeals Court Rules Against Trump" — Mimics professional news reporting format.

**Pattern**: These false negatives are fake articles that adopt a more formal, professional writing style similar to real news reporting. They avoid the most obvious markers of fake news (extreme sensationalism, unusual formatting) and discuss real events, making them harder to distinguish.

### Error Patterns

The error analysis reveals that the model's decision boundary primarily relies on **stylistic and source-specific features** rather than factual accuracy. When fake articles are well-written and mimic professional journalism, or when real articles are informal or atypical, the model struggles.

## 15. Biases and Limitations

### Dataset Biases

- **Source bias**: Real news articles often contain Reuters dateline markers ("WASHINGTON (Reuters) -"), creating a classification shortcut. We mitigate this by removing these markers during preprocessing.
- **Temporal bias**: The dataset covers a specific time period. Articles from different periods may have different vocabulary and topics.
- **Subject bias**: The distribution of topics may differ between fake and real articles.
- **Style bias**: Fake and real articles may come from sources with distinct writing styles, which the model may learn instead of content-based features.

### Model Limitations

- The model classifies based on textual patterns, not factual accuracy
- Performance may degrade on articles from different sources, periods, or topics
- The model is designed for English text only
- TF-IDF ignores word order and semantic meaning
- The binary classification (Fake/Real) is an oversimplification — real-world misinformation exists on a spectrum

### Ethical Considerations

- A "Fake News" prediction should never be treated as proof of falsehood
- Automated classification can produce both false accusations (false positives) and missed detections (false negatives)
- The system should be used as one input among many, not as an authoritative source

## 16. Streamlit Application

The project includes an interactive web application built with Streamlit that allows users to:

1. Enter or paste a news article
2. Receive a Fake/Real classification
3. View model performance metrics
4. Read about the methodology and limitations

The application includes a prominent disclaimer explaining that predictions are statistical estimates, not factual assessments.

**Launch command**:
```bash
streamlit run app/app.py
```

## 17. Conclusion

This project successfully implemented a complete NLP pipeline for fake news classification, achieving very high performance metrics:

- **Best model**: LinearSVC with F1-Score of 0.9971 (only 25 errors out of 7,820 test articles)
- **All models** significantly outperformed the DummyClassifier baseline, confirming they learn meaningful patterns
- **Error analysis** revealed that most errors occur at the boundary between classes — well-written fake articles and atypical real articles

However, the exceptionally high accuracy (99.7%) should be interpreted with caution. Several factors suggest the model benefits from dataset-specific shortcuts:
1. **Source markers**: Even after removing Reuters dateline markers, residual source-specific vocabulary likely remains
2. **Style differences**: Fake and real articles in this dataset come from fundamentally different sources with distinct writing styles
3. **Subject correlation**: Some topics strongly correlate with class labels

The model demonstrates effective **text classification** within this dataset, but its ability to generalize to new sources, time periods, or formats is limited. It is a **pattern matcher**, not a **fact-checker**.

## 18. Future Work

- **Cross-validation**: Use stratified k-fold cross-validation for more robust performance estimates
- **Feature importance**: Analyze the most discriminative TF-IDF features to understand what the model is learning
- **Out-of-distribution testing**: Test with articles from different sources and time periods
- **Preprocessing impact**: Systematically compare performance with and without Reuters marker removal
- **Title vs. Text**: Compare classification using title only, text only, and combined features
- **Decision threshold analysis**: Study the effect of classification thresholds (for models with probability outputs)

## 19. References

1. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
2. **TF-IDF**: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
3. **Fake and Real News Dataset**: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
4. **Naive Bayes for Text Classification**: McCallum, A., & Nigam, K. (1998).
5. **Streamlit Documentation**: https://docs.streamlit.io/
6. **NLTK Documentation**: https://www.nltk.org/

