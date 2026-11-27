# Fake News Detection — Presentation Outline

## Slide 1: Title

**Fake News Detection — NLP-Based Classification**

- Bouchra OULMOUK & Houda JABAL
- Module: AI and Its Applications
- Academic Year 2025–2026

---

## Slide 2: Problem & Motivation

- Misinformation is a growing challenge in the digital age
- Can we use NLP and ML to automatically detect patterns in fake vs. real news articles?
- **Goal**: Build a text classifier that distinguishes Fake from Real news based on textual content
- **Important**: This is statistical classification, not fact-checking

---

## Slide 3: Project Objectives

1. Explore and analyze a labeled news corpus
2. Preprocess and clean textual data
3. Build TF-IDF text representations
4. Train and compare 4 supervised ML models
5. Evaluate using standard classification metrics
6. Analyze classification errors
7. Build an interactive Streamlit application

---

## Slide 4: Dataset

- **Source**: Kaggle — Fake and Real News Dataset
- **Files**: Fake.csv (label 0) + True.csv (label 1)
- **Language**: English
- **Columns**: title, text, subject, date
- **Total articles**: 44,898 (raw) → 39,100 (after removing 5,793 duplicates)
- **Class distribution**: ~52% Fake, ~48% Real (approximately balanced)
- **Missing values**: None

---

## Slide 5: Exploratory Data Analysis

- **Class balance**: 23,481 Fake (52.3%) vs. 21,417 Real (47.7%) raw; 39,100 after deduplication
- **Text & title lengths**: Real titles are concise (mean 64.7 chars) vs. Fake (mean 94.2 chars); body lengths ~386 vs. ~423 words; 631 whitespace-only texts in Fake.csv
- **Subject bias**: Zero overlap between Fake (`News`, `politics`, `left-news`, ...) and Real (`politicsNews`, `worldnews`) → excluded from features
- **Source marker leakage**: 99.2% of Real articles contain `(Reuters)` vs. 0.04% of Fake → dateline marker stripped in preprocessing
- **Visualizations**: `reports/figures/class_distribution.png`, `text_length_distribution.png`, `title_length_boxplot.png`, `subject_distribution.png`

---

## Slide 6: Text Preprocessing

- Cleaning steps: HTML removal, URL removal, whitespace normalization
- Reuters marker removal (source bias mitigation)
- Special character handling (preserve contractions and hyphens)
- Combined title + text for richer features
- **Decisions documented**: what we keep and why

---

## Slide 7: TF-IDF Representation

- **Why TF-IDF?** Converts text to numerical features while weighting term importance
- **TF** (Term Frequency): how often a word appears in a document
- **IDF** (Inverse Document Frequency): penalizes words appearing in many documents
- **Parameters**: bigrams, max_features=50K, sublinear TF, stop words removed
- **Pipeline integration**: TF-IDF learned ONLY on training data (no data leakage)

---

## Slide 8: Models

| Model | Description |
|-------|-------------|
| DummyClassifier | Baseline — predicts most frequent class |
| Multinomial Naive Bayes | Probabilistic classifier suited for text |
| Logistic Regression | Linear model with probability estimates |
| LinearSVC | Support Vector Classifier optimized for text |

- All models use the same train/test split (80/20, stratified)
- All use sklearn Pipeline (TF-IDF + Classifier)

---

## Slide 9: Evaluation Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LinearSVC** | **0.9968** | **0.9955** | **0.9986** | **0.9971** |
| LogisticRegression | 0.9904 | 0.9867 | 0.9958 | 0.9912 |
| MultinomialNB | 0.9579 | 0.9611 | 0.9613 | 0.9612 |
| DummyClassifier | 0.5421 | 0.5421 | 1.0000 | 0.7030 |

- **Best model**: LinearSVC (F1-Score: 0.9971, `reports/figures/model_comparison.png`, `confusion_matrix_linearsvc.png`)
- All models significantly outperform the baseline
- Very high performance — but interpret with caution (source-specific patterns)

---

## Slide 10: Error Analysis

- **False Positives** (Real classified as Fake): 6 cases — atypical Real articles (informal, emotional, unusual topics)
- **False Negatives** (Fake classified as Real): 19 cases — well-written fake articles mimicking professional journalism
- Total errors: only 25 out of 7,820 test articles
- Possible causes:
  - Source markers creating classification shortcuts
  - Vocabulary and style differences between sources
  - Topic and time period bias

---

## Slide 11: Streamlit Application Demo

- Live demonstration of the web application
- Features:
  - Text input for classification
  - Visual result display (Fake/Real)
  - Model performance metrics
  - Disclaimer about limitations
- _[Screenshot of the app]_

---

## Slide 12: Limitations & Conclusion

### Limitations
- Trained on a specific dataset (time period, sources)
- May not generalize to other languages, sources, or formats
- Source markers can create shortcuts (partially mitigated)
- Statistical classifier ≠ fact-checker
- Class definitions depend on the dataset annotations

### Conclusion
- LinearSVC achieved best performance (F1=0.9971, 99.7% accuracy)
- All models outperformed the baseline significantly
- Very high performance likely reflects source-specific patterns, not true fact-checking
- The model is an effective text classifier within this dataset, but generalization is limited
- Statistical classification ≠ fact-checking

### Future Improvements
- Cross-validation for more robust evaluation
- Testing with out-of-distribution articles
- Feature importance analysis (most discriminative terms)
- Investigating the impact of different preprocessing strategies

