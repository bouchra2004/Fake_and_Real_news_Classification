# Fake and Real News Classification Using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

## Project Overview

This project focuses on developing a Machine Learning system capable of automatically classifying news articles as **Fake News** or **Real News** using Natural Language Processing (NLP) techniques.

The system analyzes textual content, extracts relevant linguistic features, and uses supervised Machine Learning algorithms to predict the category of a given news article.

An interactive web application built with Streamlit will allow users to enter news text and obtain a classification result.

## Project Objectives

- Explore and analyze a labeled news dataset.
- Perform text preprocessing and cleaning.
- Convert textual data into numerical features using TF-IDF.
- Train and compare different supervised Machine Learning models.
- Evaluate model performance using classification metrics.
- Analyze false positives and false negatives.
- Develop an interactive Streamlit application.

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning |
| TF-IDF | Text vectorization |
| Jupyter Notebook | Experiments and analysis |
| Streamlit | Web application |

## Machine Learning Modelss

The project will compare the following algorithms:

1. **DummyClassifier** – Baseline model.
2. **Multinomial Naive Bayes** – Probabilistic classification.
3. **Logistic Regression** – Supervised classification.
4. **LinearSVC** – Linear Support Vector Classification.

All models will be evaluated using the same training and testing protocol.

## Project Workflow

1. **Data Collection:** Obtain a public and documented news dataset.
2. **Data Exploration:** Analyze dataset size, class distribution, text lengths and duplicates.
3. **Text Preprocessing:** Clean and prepare textual data.
4. **Feature Extraction:** Apply TF-IDF vectorization.
5. **Model Training:** Train and compare the selected classification algorithms.
6. **Model Evaluation:** Measure performance and analyze classification errors.
7. **Deployment:** Build an interactive application using Streamlit.

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Structure

```text
fake-and-real-news-classification/
│
├── dataset/
├── notebooks/
├── src/
├── models/
├── app/
├── results/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd fake-and-real-news-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/app.py
```

## Application Features

- Enter a news article or text.
- Predict whether the content is classified as Fake News or Real News.
- Display the prediction result.
- View model evaluation metrics and confusion matrix.
- Display a warning about the limitations of automatic classification.

## Important Disclaimer

This project performs statistical text classification based on patterns learned from the training dataset.

A prediction labeled **Fake News** does not constitute proof that the information is false. The system is not a fact-checking tool, and its predictions may contain errors.

## Authors

**Bouchra OULMOUK & Houda JABAL**

Academic Project – Artificial Intelligence and Its Applications

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
