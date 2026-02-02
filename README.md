# fake-news-detector
A machine learning project to detect fake news using TF-IDF and Word2Vec

# Fake News Detector

A machine learning project for detecting fake news articles using natural language processing (NLP) techniques. This project implements multiple classification models, compares their performance, and considers ethical implications in automated fake news detection.


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Ethical Considerations](#ethical-considerations)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [References](#references)


## Introduction
With the rapid spread of online information, distinguishing real news from fake news is increasingly important. This project builds a tool to automatically classify news articles as real or fake using machine learning models and NLP techniques.



## Dataset
- **Source:** WELFake dataset (combined from Kaggle, CI-FAKE, FakeNewsNet, and genuine news articles from Reuters & The Guardian)
- **Size:** ~72,134 articles (full dataset not included in repo)
- **Features:**
  - `title`: Headline of the article
  - `text`: Body of the article
  - `label`: 0 = real, 1 = fake
- **Notes:** The dataset may contain political or ideological biases and subjectivity in defining "fake news."

> **Note:** Full dataset is not included due to size. You can download it from the [Kaggle WELFake dataset link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). A small sample CSV is included for testing purposes.



## Preprocessing
- Handling missing values: removed rows with missing text, filled missing titles
- Text cleaning: lowercase conversion, removed URLs, HTML tags, punctuation, special characters, numbers
- Combined `title` + `text` into `content` column
- Tokenization and lemmatization (stopwords removed)
- Feature engineering:
  - **TF-IDF** (primary)
  - **Word2Vec embeddings** (alternative approach)



## Models
Implemented using **scikit-learn**:

1. Logistic Regression (TF-IDF)
2. Logistic Regression (Word2Vec)
3. Decision Tree
4. Random Forest
5. Multinomial Naive Bayes
6. Support Vector Machine (LinearSVC)

- Hyperparameters tuned using GridSearchCV or manual search.



## Evaluation
Metrics used: Accuracy, Precision, Recall, F1-Score, and AUC (where applicable)

**Top Performing Models:**

| Model                       | Accuracy | F1-Score (Fake) |
|------------------------------|---------|----------------|
| Linear SVM (TF-IDF)         | 95.95%  | 0.96           |
| Logistic Regression (TF-IDF)| 95.65%  | 0.96           |
| Random Forest (TF-IDF)      | 94.24%  | 0.94           |

- **Observation:** TF-IDF consistently outperformed Word2Vec embeddings for this dataset.



## Ethical Considerations
- **Bias:** Models may favor mainstream news sources or political topics
- **Explainability:** Complex models like Random Forest and SVM act as black boxes
- **Consequences of Errors:**
  - False Positive (real → fake): may suppress legitimate speech
  - False Negative (fake → real): misinformation spreads
- **Mitigation:** Balanced model parameters, feature importance analysis, class imbalance auditing





