# Sentiment Analysis on Large-Scale Review Data

This project implements an end-to-end sentiment analysis system on a large-scale text review dataset using classical NLP techniques and machine learning. The pipeline covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask web application.

---

## Dataset Description

The dataset contains approximately **3.6 million text reviews** with binary sentiment labels. Each record includes a short summary, a detailed review, and a numeric score indicating sentiment polarity.

### Dataset Schema

| Column | Description |
|------|------------|
| Score | Binary rating label (1 = Negative, 2 = Positive) |
| Summary | Short headline of the review |
| Review | Full textual review content |

The dataset is **balanced**, containing an equal number of positive and negative samples, which helps ensure stable and unbiased model training.

---

## Data Preprocessing

Preprocessing is intentionally lightweight to preserve semantic information:

- Converted text to lowercase
- Normalized whitespace
- Handled missing values safely
- Converted raw score values into binary sentiment labels
- Avoided aggressive text cleaning to prevent information loss

Stopword handling and feature filtering are performed during vectorization.

---

## Feature Engineering

Text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)** with the following configuration:

- Unigrams and bigrams (`ngram_range=(1, 2)`)
- English stopword filtering
- Fixed vocabulary size for computational efficiency

Including bigrams allows the model to capture short contextual patterns such as negation (e.g., *"not good"*), improving sentiment accuracy.

---

## Model Architecture

A **Logistic Regression** classifier is trained on TF-IDF features.

**Why Logistic Regression?**
- Strong baseline for high-dimensional sparse text data
- Efficient training on millions of samples
- Interpretable and reliable performance

---

## Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~86% |
| Precision | ~0.86 |
| Recall | ~0.86 |
| F1-Score | ~0.86 |

Performance is consistent across both sentiment classes, indicating good generalization.

---

## Application Layer

The trained model is deployed using a **Flask web application** that:

- Accepts user-input review text
- Applies the same preprocessing and vectorization pipeline used during training
- Returns real-time sentiment predictions
- Provides a simple HTML-based user interface

This ensures consistency between training and inference environments.

---

## Project Structure

