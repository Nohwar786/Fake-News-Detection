# 📰 Fake News Detection System

A Machine Learning and Natural Language Processing (NLP) based application that classifies news articles as **Fake News or Real News**.

## Features

* Fake and real news classification
* TF-IDF based text feature extraction
* Logistic Regression machine learning model
* Model comparison using multiple algorithms
* Model confidence score
* Interactive Streamlit web application
* Command-line prediction system
* Model evaluation using accuracy, precision, recall, and F1-score

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Joblib
* TF-IDF
* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM

## Dataset

The project uses a dataset containing labeled fake and real news articles.

The dataset contains two files:

* `Fake.csv` — Fake news articles
* `True.csv` — Real news articles

## Machine Learning Workflow

1. Load the fake and real news datasets
2. Assign labels to the articles
3. Combine the datasets
4. Combine news title and article text
5. Split the data into training and testing sets
6. Convert text into TF-IDF features
7. Train machine learning classification models
8. Compare model performance
9. Evaluate the selected model
10. Save the trained model and vectorizer
11. Use the model for new predictions

## Model Performance

The project evaluated three machine learning algorithms using the same dataset split and TF-IDF features.

| Model                   | Accuracy |
| ----------------------- | -------: |
| Logistic Regression     |   98.57% |
| Multinomial Naive Bayes |   93.72% |
| Linear SVM              |   99.50% |

The Linear SVM achieved the highest test accuracy among the three evaluated models.

The evaluation used:

* Accuracy
* Precision
* Recall
* F1-score

The current Streamlit application uses the Logistic Regression model for prediction.

## Project Structure

```text
Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── venv/
│
├── train_model.py
├── compare_models.py
├── predict.py
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
pip install -r requirements.txt
```

## Train the Model

Run:

```powershell
python train_model.py
```

This will train the Logistic Regression model and create:

```text
fake_news_model.pkl
tfidf_vectorizer.pkl
```

## Compare Machine Learning Models

Run:

```powershell
python compare_models.py
```

This compares:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM

The comparison displays the accuracy achieved by each model.

## Command Line Prediction

Run:

```powershell
python predict.py
```

Enter a news article when prompted.

Type:

```text
exit
```

to stop the program.

## Streamlit Web Application

Run:

```powershell
streamlit run app.py
```

The application provides a web interface where users can enter a news article and receive:

* Prediction
* Confidence score
* Confidence-level information
* Low-confidence warning

## Important Note

This project classifies news based on patterns learned from the training dataset.

It does not independently verify news using external sources o
