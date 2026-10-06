import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")


# Add labels
# 0 = Fake News
# 1 = Real News
fake_news["label"] = 0
real_news["label"] = 1


# Combine datasets
news_data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)


# Create combined content
news_data["content"] = (
    news_data["title"].fillna("")
    + " "
    + news_data["text"].fillna("")
)


# Features and target
X = news_data["content"]
y = news_data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC()
}


print("\n======================================")
print("       MODEL COMPARISON")
print("======================================\n")


# Train and evaluate models
for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy * 100:.2f}%")


print("\n======================================")
print("       COMPARISON COMPLETED")
print("======================================")