import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# Load datasets
# ==========================================

fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")


# ==========================================
# Add labels
# ==========================================

# 0 = Fake News
# 1 = Real News

fake_news["label"] = 0
real_news["label"] = 1


# ==========================================
# Combine datasets
# ==========================================

news_data = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)


# ==========================================
# Create combined content
# ==========================================

news_data["content"] = (
    news_data["title"].fillna("")
    + " "
    + news_data["text"].fillna("")
)


# ==========================================
# Features and target
# ==========================================

X = news_data["content"]
y = news_data["label"]


# ==========================================
# Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# Linear SVM
# ==========================================

svm_model = LinearSVC()

model = CalibratedClassifierCV(
    svm_model,
    cv=3
)


# ==========================================
# Train model
# ==========================================

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======================================")
print("       LINEAR SVM MODEL")
print("======================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# ==========================================
# Save model
# ==========================================

joblib.dump(
    model,
    "fake_news_svm_model.pkl"
)


# ==========================================
# Save vectorizer
# ==========================================

joblib.dump(
    vectorizer,
    "tfidf_svm_vectorizer.pkl"
)


print("\nModel and vectorizer saved successfully!")

print("Created: fake_news_svm_model.pkl")
print("Created: tfidf_svm_vectorizer.pkl")