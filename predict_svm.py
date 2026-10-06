import joblib


# Load the trained SVM model
model = joblib.load("fake_news_svm_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_svm_vectorizer.pkl")


print("===================================")
print("       FAKE NEWS DETECTOR - SVM")
print("===================================")

print("\nEnter a news article below.")
print("Type 'exit' to stop the program.\n")


while True:

    news = input("Enter news: ")

    if news.lower() == "exit":
        print("\nProgram ended.")
        break

    if not news.strip():
        print("Please enter some news text.\n")
        continue

    # Convert news into TF-IDF features
    news_tfidf = vectorizer.transform([news])

    # Make prediction
    prediction = model.predict(news_tfidf)[0]

    # Get calibrated probabilities
    probabilities = model.predict_proba(news_tfidf)[0]

    # Display result
    if prediction == 0:

        confidence = probabilities[0] * 100

        print("\nPrediction: FAKE NEWS")
        print(f"Confidence: {confidence:.2f}%")

    else:

        confidence = probabilities[1] * 100

        print("\nPrediction: REAL NEWS")
        print(f"Confidence: {confidence:.2f}%")

    print("\n-----------------------------------\n")