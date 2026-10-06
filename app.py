import streamlit as st
import joblib


# ==========================================
# Load SVM model and vectorizer
# ==========================================

model = joblib.load("fake_news_svm_model.pkl")
vectorizer = joblib.load("tfidf_svm_vectorizer.pkl")


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ==========================================
# Header
# ==========================================

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the machine learning "
    "model will classify it as Fake News, Real News, or Uncertain."
)

st.markdown("---")


# ==========================================
# News input
# ==========================================

news_text = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste the complete news article here..."
)


# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Detect News", use_container_width=True):

    if not news_text.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        # Convert text into TF-IDF features
        news_tfidf = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(news_tfidf)[0]

        # Get calibrated probabilities
        probabilities = model.predict_proba(news_tfidf)[0]

        # Get confidence of predicted class
        confidence = probabilities[prediction] * 100

        # ==================================
        # Uncertain prediction
        # ==================================

        if confidence < 60:

            st.warning("⚠️ Prediction: UNCERTAIN")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            st.warning(
                "The model is uncertain about this article. "
                "The text may not contain enough patterns for "
                "a reliable classification."
            )

        # ==================================
        # Fake News
        # ==================================

        elif prediction == 0:

            st.error("⚠️ Prediction: FAKE NEWS")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            if confidence < 80:

                st.info(
                    "ℹ️ Moderate confidence prediction. "
                    "Consider verifying the information using "
                    "reliable sources."
                )

            else:

                st.info(
                    "The model has relatively high confidence "
                    "in this prediction."
                )

        # ==================================
        # Real News
        # ==================================

        else:

            st.success("✅ Prediction: REAL NEWS")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            if confidence < 80:

                st.info(
                    "ℹ️ Moderate confidence prediction. "
                    "Consider verifying the information using "
                    "reliable sources."
                )

            else:

                st.info(
                    "The model has relatively high confidence "
                    "in this prediction."
                )


# ==========================================
# About the project
# ==========================================

st.markdown("---")

st.subheader("📌 About This Project")

st.write(
    "This project uses Natural Language Processing (NLP), "
    "TF-IDF feature extraction, and Linear SVM "
    "to classify news articles."
)

st.write(
    "The model was trained using a dataset containing "
    "labeled fake and real news articles."
)


# ==========================================
# Model information
# ==========================================

st.subheader("🤖 Model Information")

st.write(
    "Model: Calibrated Linear SVM"
)

st.write(
    "Test Accuracy: 99.45%"
)


# ==========================================
# Prediction information
# ==========================================

st.subheader("📊 Prediction Information")

st.write(
    "Predictions with confidence below 60% are shown as "
    "UNCERTAIN because the model may not have enough "
    "evidence to make a reliable classification."
)


# ==========================================
# Important note
# ==========================================

st.warning(
    "⚠️ This system predicts based on patterns learned "
    "from the training dataset. It does not independently "
    "verify news against external sources."
)