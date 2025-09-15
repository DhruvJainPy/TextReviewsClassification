import streamlit as st
import pickle
import spacy


# Load saved pipeline
with open("models/naive_bayes.pkl", "rb") as f:
    pipe = pickle.load(f)


# Page config
st.set_page_config(page_title="Baby Products Sentiment Analysis", page_icon="🍼")

st.title("🍼 Baby Products Sentiment Analysis")

# Create tabs
tab1, tab2 = st.tabs(["📖 About", "🔍 Prediction"])

# ---------------- About Section ----------------
with tab1:
    st.header("About the Project")
    st.markdown(
        """
        This project performs **Customer Sentiment Analysis** on reviews of baby products.  

        ### 📊 Dataset
        - **Size:** 6 Million reviews (sampled 1.2 Lakh for training)  
        - **Features:** Rating, Title, Review Text, Verified Purchase, Helpful Votes, etc.  
        - **Target Sentiment Mapping:**  
          - ⭐ 1-2 → Negative 😞  
          - ⭐ 3 → Neutral 😐  
          - ⭐ 4-5 → Positive 😊  

        ### 🔧 Methodology
        - **Preprocessing:** spaCy (lemmatization, stopword removal)  
        - **Feature Extraction:** TF-IDF Vectorizer (inside pipeline)  
        - **Modeling:** Naïve Bayes (baseline) + Voting Classifier (tested)  
        - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC  

        ### 📌 Insights
        - Naïve Bayes → More balanced & interpretable.  
        - Future Scope: Multimodal (Text + Image) sentiment analysis.  

        ---
        **Built with ❤️ using Streamlit, scikit-learn, and spaCy.**
        """
    )

# ---------------- Prediction Section ----------------
with tab2:
    st.header("Sentiment Prediction")
    user_input = st.text_area("✍️ Enter a customer review:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review before predicting.")
        else:

            # Predict using pipeline
            prediction = pipe.predict([user_input])[0]
            prediction_proba = pipe.predict_proba([user_input])[0]

            # Map prediction
            sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
            st.success(f"**Predicted Sentiment:** {sentiment_map[prediction]}")

            # Show probability scores
            st.subheader("Confidence Scores")
            st.write(f"Negative: {prediction_proba[0]*100:.2f}%")
            st.write(f"Neutral: {prediction_proba[1]*100:.2f}%")
            st.write(f"Positive: {prediction_proba[2]*100:.2f}%")
