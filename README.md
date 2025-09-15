# Classification Project

**Deployed App:** [https://textreviewsclassification.streamlit.app/](https://textreviewsclassification.streamlit.app/)

## Overview
This project is a **text and data classification pipeline** built in Python. It focuses on predicting ratings for baby products using machine learning. The project includes:

- Data preprocessing and cleaning
- Feature extraction using **TF-IDF**
- Model training with **Naive Bayes**
- Semantic analysis with **spaCy**
- A **Streamlit app** for easy interaction and prediction

---

## Features

- **Data Preprocessing:** Handles missing values, text normalization, and encoding.
- **Feature Engineering:** TF-IDF vectorization of text data.
- **Modeling:** Naive Bayes classifier for product rating prediction.
- **Interactive UI:** Streamlit app to make predictions in real-time.
- **Semantic Search:** Uses spaCy embeddings for advanced text analysis.

---
## Installation

Clone the repository, create a virtual environment, install dependencies, download the spaCy model, and run the Streamlit app all in one go:

```bash
# Clone the repo
git clone https://github.com/username/repo.git
cd repo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download spaCy language model
python -m spacy download en_core_web_lg

# Run the Streamlit app locally
streamlit run app.py
