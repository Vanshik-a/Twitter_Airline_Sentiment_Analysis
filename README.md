# ✈️ Twitter Airline Sentiment Analysis

## 📌 Project Overview

This project analyzes airline-related tweets and classifies them into:

- ✅ Positive  
- ❌ Negative  
- ➖ Neutral  

The goal is to understand customer feedback using Natural Language Processing (NLP) and Machine Learning.

This project demonstrates a complete Machine Learning pipeline including:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Text Preprocessing
- Feature Engineering (TF-IDF)
- Model Training (Logistic Regression)
- Model Evaluation
- Real-time Prediction Function

---

## 📂 Dataset Information

The dataset contains airline tweets with the following key columns:

- `airline` → Name of the airline  
- `text` → Tweet content  
- `airline_sentiment` → Sentiment label (Positive, Negative, Neutral)  
- `negativereason` → Reason for negative feedback  
- `user_timezone` → User timezone  

For faster execution, 1000 rows were used during training.

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Seaborn & Matplotlib (Visualization)  
- NLTK (Natural Language Processing)  
- Scikit-learn (Machine Learning)  
- WordCloud  

---

## 🔍 Exploratory Data Analysis (EDA)

The following visualizations were created:

- 📊 Sentiment Distribution
- ✈️ Airline-wise Sentiment Comparison
- ❌ Reasons for Negative Sentiment
- 🌍 Top User Timezones
- ☁️ WordCloud for Positive and Negative Tweets

EDA helps understand patterns before model building.

---

## 🧹 Text Preprocessing Steps

Raw tweets contain noise such as:
- URLs
- Mentions (@user)
- Hashtags
- Special characters

The cleaning pipeline includes:

1. Removing URLs
2. Removing mentions
3. Removing hashtags
4. Removing special characters
5. Converting text to lowercase
6. Removing English stopwords

This improves model performance significantly.

---

## 🧠 Feature Engineering

The text data is converted into numerical format using:

### TF-IDF Vectorizer

TF-IDF (Term Frequency – Inverse Document Frequency) assigns importance to words based on how relevant they are within the dataset.

This allows the machine learning model to understand textual patterns.

---

## 🤖 Model Used

### Logistic Regression

- A supervised classification algorithm
- Suitable for text classification tasks
- Efficient and interpretable

The dataset was split into:
- 80% Training Data
- 20% Testing Data

---

## 📊 Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The model achieved approximately:
Test Accuracy: 67.0 %
---

## 🔎 Example Predictions
"Flight delayed for 3 hours again!" → Negative
"Amazing service by Delta Airlines!" → Positive
"Okay experience, nothing special." → Neutral
