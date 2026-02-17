# ==============================
# Twitter Airline Sentiment Analysis
# ==============================

# --------- Import Libraries ---------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud

# --------- Load Dataset ---------
# Use forward slash OR raw string in Windows
df = pd.read_csv("C:/Users/vansh/Downloads/Tweets.csv", nrows=1000)

print("✅ Dataset Loaded Successfully!\n")
print(df[['airline', 'airline_sentiment', 'text']].head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# --------- Exploratory Data Analysis ---------

# Sentiment Distribution
plt.figure(figsize=(7,5))
sns.countplot(x='airline_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Airline-wise Sentiment
plt.figure(figsize=(10,6))
sns.countplot(x='airline', hue='airline_sentiment', data=df)
plt.xticks(rotation=30)
plt.title("Airline-wise Sentiment Count")
plt.show()

# Negative Reasons
plt.figure(figsize=(10,6))
negative_df = df[df['airline_sentiment'] == 'negative']
sns.countplot(
    y='negativereason',
    data=negative_df,
    order=negative_df['negativereason'].value_counts().index
)
plt.title("Reasons for Negative Sentiment")
plt.show()

# --------- Text Preprocessing ---------

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))   # remove URLs
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"#\w+", "", text)          # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)   # remove special chars
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

print("\n✅ Sample Cleaned Text:\n")
print(df[['text','clean_text']].head(3))

# --------- WordCloud ---------

neg_text = " ".join(df[df['airline_sentiment']=='negative']['clean_text'])
pos_text = " ".join(df[df['airline_sentiment']=='positive']['clean_text'])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(neg_text))
plt.axis('off')
plt.title("Negative Tweets")

plt.subplot(1,2,2)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(pos_text))
plt.axis('off')
plt.title("Positive Tweets")

plt.tight_layout()
plt.show()

# --------- Feature Engineering ---------

X = df['clean_text']
y = df['airline_sentiment']

vectorizer = TfidfVectorizer(max_features=5000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------- Model Training ---------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --------- Evaluation ---------

y_pred = model.predict(X_test_vec)

print("\n📊 Model Evaluation")
print("Train Accuracy:", round(model.score(X_train_vec, y_train)*100, 2), "%")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------- Custom Prediction Function ---------

def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

print("\n🔍 Example Predictions:")
print("1️⃣ Flight delayed for 3 hours again! →",
      predict_sentiment("Flight delayed for 3 hours again!"))

print("2️⃣ Amazing service by Delta Airlines! →",
      predict_sentiment("Amazing service by Delta Airlines!"))

print("3️⃣ Okay experience, nothing special. →",
      predict_sentiment("Okay experience, nothing special."))
print("Test Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

