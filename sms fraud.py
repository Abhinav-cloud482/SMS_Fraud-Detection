import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")  # Ensure correct encoding
df = df[['v1', 'v2']]  # Keep necessary columns
df.columns = ['label', 'message']

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # Spam = 1, Ham = 0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# SVM Classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Evaluate models
nb_acc = accuracy_score(y_test, nb_model.predict(X_test_tfidf))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_tfidf))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_tfidf))

print(f"Naïve Bayes Accuracy: {nb_acc:.2f}")
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"SVM Accuracy: {svm_acc:.2f}")

# Spam Prediction Function
def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = nb_model.predict(message_tfidf)[0]
    return "Spam" if prediction else "Ham"

# Example Prediction
example_message = "Congratulations! You've won a free lottery. Click here to claim."
print(f"Prediction: {predict_spam(example_message)}")
