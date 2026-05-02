import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # Ham = 0, Spam = 1

# Check class distribution
print("Class Distribution:\n", df['label'].value_counts(normalize=True), "\n")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)
svm_model = SVC(kernel='linear', probability=True)

nb_model.fit(X_train_tfidf, y_train)
lr_model.fit(X_train_tfidf, y_train)
svm_model.fit(X_train_tfidf, y_train)

# Evaluate models
models = {
    "Naïve Bayes": nb_model,
    "Logistic Regression": lr_model,
    "SVM": svm_model
}

for name, model in models.items():
    preds = model.predict(X_test_tfidf)
    proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    if proba is not None:
        print(f"ROC AUC Score: {roc_auc_score(y_test, proba):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# Cross-validation example (optional)
print("\nCross-Validation (Naïve Bayes):")
cv_scores = cross_val_score(nb_model, vectorizer.transform(df['message']), df['label'], cv=5)
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

# Spam prediction function
def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = nb_model.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# Example prediction
example_message = "Congratulations! You've won a free lottery. Click here to claim."
print("\nExample Prediction:")
print(f"Message: {example_message}")
print(f"Prediction: {predict_spam(example_message)}")
