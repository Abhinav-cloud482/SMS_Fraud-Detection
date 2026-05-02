# SMS_Fraud-Detection
A machine learning-based SMS spam detection system using TF-IDF and multiple classifiers (Naïve Bayes, Logistic Regression, and SVM) for accurate text classification.


# 📩 SMS Spam Detection using Machine Learning

A complete machine learning pipeline to detect SMS spam messages using Natural Language Processing (NLP) techniques and multiple classification algorithms including Naïve Bayes, Logistic Regression, and Support Vector Machine (SVM).

---

## 🚀 Features

* Preprocessing of SMS text data
* TF-IDF vectorization for feature extraction
* Multiple ML models:

  * Naïve Bayes
  * Logistic Regression
  * Support Vector Machine (SVM)
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
  * Classification Report
  * ROC-AUC Score
* Cross-validation support
* Custom spam prediction function

---

## 📂 Project Structure

```
├── sms fraud.py        # Basic training and prediction script
├── testing.py          # Advanced evaluation and testing script
├── spam.csv            # Dataset (SMS Spam Collection)
└── README.md           # Project documentation
```

---

## 📊 Dataset

The dataset contains labeled SMS messages:

* **ham** → Legitimate messages
* **spam** → Fraudulent or promotional messages

### Example:

| Label | Message                          |
| ----- | -------------------------------- |
| ham   | Go until jurong point...         |
| spam  | WINNER!! You have won a prize... |

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

* Load dataset using Pandas
* Select relevant columns
* Encode labels (Ham = 0, Spam = 1)

### 2. Train-Test Split

* 80% training / 20% testing
* Stratified sampling (in `testing.py`)

### 3. Feature Extraction

* TF-IDF Vectorizer
* Removal of English stopwords

### 4. Model Training

* Multinomial Naïve Bayes
* Logistic Regression
* Support Vector Machine (Linear Kernel)

---

## 📈 Model Performance

Typical outputs include:

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-score
* ROC-AUC Score (where applicable)

---

## 🔍 Example Output

```
Naïve Bayes Accuracy: 0.98
Logistic Regression Accuracy: 0.97
SVM Accuracy: 0.98

Prediction: Spam
```

---

## 🧪 Example Prediction

```python
example_message = "Congratulations! You've won a free lottery. Click here to claim."
print(predict_spam(example_message))
```

**Output:**

```
Spam
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn nltk
```

---

## ▶️ Usage

### Run basic model:

```bash
python "sms fraud.py"
```

### Run advanced evaluation:

```bash
python testing.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK

---

## 📌 Future Improvements

* Add deep learning models (LSTM, BERT)
* Deploy as a web application (Flask/Streamlit)
* Real-time SMS filtering API
* Improve preprocessing with stemming/lemmatization

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Your Name**
GitHub: https://github.com/your-username
