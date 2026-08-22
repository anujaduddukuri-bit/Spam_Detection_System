# 📧 Email Spam Detection Using Machine Learning

A machine learning-based text classification project that automatically identifies emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and supervised learning techniques.

## 📌 Project Overview

Email spam is a common problem where unwanted, misleading, or potentially harmful messages are sent to users.

This project develops a basic **Email Spam Detection System** that analyzes the text content of an email and predicts whether the message is:

* 🚨 **Spam** — unwanted or suspicious email
* ✅ **Ham** — legitimate email

The project demonstrates the complete machine learning workflow, including **data preprocessing, text feature extraction, model training, evaluation, and prediction**.

---

## 🎯 Objectives

* Build a text classification model for spam detection.
* Clean and preprocess email text using NLP techniques.
* Convert textual data into numerical features.
* Train a machine learning classification model.
* Evaluate the model using standard classification metrics.
* Predict whether new messages are spam or legitimate.

---

## 🧠 Machine Learning Approach

The project follows these major steps:

```text
Email Dataset
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Feature Extraction
      ↓
Train/Test Split
      ↓
Machine Learning Model
      ↓
Model Evaluation
      ↓
Spam / Ham Prediction
```

---

## 🔤 Text Preprocessing

The following preprocessing techniques are applied to the email text:

1. Convert text to lowercase.
2. Remove unnecessary punctuation.
3. Remove special characters.
4. Remove stop words.
5. Tokenize text where required.
6. Remove unnecessary whitespace.
7. Convert cleaned text into numerical features.

Example:

```text
Original:
"Congratulations!!! You have WON a FREE prize. Click NOW!"

After preprocessing:
"congratulations won free prize click"
```

---

## 🔢 Feature Extraction

Since machine learning models cannot directly process raw text, the email messages are converted into numerical representations.

### TF-IDF

**Term Frequency–Inverse Document Frequency (TF-IDF)** is used to represent the importance of words within the email dataset.

TF-IDF gives higher importance to words that are useful for distinguishing between spam and legitimate messages.

Example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(messages)
```

---

## 🤖 Machine Learning Model

A supervised machine learning classifier is trained using the extracted text features.

### Logistic Regression

Logistic Regression can be used as the primary classification algorithm because it performs well for many binary text-classification problems.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
```

The trained model predicts one of two classes:

```text
0 → Ham
1 → Spam
```

---

## 📊 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example:

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Why These Metrics Matter

**Precision:** Measures how many emails predicted as spam are actually spam.

**Recall:** Measures how many actual spam emails were successfully detected.

**F1-Score:** Provides a balance between precision and recall.

---

## 📂 Dataset

The project uses an email/SMS spam classification dataset containing text messages and their corresponding labels.

### Dataset Columns

| Column    | Description        |
| --------- | ------------------ |
| `label`   | Spam or Ham        |
| `message` | Email/message text |

Example:

| Label | Message                                  |
| ----- | ---------------------------------------- |
| ham   | "Hey, are we meeting today?"             |
| spam  | "Congratulations! You won a free prize!" |

---

## 🛠️ Technologies Used

| Category             | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Machine Learning     | Scikit-learn                    |
| NLP                  | Natural Language Processing     |
| Data Processing      | Pandas, NumPy                   |
| Feature Extraction   | TF-IDF                          |
| Visualization        | Matplotlib, Seaborn             |
| Development          | Jupyter Notebook / Google Colab |

---

## 📦 Libraries

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📂 Project Structure

```text
email-spam-detection/
│
├── spam_detection.ipynb
├── spam.csv
├── requirements.txt
├── README.md
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
└── screenshots/
    └── prediction.png
```

> Modify the structure if your actual project uses different filenames.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/email-spam-detection.git
```

### 2. Navigate to the Project

```bash
cd email-spam-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the Notebook

```bash
jupyter notebook
```

Open:

```text
spam_detection.ipynb
```

Run all the cells sequentially.

---

## 🧪 Example Predictions

### Example 1

**Input:**

```text
Congratulations! You have won a $1,000 cash prize. Claim now!
```

**Prediction:**

```text
🚨 Spam
```

### Example 2

**Input:**

```text
Hi Anuja, can you send me the project report by tomorrow?
```

**Prediction:**

```text
✅ Ham
```

### Example 3

**Input:**

```text
You have been selected for a special offer. Click the link to claim your reward.
```

**Prediction:**

```text
🚨 Spam
```

---

## 💡 Key Features

* 📧 Email/text classification
* 🧹 NLP-based text preprocessing
* 🔤 TF-IDF feature extraction
* 🤖 Machine learning classification
* 📊 Model performance evaluation
* 🔍 Real-time prediction for new messages
* 📈 Confusion matrix and classification metrics

---

## 🔮 Future Enhancements

* Develop a **Flask or Streamlit web application**.
* Add multiple machine learning algorithms.
* Compare Logistic Regression, Naive Bayes, SVM, and Random Forest.
* Use advanced NLP techniques.
* Add email header and metadata analysis.
* Implement deep learning models such as LSTM or BERT.
* Deploy the model as an online spam detection service.
* Add an interactive dashboard for prediction results.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Natural Language Processing
* Text preprocessing
* Feature engineering
* TF-IDF vectorization
* Supervised machine learning
* Binary classification
* Model evaluation
* Python data analysis
* Building an end-to-end ML pipeline

---

## 👩‍💻 Author

**DUDDUKURI ANUJA**

B.Tech Student | AI/ML Enthusiast

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Data Science
* Python
* Deep Learning

---

## ⭐ Future Goal

The goal of this project is to demonstrate how machine learning and NLP can be applied to solve a practical cybersecurity and communication problem by automatically filtering unwanted messages.

---

## 📄 License

This project is developed for **educational and research purposes**.
