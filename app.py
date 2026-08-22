from flask import Flask, render_template, request
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ==============================
# LOAD DATASET
# ==============================

DATASET_PATH = "combined_dataset.csv"

try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("\nERROR: combined_dataset.csv not found!")
    print("Make sure combined_dataset.csv is inside the spam_detection folder.")
    exit()


# ==============================
# CHECK COLUMNS
# ==============================

print("Dataset columns:", list(df.columns))

if "target" not in df.columns or "text" not in df.columns:
    print("\nERROR: Dataset must contain:")
    print("target")
    print("text")
    exit()


# ==============================
# CONVERT LABELS
# ==============================

df["target"] = df["target"].map({
    "ham": 0,
    "spam": 1
})

df.dropna(subset=["text", "target"], inplace=True)


# ==============================
# PREPROCESSING
# ==============================

def preprocess_email(text):

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


print("\nPreprocessing messages...")

df["clean_message"] = df["text"].apply(
    preprocess_email
)


# ==============================
# TF-IDF
# ==============================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    df["clean_message"]
)

y = df["target"]


# ==============================
# TRAIN MODEL
# ==============================

print("Training Naive Bayes model...")

model = MultinomialNB()

model.fit(X, y)


print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")
print("Total messages:", len(df))
print("Features:", X.shape[1])
print("================================\n")


# ==============================
# PREDICTION
# ==============================

def predict_message(message):

    cleaned = preprocess_email(message)

    if not cleaned.strip():
        return 0

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(vector)[0]

    return prediction


# ==============================
# FLASK ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        message = request.form.get(
            "message",
            ""
        )

        if not message.strip():

            prediction = "Please enter a message."

        else:

            result = predict_message(message)

            if result == 1:
                prediction = "SPAM"
            else:
                prediction = "HAM (Not Spam)"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


# ==============================
# START WEBSITE
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )