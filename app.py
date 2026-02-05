from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
