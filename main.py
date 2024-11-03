from flask import Flask, jsonify, render_template, request
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON file containing questions and answers
with open("questions_and_answers.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame for easier processing
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["prompt"])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask/<user_question>', methods=['GET'])
def get_answer(user_question):
    user_question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    closest_idx = similarities.argmax()
    answer = df.iloc[closest_idx]["response"]
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
