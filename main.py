from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the JSON file containing questions and answers
try:
    with open(r"C:\Users\15732\OneDrive\Desktop\questions_and_answers.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'questions_and_answers.json' file not found.")
    data = []

# Convert JSON data to DataFrame for easier processing
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Check if DataFrame is empty
if not df.empty:
    # Fit and transform the questions to create TF-IDF vectors
    question_vectors = vectorizer.fit_transform(df["prompt"])
else:
    print("Error: No data found in JSON file.")

# Function to find the closest matching question and return its answer
def get_answer(user_question, df, question_vectors):
    user_question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    closest_idx = similarities.argmax()
    answer = df.iloc[closest_idx]["response"]
    return answer

# Root route to confirm API is running
@app.route('/')
def home():
    return "Chatbot API is running!"

# Define a route for chatbot responses
@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_question = request.json.get("question")
    if user_question:
        answer = get_answer(user_question, df, question_vectors)
        return jsonify({"response": answer})
    else:
        return jsonify({"response": "Please provide a question."}), 400

if __name__ == "__main__":
    app.run(debug=False, port=5002)
