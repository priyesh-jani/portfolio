from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)

# Load the JSON file containing questions and answers
try:
    with open("questions_and_answers.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'questions_and_answers.json' file not found.")
    data = []

df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()

# Check if DataFrame is empty
if not df.empty:
    question_vectors = vectorizer.fit_transform(df["prompt"])
else:
    print("Error: No data found in JSON file.")
    question_vectors = None

threshold = 0.4  # Adjust as needed

def get_predefined_answer(user_question, df, question_vectors):
    # If no Q&A data, return None
    if question_vectors is None:
        return None

    user_question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    closest_idx = similarities.argmax()
    best_score = similarities[0, closest_idx]

    # If best_score is below threshold, return None to fallback to LLM
    if best_score < threshold:
        return None
    return df.iloc[closest_idx]["response"]

# Load distilgpt2 model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
##model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True))

def get_answer_from_llm(user_question):
    inputs = tokenizer.encode(user_question, return_tensors="pt")
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

@app.route('/')
def home():
    return "Chatbot API is running!"

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_question = request.json.get("question")
    if user_question:
        # First try predefined Q&A
        predefined_answer = get_predefined_answer(user_question, df, question_vectors)
        if predefined_answer:
            return jsonify({"response": predefined_answer})
        else:
            # Fallback to LLM
            llm_answer = get_answer_from_llm(user_question)
            return jsonify({"response": llm_answer})
    else:
        return jsonify({"response": "Please provide a question."}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
