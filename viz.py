from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the LLM
try:
    print("Loading LLM model...")
    model_name = "google/flan-t5-small"  # Lightweight model for CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("LLM model loaded successfully.")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    model = None

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Your exact code begins here
def process_file(file_path):
    """Load the CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("File loaded successfully!")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def analyze_data(df):
    """Analyze the dataset to personalize insights and visualizations."""
    analysis = {
        "categorical": [],
        "numeric": [],
        "correlations": {},
        "outliers": {}
    }

    # Separate categorical and numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            analysis["categorical"].append(col)
        else:
            analysis["numeric"].append(col)

    # Correlation analysis
    if len(analysis["numeric"]) > 1:
        correlations = df[analysis["numeric"]].corr()
        analysis["correlations"] = correlations

    # Outlier detection using IQR
    for col in analysis["numeric"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        analysis["outliers"][col] = outliers

    return analysis

def suggest_and_generate_visualizations(df, analysis):
    """Generate personalized visualizations and save them to static folder."""
    print("\nGenerating Visualizations...")
    chart_files = []  # List to store generated chart filenames

    # Ensure the static folder exists
    static_folder = "static"
    os.makedirs(static_folder, exist_ok=True)

    # Bar chart for categorical vs numeric
    if analysis["categorical"] and analysis["numeric"]:
        filename = os.path.join(static_folder, "bar_chart.png")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x=analysis["categorical"][0], y=analysis["numeric"][0])
        plt.title(f"{analysis['numeric'][0]} by {analysis['categorical'][0]}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        chart_files.append(filename)

    # Scatter plot for numeric correlations
    if len(analysis["numeric"]) > 1:
        filename = os.path.join(static_folder, "scatter_plot.png")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=analysis["numeric"][0], y=analysis["numeric"][1])
        plt.title(f"{analysis['numeric'][1]} vs {analysis['numeric'][0]}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        chart_files.append(filename)

    # Distribution plots for numeric columns
    for col in analysis["numeric"]:
        filename = os.path.join(static_folder, f"distribution_{col}.png")
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        chart_files.append(filename)

    # Return list of generated chart file paths
    return chart_files

def generate_insights_with_llm(df, analysis):
    """Generate insights using LLM."""
    if model is None:
        return "LLM model not available."
    
    try:
        # Prepare the prompt
        prompt = f"Analyze the following dataset and generate key insights:\n{df.describe(include='all').to_string()}"
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7)
        insights = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nInsights from LLM:")
        print(insights)
        return insights
    except Exception as e:
        print(f"Error generating insights: {e}")
        return "Error generating insights."

def answer_question_with_llm(question, df):
    """Answer user questions about the data using LLM."""
    if model is None:
        return "LLM model not available."
    
    try:
        # Prepare the prompt
        prompt = f"Answer the following question based on this dataset:\n{df.describe(include='all').to_string()}\nQuestion: {question}"
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error answering question: {e}"

# Flask Routes
@app.route("/")
def home():
    return "VizAI API is running!"

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Process and analyze the file
    df = process_file(file_path)
    if df is None:
        return jsonify({"error": "Failed to process the file"}), 500

    analysis = analyze_data(df)
    suggest_and_generate_visualizations(df, analysis)
    insights = generate_insights_with_llm(df, analysis)

    return jsonify({
        "insights": insights,
        "charts": [
            "static/bar_chart.png",
            "static/scatter_plot.png"
        ]
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    file_path = data.get("file_path")  # Path to the uploaded file

    if not question or not file_path:
        return jsonify({"error": "Invalid request"}), 400

    df = process_file(file_path)
    if df is None:
        return jsonify({"error": "Failed to process the file"}), 500

    answer = answer_question_with_llm(question, df)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
