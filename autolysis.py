import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from io import BytesIO
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN is not set. Please set the token as an environment variable.")
    exit(1)

def query_llm_analysis(dataset_summary):
    dataset_summary_str = dataset_summary.head(10).to_string()

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json',
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that analyzes datasets and provides insights."
            },
            {
                "role": "user",
                "content": f"Here is the dataset summary: {dataset_summary_str}. Can you analyze it and provide insights?"
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        analysis_results = response.json()
        return analysis_results['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def create_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='number').dropna()
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    correlation_plot_path = Path("correlation_heatmap.png")
    plt.savefig(correlation_plot_path)
    plt.close()
    return correlation_plot_path

def generate_readme(df, analysis_results, correlation_plot_path):
    with open("README.md", "w", encoding="utf-8") as file:
        file.write("# Dataset Analysis Report\n")
        file.write("## Data Summary\n")
        file.write(f"```\n{df.describe()}\n```\n")
        file.write("## Insights from AI Analysis\n")
        file.write(f"{analysis_results}\n")
        file.write("## Data Visualizations\n")
        file.write(f"![Correlation Heatmap]({correlation_plot_path})\n")

def read_csv_with_encodings(file_path):
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']
    for encoding in encodings_to_try:
        try:
            print(f"Trying to read the file with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            print(f"Failed to read the file with encoding: {encoding}")
    print("Error: Could not read the file with any of the tried encodings.")
    exit(1)

def main(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return

    df = read_csv_with_encodings(file_path)

    correlation_plot_path = create_correlation_heatmap(df)
    print(f"Correlation Heatmap saved at: {correlation_plot_path}")

    analysis_results = query_llm_analysis(df)

    generate_readme(df, analysis_results, correlation_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dataset analysis')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    args = parser.parse_args()

    main(args.file_path)