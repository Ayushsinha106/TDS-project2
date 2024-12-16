# Importing necessary libraries
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

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API Proxy token from the environment variables
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Check if the API token is set; exit the program if not found
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN is not set. Please set the token as an environment variable.")
    exit(1)

# Function to query the language model for dataset analysis insights
def query_llm_analysis(dataset_summary):
    # Convert the first 10 rows of the dataset summary to a string format
    dataset_summary_str = dataset_summary.head(10).to_string()

    # Define the API endpoint for the language model
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Set up the headers for the API request, including authorization
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json',
    }

    # Create the payload for the API request
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

    # Make the API request to the language model
    response = requests.post(url, headers=headers, json=data)

    # If the request is successful, return the analysis results
    if response.status_code == 200:
        analysis_results = response.json()
        print(analysis_results,"analysis_results")
        return analysis_results['choices'][0]['message']['content']
    else:
        # Print an error message if the request fails
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to create and save a correlation heatmap
def create_correlation_heatmap(df):
    # Select only numeric columns and drop rows with missing values
    numeric_df = df.select_dtypes(include='number').dropna()

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)

    # Save the heatmap to a file
    correlation_plot_path = Path("correlation_heatmap.png")
    plt.savefig(correlation_plot_path)
    plt.close()

    return correlation_plot_path


# Function to generate a README file summarizing the dataset analysis
def generate_readme(df, analysis_results, correlation_plot_path):
    # Open the README file in write mode
    with open("README.md", "w", encoding="utf-8") as file:
        # Add a title and data summary section
        file.write("# Dataset Analysis Report\n")
        file.write("## Data Summary\n")
        file.write(f"```\n{df.describe()}\n```\n")

        # Add insights from the AI analysis
        file.write("## Insights from AI Analysis\n")
        file.write(f"{analysis_results}\n")

        # Add a section for data visualizations
        file.write("## Data Visualizations\n")
        file.write(f"![Correlation Heatmap]({correlation_plot_path})\n")
# Function to read a CSV file with various encodings
def read_csv_with_encodings(file_path):
    # List of encodings to try
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16']

    # Attempt to read the file with each encoding
    for encoding in encodings_to_try:
        try:
            print(f"Trying to read the file with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            # Print a message if the current encoding fails
            print(f"Failed to read the file with encoding: {encoding}")

    # Exit the program if no encoding works
    print("Error: Could not read the file with any of the tried encodings.")
    exit(1)

# Main function to orchestrate the dataset analysis process
def main(file_path):
    # Check if the specified file exists
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} does not exist.")
        return

    # Read the dataset
    df = read_csv_with_encodings(file_path)

    # Generate and save the correlation heatmap
    print(df)
    correlation_plot_path = create_correlation_heatmap(df)
    print(f"Correlation Heatmap saved at: {correlation_plot_path}")

    # Query the AI model for dataset analysis insights
    analysis_results = query_llm_analysis(df)

    # Generate the README report with the analysis results and visualizations
    generate_readme(df, analysis_results, correlation_plot_path, bar_chart)

# Entry point of the script
if __name__ == "__main__":
    # Set up argument parsing for the file path input
    parser = argparse.ArgumentParser(description='Run dataset analysis')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the main function with the provided file path
    main(args.file_path)
