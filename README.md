# Capstone-GenAI-for-Good

# Veracity Machine

The **Veracity Machine** is a tool built using [Mesop](https://google.github.io/mesop/), designed to assess the veracity of news articles. Users can upload a news article in PDF format, and then input a prompt. The system extracts the text from the article, combines it with the user prompt, and checks the veracity of the content by sending the combined input to Google's **Gemini Pro** AI model. The AI analyzes the content and returns an assessment or generated response based on the input.

## Purpose
The **Veracity Machine** helps users fact-check and analyze the truthfulness of content from various news articles. It can assist in:
- Fact-checking political statements.
- Analyzing news articles for bias.
- Verifying claims made in media outlets.

## Environment Creation
```bash
conda create --name <env_name> --file requirements.txt
```

## How to run
```python
mesop veracity_machine.py
```

## Features
- **PDF Upload**: Upload news articles or other content in PDF format.
- **Text Extraction**: Extracts the full text from uploaded PDF files using `PyPDF2`.
- **Prompt Combination**: Combines user input (prompts) with the extracted text from the PDF for analysis.
- **Google Generative AI Integration**: Uses Google's **Gemini Pro 1.5** model to assess the veracity of the combined input and article.
- **ChromaDB Integration**: Allows for storing and querying the extracted data from articles using ChromaDB.

## Technologies Used
- **Mesop**: For creating a user-friendly interface.
- **Google Generative AI (Gemini Pro 1.5)**: For analyzing and generating content based on the prompt and article text.
- **ChromaDB**: To store and retrieve article data.
- **PyPDF2**: For extracting text from uploaded PDFs.
- **Python**: Core programming language for logic and processing.

## Installation

### Requirements
- Python 3.7 or above
- API key for [Google Generative AI](https://developers.generativeai.google/cloud).

### Libraries
Install the required Python libraries using:

```bash
pip install mesop google-generativeai chromadb PyPDF2
```

## Pipeline

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>/src
   ```

2. **Set up your environment variables:**
    In the root directory, create a `.env` file and add your Google API key.

3. **Add more factuality factors for Predictive AI if you want:**
   ```bash
   cd PredictiveAI
   ```
   Change the data path or add more factors, and then:
   ```bash
   bash run.sh
   ```
   In order to get datasets with factors as features and average score of different sets.

4. **Run the app:**
    ```python
    mesop veracity_machine.py
    ```

## Usage

1. **Upload a PDF:**  
   Use the "Upload PDF" button to upload a news article or report in PDF format.

2. **Enter a Prompt:**  
   After the PDF is uploaded and its text is extracted, enter a prompt, such as a question or instruction to analyze the article.

3. **Combined Input:**  
   The Veracity Machine will combine the extracted PDF text with your prompt for fact-checking.

4. **Generate Response:**  
   The combined input is sent to the Gemini Pro AI, and the generated analysis or content is displayed in the output section.

5. **Store and Query Articles:**  
   Store articles and retrieve them using ChromaDB for future reference.


## src Folder Contents
- **PredictiveAI/**: Includes scripts and modules for predictive analysis and modeling.
- **chroma_db/**: Stores structured data embeddings and database files for ChromaDB management.
- **saved_models/**: Contains optimized pre-trained models with reduced storage requirements.
- **2024_polifact.py**: Script for scraping 2024 polifacts data.
- **main.py**: Initial Mesop outline with foundational project functionality
- **news.pdf**: Sample dataset of political news in PDF format for model evaluation.
- **prediction_engine.py**: Core machine learning pipeline for building and evaluating predictive models.
- **utils.py**: Provides utility and helper functions for function calling.
- **veracity_machine.py**: Main script integrating PredictiveAI and GenAI models functionalities.