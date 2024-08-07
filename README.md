# Conversational Document Query App

This project implements a conversational document query application using Langchain and Streamlit. It enables users to upload PDF or DOCX documents and query them using a Retrieval-Augmented Generation (RAG) model powered by OpenAI's GPT-4o-mini. The application retrieves relevant chunks of text from the documents and generates comprehensive answers to user queries.

## Features

- **PDF and DOCX Support**: Upload and process PDF and DOCX files.
- **Advanced Query Handling**: Split and handle multi-part queries, retrieving relevant information from multiple document chunks.
- **Interactive Interface**: Engage with the application via a chat-like interface, complete with conversation history.
- **Deployment Ready**: Built to be deployed on Streamlit Cloud for easy online access.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MSoffar/Document-Query-Rag-Langchain
   cd Document-Query-Rag-Langchain

2.Create and activate a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   pip install -r requirements.txt

Usage
 1.Run the Streamlit application:
      streamlit run app2.py
 2.Upload Documents: Use the interface to upload PDF or DOCX files.

 3.Enter Queries: Type your queries in the provided input field to get responses based on the uploaded documents.

Configuration
  OpenAI API Key
      The application requires an OpenAI API key to function. Store the API key in a secrets.toml file in the following format:
           [openai]
           api_key = "sk-xxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key

Streamlit Secrets
If deploying on Streamlit Cloud, manage your secrets through the Streamlit interface:

  Go to your app’s settings on Streamlit Cloud.
  Add your API key in the Secrets section using the same format as above.
  Deployment
  Push your code to a GitHub repository.
  Deploy to Streamlit Cloud:
  Sign in to Streamlit Cloud with your GitHub account.
  Click on New App, select your repository, branch, and the app script file.
  Configure any secrets if necessary and deploy your app.

  ### Key Sections:

- **Features**: Highlights the main features of your application.
- **Installation and Usage**: Provides step-by-step instructions for setting up and using the application locally.
- **Configuration**: Details how to set up the OpenAI API key using Streamlit secrets.
- **Deployment**: Guides users through deploying the app on Streamlit Cloud.
