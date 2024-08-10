import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import openai
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization import summarize
import asyncio
import os

# NLTK data setup
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit app setup
st.title("Conversational Document Query App with FAISS")

# Utility functions for reading and processing documents
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def convert_docx_to_text(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def download_and_convert_pdf_to_text(pdf_url):
    response = requests.get(pdf_url, timeout=50)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    return read_pdf(pdf_file)

def download_and_convert_docx_to_text(docx_url):
    response = requests.get(docx_url, timeout=50)
    response.raise_for_status()
    docx_file = BytesIO(response.content)
    return convert_docx_to_text(docx_file)

def extract_keywords(text, top_k=5):
    """Extract top-k keywords from the text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    indices = X.toarray().argsort()[0, -top_k:][::-1]
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in indices]

def generate_summary(text, ratio=0.2):
    """Generate a summary for the text using Gensim's summarize."""
    try:
        return summarize(text, ratio=ratio)
    except ValueError:
        return text  # If the text is too short to summarize, return it as is.

def split_text_into_chunks_with_titles(text, chunk_size=500):
    """Splits text into chunks, including titles, keywords, and summaries."""
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = ""
    current_title = "Untitled Section"

    for sentence in sentences:
        if sentence.strip().endswith(":"):  # Assuming titles end with a colon
            current_title = sentence.strip().rstrip(":")
            continue  # Skip adding the title sentence to the chunk

        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            # Augment the chunk before appending
            keywords = extract_keywords(current_chunk)
            summary = generate_summary(current_chunk)
            augmented_chunk = f"Title: {current_title}\n{current_chunk.strip()}\nKeywords: {', '.join(keywords)}\nSummary: {summary}"
            chunks.append(augmented_chunk)
            current_chunk = sentence + " "

    if current_chunk:
        keywords = extract_keywords(current_chunk)
        summary = generate_summary(current_chunk)
        augmented_chunk = f"Title: {current_title}\n{current_chunk.strip()}\nKeywords: {', '.join(keywords)}\nSummary: {summary}"
        chunks.append(augmented_chunk)

    return chunks

def create_embeddings_and_store(documents):
    """Create embeddings for the documents and store them in FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    all_chunks = []
    for document in documents:
        chunks = split_text_into_chunks_with_titles(document, chunk_size=500)
        all_chunks.extend(chunks)

    # Create and store embeddings in FAISS
    vector_store = FAISS.from_texts(all_chunks, embeddings)
    return vector_store

# Streamlit UI setup for file uploads and URLs
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
url_input = st.text_area("Enter PDF/DOCX URLs (one per line)")

# Initialize conversation history and vector store
if "history" not in st.session_state:
    st.session_state.history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process and store documents
if st.button("Process Documents"):
    urls = url_input.splitlines() if url_input else []
    documents = process_documents(uploaded_files, urls)

    if documents:
        st.session_state.vector_store = create_embeddings_and_store(documents)
        st.success("Documents processed and embeddings created.")
    else:
        st.warning("No documents to process.")

# Display the chat interface
st.header("Chat with your Documents")
for entry in st.session_state.history:
    st.write(f"**User:** {entry['query']}")
    st.write(f"**Assistant:** {entry['response']}")

# Handle user queries
query = st.text_input("Please enter your query:", key="user_query")

async def get_relevant_chunks(sub_query, retriever, top_k=5):
    retrieved_docs = await retriever.ainvoke(sub_query)
    return [doc.page_content for doc in retrieved_docs[:top_k]]

if st.button("Delete Chat"):
    st.session_state.history = []  # Clear the chat history

if st.button("Ask") and query:
    if st.session_state.vector_store:
        vector_store = st.session_state.vector_store
        retriever = VectorStoreRetriever(vectorstore=vector_store)

        sub_queries = query.split('?')

        responses = []

        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                top_chunks = asyncio.run(get_relevant_chunks(sub_query, retriever, top_k=5))

                if top_chunks:
                    system_prompt = (
                        "You are a helpful and knowledgeable assistant. You are given a set of text chunks from documents. "
                        "Please find the most relevant information based on the question below, "
                        "using only the provided chunks. Ensure your response is comprehensive, accurate, and informative, "
                        "covering all aspects of the question to the best of your ability. Do not reference the chunks directly. "
                        "Your goal is to provide a full and complete answer that is easy to understand and helpful to the user."
                        "Don't answer from your own knowledge, ONLY FROM CHUNKS!"
                    )
                    user_prompt = sub_query + "\n\n" + "\n\n".join(
                        f"Chunk {i + 1}: {chunk[:200]}..." for i, chunk in enumerate(top_chunks)
                    )

                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    refined_response = response.choices[0].message.content.strip()
                    responses.append(f"**{sub_query}**: {refined_response}")

                else:
                    responses.append(f"**{sub_query}**: No relevant chunks retrieved.")

        final_response = "\n\n".join(responses)
        st.write("Refined Response:", final_response)

        st.session_state.history.append({"query": query, "response": final_response})
        query = ""
    else:
        st.warning("Please process documents before querying.")
