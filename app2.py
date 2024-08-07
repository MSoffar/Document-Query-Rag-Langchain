import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from docx import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import openai
import asyncio

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]
# Streamlit app setup
st.title("Conversational Document Query App with FAISS")

def read_pdf(file):
    """Read a PDF file and convert it to text."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def convert_docx_to_text(file):
    """Convert a DOCX file to text."""
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def download_and_convert_pdf_to_text(pdf_url):
    """Download a PDF from a URL and convert it to text."""
    response = requests.get(pdf_url, timeout=50)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    return read_pdf(pdf_file)

def download_and_convert_docx_to_text(docx_url):
    """Download a DOCX from a URL and convert it to text."""
    response = requests.get(docx_url, timeout=50)
    response.raise_for_status()
    docx_file = BytesIO(response.content)
    return convert_docx_to_text(docx_file)

def process_documents(uploaded_files, urls):
    """Process PDF and DOCX files and URLs."""
    documents = []

    # Process uploaded files
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = read_pdf(file)
            documents.append(text)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = convert_docx_to_text(file)
            documents.append(text)

    # Process URLs
    for url in urls:
        try:
            if url.lower().endswith('.pdf'):
                text = download_and_convert_pdf_to_text(url)
                documents.append(text)
            elif url.lower().endswith('.docx'):
                text = download_and_convert_docx_to_text(url)
                documents.append(text)
        except Exception as e:
            st.error(f"Failed to process URL {url}: {e}")

    return documents

def create_embeddings_and_store(documents):
    """Create embeddings for the documents and store them in FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Split documents into smaller chunks with overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.create_documents(documents)

    # Create and store embeddings in FAISS
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# File uploader for PDF and DOCX files
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Text area for URL input
url_input = st.text_area("Enter PDF/DOCX URLs (one per line)")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process documents if they haven't been processed yet
if st.button("Process Documents"):
    urls = url_input.splitlines() if url_input else []
    documents = process_documents(uploaded_files, urls)

    if documents:
        # Create and store embeddings
        st.session_state.vector_store = create_embeddings_and_store(documents)
        st.success("Documents processed and embeddings created.")
    else:
        st.warning("No documents to process.")

# Display the chat-like interface for the conversation
st.header("Chat with your Documents")
for entry in st.session_state.history:
    st.write(f"**User:** {entry['query']}")
    st.write(f"**Assistant:** {entry['response']}")

# Text input for the user's query
query = st.text_input("Please enter your query:", key="user_query")

async def get_relevant_chunks(sub_query, retriever, top_k=5):
    # Retrieve the top K most relevant chunks asynchronously
    retrieved_docs = await retriever.ainvoke(sub_query)
    return [doc.page_content for doc in retrieved_docs[:top_k]]

if st.button("Delete Chat"):
    st.session_state.history = []  # Clear the chat history

if st.button("Ask") and query:
    # Ensure vector store is available
    if st.session_state.vector_store:
        vector_store = st.session_state.vector_store
        retriever = VectorStoreRetriever(vectorstore=vector_store)

        # Split the query into sub-queries if needed
        sub_queries = query.split('?')  # Splitting by question marks as a simple heuristic

        responses = []

        # Process each sub-query
        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if sub_query:
                # Run the async task to get the relevant chunks for each sub-query
                top_chunks = asyncio.run(get_relevant_chunks(sub_query, retriever, top_k=5))

                if top_chunks:
                    # Create a prompt using the retrieved chunks
                    system_prompt = (
                        "You are a helpful assistant. You are given a set of text chunks from documents. "
                        "Please find the most relevant information based on the question below, "
                        "using only the provided chunks. Don't use your own knowledge! "
                        "In your response, Be complete, concise and accurate and answer the query well!."
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

        # Join and display all responses
        final_response = "\n\n".join(responses)
        st.write("Refined Response:", final_response)

        # Save conversation history
        st.session_state.history.append({"query": query, "response": final_response})

        # Clear the input box after processing
        query = ""  # Reset the query input value
    else:
        st.warning("Please process documents before querying.")
