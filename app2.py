import os
import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
import spacy

# Load the spaCy model from a local path
model_path = os.path.join(os.path.dirname(__file__), 'en_core_web_sm/en_core_web_sm-3.6.0')
nlp = spacy.load(model_path)

# Streamlit app setup
st.title("Conversational Document Query App with FAISS")

# Document reading functions
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

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Augmentation functions using spaCy
def generate_title(chunk: str) -> str:
    doc = nlp(chunk)
    noun_phrases = [np.text for np in doc.noun_chunks]

    # Use the most frequent noun phrases to generate a title
    if noun_phrases:
        title = " ".join(noun_phrases[:3])  # Join the top 3 noun phrases
        return title if title else chunk[:50]  # Fallback to first 50 characters if no noun phrases
    return chunk[:50]  # Fallback if no noun phrases

def generate_summary(chunk: str) -> str:
    doc = nlp(chunk)
    sentences = [sent.text for sent in doc.sents]

    # Calculate sentence importance based on the presence of noun phrases
    sentence_scores = {}
    for sent in sentences:
        sentence_scores[sent] = sum([1 for np in nlp(sent).noun_chunks])

    # Select top 2-3 sentences as the summary
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]
    summary = ' '.join(summary_sentences)
    return summary if summary else chunk[:100]  # Fallback to first 100 characters

def extract_keywords(chunk: str) -> list:
    r = Rake(stopwords=spacy.lang.en.stop_words.STOP_WORDS)
    r.extract_keywords_from_text(chunk)
    return r.get_ranked_phrases()[:5]  # Top 5 keywords

def augment_chunk(chunk: str) -> dict:
    title = generate_title(chunk)
    summary = generate_summary(chunk)
    keywords = extract_keywords(chunk)

    return {
        "chunk": chunk,
        "title": title,
        "summary": summary,
        "keywords": keywords,
    }

def create_embeddings_and_store(documents):
    """Create embeddings for the documents and store them in FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Split documents into smaller chunks and augment them
    all_augmented_chunks = []
    for document in documents:
        chunks = split_text_into_chunks(document, chunk_size=500)
        for chunk in chunks:
            augmented_chunk = augment_chunk(chunk)
            all_augmented_chunks.append(augmented_chunk)

    # Create and store embeddings in FAISS
    vector_store = FAISS.from_texts([chunk["chunk"] for chunk in all_augmented_chunks], embeddings)
    return vector_store, all_augmented_chunks

# Streamlit Interface
uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
url_input = st.text_area("Enter PDF/DOCX URLs (one per line)")

if "history" not in st.session_state:
    st.session_state.history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.augmented_chunks = []

if st.button("Process Documents"):
    urls = url_input.splitlines() if url_input else []
    documents = process_documents(uploaded_files, urls)

    if documents:
        st.session_state.vector_store, st.session_state.augmented_chunks = create_embeddings_and_store(documents)
        st.success("Documents processed and embeddings created.")
    else:
        st.warning("No documents to process.")

st.header("Chat with your Documents")
for entry in st.session_state.history:
    st.write(f"**User:** {entry['query']}")
    st.write(f"**Assistant:** {entry['response']}")

query = st.text_input("Please enter your query:", key="user_query")

async def get_relevant_chunks(sub_query, retriever, top_k=5):
    retrieved_docs = await retriever.ainvoke(sub_query)
    return [doc.page_content for doc in retrieved_docs[:top_k]]

if st.button("Delete Chat"):
    st.session_state.history = []

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
