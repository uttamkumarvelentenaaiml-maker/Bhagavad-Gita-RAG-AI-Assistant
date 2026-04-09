from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

print("Loading RAG system...")

# Load data
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# 🔥 SPLIT INTO CHUNKS (IMPORTANT)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Vector DB
db = FAISS.from_documents(docs, embeddings)

# Model
generator = pipeline(
    "text-generation",
    model="google/flan-t5-large"
)

print("RAG system ready!")

# Retrieve relevant context
def retrieve(query):
    results = db.similarity_search(query, k=2)
    return " ".join([doc.page_content for doc in results])

# Final function
def ask(query):
    context = retrieve(query)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question: {query}

    Give a clear and short answer:
    """

    result = generator(prompt, max_length=200)

    return result[0]["generated_text"]