import json
import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import requests

# Set your GROQ API key directly
GROQ_API_KEY = "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"

# Page configuration
st.set_page_config(
    page_title="SHL Test Recommender",
    page_icon="🔍",
    layout="wide"
)

st.title("SHL Test Recommender")
st.write("Enter a job description or query to find the most relevant SHL assessments.")

# File uploader for the JSON data
uploaded_file = st.file_uploader("Upload your SHL data JSON file", type="json")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def load_models():
    """Load and cache the tokenizer and model"""
    try:
        device = torch.device("cpu")  # Forcing CPU for Streamlit Cloud compatibility
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def embed(texts, tokenizer, model, device):
    """Generate embeddings for input texts"""
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            model_output = model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def process_data(data):
    """Process the SHL data to extract corpus and metadata"""
    corpus = []
    metadata = []
    
    for entry in data:
        content_parts = [
            entry.get("title", ""),
            entry.get("content", ""),
            entry.get("pdf_content", "")
        ]
        combined = "\n".join(filter(None, content_parts))
        corpus.append(combined)
        metadata.append({
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "test_type": entry.get("test_type_full", ""),
            "adaptive_irt": entry.get("adaptive_irt", False),
            "remote_testing": entry.get("remote_testing", False)
        })
    
    return corpus, metadata

def build_index(corpus, tokenizer, model, device):
    """Build FAISS index and TF-IDF vectorizer"""
    try:
        # Embed corpus
        embeddings = embed(corpus, tokenizer, model, device)
        if embeddings is None:
            return None, None, None, None
            
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # BM25 for lexical similarity
        vectorizer = TfidfVectorizer().fit(corpus)
        corpus_tfidf = vectorizer.transform(corpus)
        
        return index, vectorizer, corpus_tfidf, embeddings
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return None, None, None, None

def hybrid_search(query, index, vectorizer, corpus_tfidf, embeddings, corpus, metadata, tokenizer, model, device, top_k=5, alpha=0.5):
    """Perform hybrid search combining semantic and lexical search"""
    try:
        # Semantic search
        query_embedding = embed([query], tokenizer, model, device)
        if query_embedding is None:
            return [], []
            
        _, faiss_ids = index.search(query_embedding, top_k)
        
        # Calculate semantic scores
        faiss_scores = [1 - np.linalg.norm(query_embedding - embeddings[i]) for i in faiss_ids[0]]

        # Lexical search
        query_tfidf = vectorizer.transform([query])
        bm25_scores = corpus_tfidf.dot(query_tfidf.T).toarray().flatten()
        bm25_top_ids = np.argsort(bm25_scores)[-top_k:][::-1]

        # Combine results
        all_ids = list(set(faiss_ids[0]) | set(bm25_top_ids))
        results = []
        for i in all_ids:
            bm25_score = bm25_scores[i]
            faiss_score = 1 - np.linalg.norm(query_embedding - embeddings[i])
            combined_score = alpha * faiss_score + (1 - alpha) * bm25_score
            results.append((combined_score, i))
        
        results.sort(reverse=True)
        return [metadata[i] for _, i in results[:top_k]], [corpus[i] for _, i in results[:top_k]]
    except Exception as e:
        st.error(f"Error in hybrid search: {str(e)}")
        return [], []

def call_llama(prompt):
    """Call the LLaMA model via Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an SHL test recommender. Suggest suitable assessments based on user input."},
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling LLaMA API: {str(e)}")
        return f"Error generating recommendation: {str(e)}"

# Main app logic
if uploaded_file is not None:
    try:
        # Load data from uploaded file
        shl_data = json.load(uploaded_file)
        
        # Process data
        corpus, metadata = process_data(shl_data)
        
        # Load models if not already loaded
        if not st.session_state.initialized:
            with st.spinner("Loading models..."):
                tokenizer, model, device = load_models()
                if None not in (tokenizer, model, device):
                    # Build index
                    with st.spinner("Building search index..."):
                        index, vectorizer, corpus_tfidf, embeddings = build_index(corpus, tokenizer, model, device)
                        
                        if None not in (index, vectorizer, corpus_tfidf, embeddings):
                            # Store in session state
                            st.session_state.tokenizer = tokenizer
                            st.session_state.model = model
                            st.session_state.device = device
                            st.session_state.corpus = corpus
                            st.session_state.metadata = metadata
                            st.session_state.index = index
                            st.session_state.vectorizer = vectorizer
                            st.session_state.corpus_tfidf = corpus_tfidf
                            st.session_state.embeddings = embeddings
                            st.session_state.initialized = True
                            
                            st.success("✅ Models and index built successfully!")
        
        # Query input
        query = st.text_area("Enter your query:", height=100, 
                            placeholder="Example: Looking for a test to assess civil engineering graduates with aptitude in transportation and water resources")
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        with col2:
            alpha = st.slider("Semantic vs. Lexical weight (higher = more semantic):", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        
        # Submit button
        if st.button("Search") and query and st.session_state.initialized:
            with st.spinner("Searching and generating recommendations..."):
                # Perform hybrid search
                top_meta, top_docs = hybrid_search(
                    query, 
                    st.session_state.index,
                    st.session_state.vectorizer,
                    st.session_state.corpus_tfidf,
                    st.session_state.embeddings,
                    st.session_state.corpus,
                    st.session_state.metadata,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    st.session_state.device,
                    top_k=top_k,
                    alpha=alpha
                )
                
                if top_meta and top_docs:
                    # Generate LLM recommendation
                    context = "\n\n".join(top_docs)
                    prompt = f"Here is the context of available SHL tests:\n\n{context}\n\nBased on this, suggest the most relevant assessments for the following job description or query:\n{query}"
                    llama_response = call_llama(prompt)
                    
                    # Display results
                    st.subheader("Top Matching Tests")
                    
                    # Create DataFrame from results
                    df = pd.DataFrame([{
                        "Title": r["title"],
                        "Remote Testing": "✅" if r["remote_testing"] else "❌",
                        "Adaptive/IRT": "✅" if r["adaptive_irt"] else "❌", 
                        "URL": r["url"]
                    } for r in top_meta])
                    
                    # Display as a table with clickable links
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn("URL")
                        },
                        use_container_width=True
                    )
                    
                    # Display LLM recommendation
                    st.subheader("AI Recommendation")
                    st.markdown(llama_response)
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload your SHL data JSON file to get started.")

# Add instructions and information at the bottom
with st.expander("ℹ️ About this app"):
    st.markdown("""
    **SHL Test Recommender** uses a hybrid search system combining:
    
    - **Semantic search** (FAISS with MPNet embeddings)
    - **Keyword search** (BM25/TF-IDF)
    - **LLaMA-powered recommendations**
    
    The system helps match job requirements with appropriate SHL assessments.
    """)