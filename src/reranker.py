from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

def rerank_retrieved_chunks(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """
    Uses a Sentence-Transformers Cross-Encoder to re-rank the CV chunks 
    retrieved from vector db for a query.
    """
    if not documents:
        return []
        
    # Initialize the local Cross-Encoder model
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    scoring_pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict relevance scores for all pairs simultaneously
    scores = model.predict(scoring_pairs)
    
    # Zip the documents with their new scores and sort them descending
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the sorted Document objects up to top_n
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_n]]
        
    return reranked_docs