from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
from typing import List, Literal
from src.state import ResumeState
import psycopg

DB_CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME = "cv_evaluation_corpus"

def get_vector_store(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> PGVector:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    """ Initializes and returns the PGVector storage wrapper using local Sentence Transformer embeddings. """
    embeddings_engine = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

    
    return PGVector(
        embeddings=embeddings_engine,
        collection_name=COLLECTION_NAME,
        connection=DB_CONNECTION,
        use_jsonb=True
    )

def save_chunks_to_db(state: ResumeState, vector_store: PGVector) -> None:
    """
    Parse the anonymized markdown text, add section headers, 
    and associate chunks with the dataframe source via cv_id.
    """
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = markdown_splitter.split_text(state["anonymized_resume"])
    
    langchain_documents = []
    
    for doc in docs:
        # Stitch headers together to add context to text snippets
        header_context = " -> ".join([val for key, val in doc.metadata.items() if key.startswith("Header")])
        enriched_payload = f"Section: {header_context}\nContent: {doc.page_content}"
        
        # Streamlined metadata: Just structural layout info and the Foreign Key
        metadata_payload = {
            "cv_id": state["cv_id"],
            "section_path": header_context,
            "header_1": doc.metadata.get("Header_1", "Root"),
            "header_2": doc.metadata.get("Header_2", "None"),
            "header_3": doc.metadata.get("Header_3", "None")
        }
        
        langchain_documents.append(
            Document(page_content=enriched_payload, metadata=metadata_payload)
        )
        
    if langchain_documents:
        vector_store.add_documents(langchain_documents)

def native_pg_hybrid_search(query: str, k: int = 3, rrf_constant: int = 60) -> List[Document]:
    """
    Executes a native PostgreSQL Hybrid Search combining pgvector distance 
    and full-text keyword matching mapped via Reciprocal Rank Fusion (RRF).
    """
    # Generate the dense vector for the semantic half of the search
    embeddings_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_vector = embeddings_engine.embed_query(query)
    
    # Simplified SQL: Directly targets `cmetadata` and removes unnecessary joins
    sql_query = """
        WITH vector_search AS (
            SELECT 
                cpe.document, 
                cpe.cmetadata as meta,
                ROW_NUMBER() OVER (ORDER BY cpe.embedding <=> %s::vector) as rank
            FROM langchain_pg_embedding cpe
            JOIN langchain_pg_collection cpc ON cpe.collection_id = cpc.uuid
            WHERE cpc.name = %s
            LIMIT (%s * 4)
        ),
        keyword_search AS (
            SELECT 
                cpe.document, 
                cpe.cmetadata as meta,
                ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', cpe.document), plainto_tsquery('english', %s)) DESC) as rank
            FROM langchain_pg_embedding cpe
            JOIN langchain_pg_collection cpc ON cpe.collection_id = cpc.uuid
            WHERE cpc.name = %s AND to_tsvector('english', cpe.document) @@ plainto_tsquery('english', %s)
            LIMIT (%s * 4)
        )
        SELECT 
            COALESCE(v.document, k.document) as page_content,
            COALESCE(v.meta, k.meta) as metadata,
            (1.0 / (%s + COALESCE(v.rank, 100000))) + (1.0 / (%s + COALESCE(k.rank, 100000))) as rrf_score
        FROM vector_search v
        FULL OUTER JOIN keyword_search k ON v.document = k.document
        ORDER BY rrf_score DESC
        LIMIT %s;
    """
    
    retrieved_docs = []
    
    # Execute query directly against your local Docker database
    with psycopg.connect("postgresql://langchain:langchain@localhost:6024/langchain") as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, (
                query_vector, COLLECTION_NAME, k, 
                query, COLLECTION_NAME, query, k,
                rrf_constant, rrf_constant, k
            ))
            rows = cur.fetchall()
            
            for row in rows:
                retrieved_docs.append(
                    Document(page_content=row[0], metadata=row[1])
                )
                
    return retrieved_docs

def retrieve_context(
    query: str, 
    vector_store: PGVector, 
    strategy: Literal["reranker", "hybrid"], 
    top_n: int = 3
) -> List[Document]:
    """
    Unified entrypoint allowing clean runtime strategy changes 
    inside interactive Jupyter notebooks.
    """
    if strategy == "hybrid":
        # Strategy A: Blend sparse and dense lists natively inside Postgres
        return native_pg_hybrid_search(query, k=top_n)
        
    elif strategy == "reranker":
        # Strategy B: Broad dense vector fetch followed by local Cross-Encoder filtering
        from src.reranker import rerank_retrieved_chunks
        raw_chunks = vector_store.similarity_search(query, k=15)
        return rerank_retrieved_chunks(query, raw_chunks, top_n=top_n)
        
    else:
        raise ValueError(f"Unknown retrieval strategy option: {strategy}")