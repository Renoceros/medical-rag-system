"""
Vector Store Manager - Handles document indexing and retrieval.

Uses FAISS for efficient similarity search.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2

class VectorStoreManager:
    """
    Manages document indexing and vector similarity search.
    
    Features:
    - Document chunking with overlap
    - Embedding generation
    - FAISS indexing
    - Similarity search with thresholds
    """
    
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize vector store with embedding model."""
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def build_from_documents(self, doc_directory: str):
        """
        Build vector store from PDF documents in directory.
        
        Args:
            doc_directory: Path to directory containing PDF files
        """
        doc_path = Path(doc_directory)
        if not doc_path.exists():
            raise ValueError(f"Document directory not found: {doc_directory}")
        
        # Load and chunk documents
        all_chunks = []
        for pdf_file in doc_path.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            text = self._extract_pdf_text(pdf_file)
            chunks = self._chunk_text(text, source=pdf_file.name)
            all_chunks.extend(chunks)
        
        print(f"Total chunks created: {len(all_chunks)}")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
        self.index.add(embeddings.astype('float32'))
        self.chunks = all_chunks
        
        print(f"Vector store built successfully with {len(self.chunks)} chunks")
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )
        
        splits = splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(splits):
            if len(chunk_text.strip()) >= 50:  # Minimum chunk size
                chunks.append({
                    "id": f"{source}_chunk_{i}",
                    "text": chunk_text.strip(),
                    "source": source
                })
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Search for most similar chunks to query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            {
                "chunks": [{"chunk": dict, "score": float}],
                "top_score": float
            }
        """
        if self.index is None:
            raise ValueError("Vector store not initialized. Call build_from_documents() or load() first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(top_k, len(self.chunks))
        )
        
        # Convert distances to similarity scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                similarity = 1 / (1 + dist)  # Convert L2 distance to similarity
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(similarity)
                })
        
        top_score = results[0]["score"] if results else 0.0
        
        return {
            "chunks": results,
            "top_score": top_score
        }
    
    def save(self, save_path: str):
        """Save vector store to disk."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # Save chunks
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"Vector store saved to {save_path}")
    
    def load(self, load_path: str):
        """Load vector store from disk."""
        load_dir = Path(load_path)
        
        if not load_dir.exists():
            raise ValueError(f"Vector store path not found: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_dir / "faiss.index"))
        
        # Load chunks
        with open(load_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        print(f"Vector store loaded from {load_path} with {len(self.chunks)} chunks")


# Setup script to build vector store
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path_to_medical_documents>")
        sys.exit(1)
    
    doc_directory = sys.argv[1]
    
    print("Building vector store...")
    vsm = VectorStoreManager()
    vsm.build_from_documents(doc_directory)
    vsm.save("vector_db")
    print("Done!")