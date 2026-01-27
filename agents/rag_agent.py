"""
RAG Agent - Retrieves context from medical documents and generates grounded answers.

This is the ONLY agent allowed to generate medical content.
"""

from typing import Dict, List, Optional
import os
from anthropic import Anthropic

class RAGAgent:
    """
    Retrieval-Augmented Generation agent for medical queries.
    
    Responsibilities:
    - Retrieve relevant document chunks from vector store
    - Generate answers ONLY from retrieved context
    - Explicitly state when information is not available
    - Maintain source attribution
    """
    
    # Retrieval thresholds
    RELEVANCE_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    def __init__(self, vector_store_manager):
        """
        Initialize RAG agent.
        
        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store = vector_store_manager
        
        # Initialize LLM (Claude)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        
    def query(
        self,
        query: str,
        conversation_summary: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Retrieve relevant documents and generate grounded answer.
        
        Args:
            query: User's question
            conversation_summary: Previous conversation context
            top_k: Number of chunks to retrieve
            
        Returns:
            {
                "answer": str,
                "sources": [{"chunk_id": str, "text": str, "score": float}],
                "retrieval_confidence": float,
                "contains_answer": bool
            }
        """
        
        # STEP 1: Retrieve relevant chunks
        retrieval_result = self.vector_store.search(query, top_k=top_k)
        
        # STEP 2: Check if we have relevant documents
        if not retrieval_result["chunks"]:
            return {
                "answer": "I don't have information about this topic in the medical reference book.",
                "sources": [],
                "retrieval_confidence": 0.0,
                "contains_answer": False
            }
        
        # Filter by relevance threshold
        relevant_chunks = [
            chunk for chunk in retrieval_result["chunks"]
            if chunk["score"] > self.RELEVANCE_THRESHOLD
        ]
        
        if not relevant_chunks:
            return {
                "answer": "I don't have reliable information about this topic in the medical reference book.",
                "sources": [],
                "retrieval_confidence": retrieval_result["top_score"],
                "contains_answer": False
            }
        
        # STEP 3: Generate answer using retrieved context
        answer = self._generate_grounded_answer(
            query=query,
            chunks=relevant_chunks,
            conversation_summary=conversation_summary
        )
        
        # STEP 4: Format sources
        sources = [
            {
                "chunk_id": chunk["chunk"]["id"],
                "text": chunk["chunk"]["text"],
                "score": chunk["score"]
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_confidence": retrieval_result["top_score"],
            "contains_answer": True
        }
    
    def _generate_grounded_answer(
        self,
        query: str,
        chunks: List[Dict],
        conversation_summary: Optional[str]
    ) -> str:
        """
        Generate answer using LLM, strictly grounded in retrieved chunks.
        
        This uses Claude with a system prompt that enforces grounding.
        """
        
        # Build context from retrieved chunks
        context = self._format_context(chunks)
        
        # Build prompt
        system_prompt = self._get_grounding_system_prompt()
        user_prompt = self._build_user_prompt(query, context, conversation_summary)
        
        # Call Claude
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.3,  # Low temperature for factual responses
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.content[0].text
            return answer
            
        except Exception as e:
            return f"I encountered an error generating a response: {str(e)}"
    
    def _get_grounding_system_prompt(self) -> str:
        """System prompt that enforces strict grounding."""
        return """You are a medical information assistant. Your ONLY job is to answer questions using EXCLUSIVELY the provided medical document excerpts.

STRICT RULES - THESE ARE MANDATORY:
1. ONLY use information from the [RETRIEVED CONTEXT] provided below
2. If the context doesn't contain the answer, you MUST say: "The medical book doesn't provide information about this topic."
3. NEVER add information from your general knowledge or training
4. ALWAYS indicate which part of the context supports your answer
5. If the context is ambiguous or contradictory, acknowledge the uncertainty
6. Use cautious language: "According to the medical book..." or "The provided information indicates..."
7. If asked for medical advice, diagnosis, or treatment decisions, remind the user to consult a healthcare professional

Your goal is to provide accurate information from the reference material, not to be helpful by making things up."""
    
    def _build_user_prompt(
        self,
        query: str,
        context: str,
        conversation_summary: Optional[str]
    ) -> str:
        """Build user prompt with context and query."""
        
        prompt_parts = []
        
        # Add conversation summary if available
        if conversation_summary:
            prompt_parts.append(f"[CONVERSATION CONTEXT]\n{conversation_summary}\n")
        
        # Add retrieved context
        prompt_parts.append(f"[RETRIEVED CONTEXT FROM MEDICAL BOOK]\n{context}\n")
        
        # Add question
        prompt_parts.append(f"[QUESTION]\n{query}\n")
        
        # Add instruction
        prompt_parts.append("[YOUR ANSWER - GROUNDED ONLY IN ABOVE CONTEXT]\nAnswer the question using ONLY the information in the retrieved context above. If the context doesn't contain the answer, say so explicitly.")
        
        return "\n".join(prompt_parts)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for i, chunk_data in enumerate(chunks, 1):
            chunk = chunk_data["chunk"]
            score = chunk_data["score"]
            
            context_parts.append(
                f"--- Source {i} (Relevance: {score:.2f}) ---\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)