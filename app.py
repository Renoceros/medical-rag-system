"""
Multi-Agent Medical RAG System - Streamlit UI
Entry point for the application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.planner_agent import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.summarizer_agent import SummarizerAgent
from utils.vector_store import VectorStoreManager

# Page configuration
st.set_page_config(
    page_title="Medical Assistant - Multi-Agent RAG",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = None
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# Initialize agents (cached)
@st.cache_resource
def initialize_agents():
    """Initialize all agents and load vector store."""
    
    # Load or build vector store
    vector_store_path = Path("vector_db")
    if not vector_store_path.exists():
        st.warning("⚠️ Vector store not found. Please run setup to index documents first.")
        st.stop()
    
    vsm = VectorStoreManager()
    vsm.load(str(vector_store_path))
    
    # Initialize agents
    rag_agent = RAGAgent(vsm)
    evaluator_agent = EvaluatorAgent()
    summarizer_agent = SummarizerAgent()
    planner_agent = PlannerAgent(rag_agent, evaluator_agent, summarizer_agent)
    
    return planner_agent

planner = initialize_agents()

# UI Layout
st.title("🏥 Medical Information Assistant")
st.caption("Multi-Agent RAG System with Safety Evaluation")

# Sidebar - System Information
with st.sidebar:
    st.header("System Information")
    st.info("""
    **Active Agents:**
    - 🧭 Planner (Orchestrator)
    - 📚 RAG (Retrieval-Augmented Generation)
    - ✅ Evaluator (Quality Assessment)
    - 📝 Summarizer (Memory Management)
    """)
    
    st.divider()
    
    st.header("Conversation Stats")
    st.metric("Total Turns", st.session_state.turn_count)
    
    if st.session_state.conversation_summary:
        st.subheader("Current Summary")
        st.text_area(
            "Summary",
            st.session_state.conversation_summary,
            height=150,
            disabled=True
        )
    
    st.divider()
    
    # Settings
    st.header("Settings")
    show_sources = st.checkbox("Show Retrieved Sources", value=True)
    show_eval_details = st.checkbox("Show Evaluation Details", value=True)
    show_agent_reasoning = st.checkbox("Show Agent Reasoning", value=False)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_summary = None
        st.session_state.turn_count = 0
        st.rerun()

# Medical Disclaimer (always visible)
st.warning("""
⚠️ **MEDICAL DISCLAIMER**: This system provides general medical information from reference materials only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified 
healthcare provider for medical concerns.
""")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if "metadata" in message and message["role"] == "assistant":
            metadata = message["metadata"]
            
            # Confidence badge
            if "confidence" in metadata:
                confidence = metadata["confidence"]
                if confidence >= 0.7:
                    st.success(f"✅ Confidence: {confidence:.1%}")
                elif confidence >= 0.5:
                    st.warning(f"⚠️ Confidence: {confidence:.1%}")
                else:
                    st.error(f"❌ Low Confidence: {confidence:.1%}")
            
            # Show sources
            if show_sources and "sources" in metadata:
                with st.expander("📚 Retrieved Sources"):
                    for i, source in enumerate(metadata["sources"], 1):
                        st.markdown(f"**Source {i}** (Score: {source['score']:.3f})")
                        st.text(source["text"][:300] + "...")
                        st.divider()
            
            # Show evaluation
            if show_eval_details and "evaluation" in metadata:
                with st.expander("✅ Quality Evaluation"):
                    eval_data = metadata["evaluation"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Factuality", f"{eval_data.get('factuality_score', 0):.1%}")
                    with col2:
                        st.metric("Completeness", f"{eval_data.get('completeness_score', 0):.1%}")
                    with col3:
                        st.metric("Tone", f"{eval_data.get('tone_appropriateness', 0):.1%}")
                    
                    if eval_data.get("issues_found"):
                        st.warning("Issues: " + ", ".join(eval_data["issues_found"]))
            
            # Show agent reasoning
            if show_agent_reasoning and "reasoning" in metadata:
                with st.expander("🧭 Agent Decision Log"):
                    st.json(metadata["reasoning"])

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.turn_count += 1
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with planner
    with st.chat_message("assistant"):
        with st.spinner("🤔 Planning response..."):
            
            # Call planner to orchestrate agents
            result = planner.process_query(
                query=prompt,
                conversation_history=st.session_state.messages[:-1],  # Exclude current query
                conversation_summary=st.session_state.conversation_summary
            )
            
            # Handle different response types
            if result["status"] == "success":
                st.markdown(result["response"])
                
                # Update conversation summary if generated
                if "summary" in result:
                    st.session_state.conversation_summary = result["summary"]
                
                # Store assistant message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "metadata": {
                        "confidence": result.get("confidence", 0),
                        "sources": result.get("sources", []),
                        "evaluation": result.get("evaluation", {}),
                        "reasoning": result.get("reasoning", {})
                    }
                })
                
            elif result["status"] == "escalated":
                st.error("🚨 " + result["response"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })
                
            elif result["status"] == "refused":
                st.warning("⚠️ " + result["response"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })
                
            elif result["status"] == "clarification_needed":
                st.info("❓ " + result["response"])
                if "clarification_questions" in result:
                    st.markdown("**Please clarify:**")
                    for q in result["clarification_questions"]:
                        st.markdown(f"- {q}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"]
                })

# Footer
st.divider()
st.caption("Powered by Multi-Agent RAG Architecture | Built with Streamlit")