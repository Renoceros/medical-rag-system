
# Medical Information Assistant - Multi-Agent RAG System

A production-grade medical information system using multi-agent orchestration, retrieval-augmented generation (RAG), and safety evaluation.

## 🏥 Overview

This Streamlit application provides evidence-based medical information by combining:
- **Planner Agent**: Orchestrates workflow and makes safety decisions
- **RAG Agent**: Retrieves relevant information from medical documents
- **Evaluator Agent**: Assesses answer quality and detects hallucinations
- **Summarizer Agent**: Compresses conversation history for context efficiency

## ⚠️ Medical Disclaimer

This system provides **general medical information only** and is **NOT a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## 📋 Features

- **Multi-Agent Architecture**: Specialized agents for retrieval, evaluation, and orchestration
- **Safety Guardrails**: Emergency detection, dangerous procedure prevention, escalation logic
- **Quality Metrics**: Confidence scoring, hallucination detection, factuality assessment
- **Source Attribution**: Retrieved documents with relevance scores
- **Conversation Memory**: Automatic summarization every 5 turns
- **Expandable UI**: Toggle sources, evaluation details, and agent reasoning

## 🛠️ Project Structure

```
medical-rag-system/
├── agents/                      # Agent implementations
│   ├── planner_agent.py        # Orchestrator & routing logic
│   ├── rag_agent.py            # Retrieval-augmented generation
│   ├── evaluator_agent.py      # Quality & safety assessment
│   ├── summarizer_agent.py     # Conversation compression
│   └── __init__.py
├── data/
│   ├── medical_book/           # Source documents
│   └── vector_db/              # Embedded knowledge base
├── app.py                       # Streamlit UI entry point
└── README.md
```

## 🚀 Quick Start

```bash
# Make a Virtual Enviorment
python -m venv medaivenv | source medaivenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run application
streamlit run app.py
```

## 🔒 Safety Features

- **Emergency Detection**: Identifies crises and routes to emergency services
- **Procedure Prevention**: Blocks instructions for self-performed medical procedures
- **Hallucination Detection**: Evaluates answer grounding in sources
- **Confidence Thresholds**: Accepts/retries/escalates based on quality metrics
- **Automatic Retry**: Refines queries for low-confidence responses

## 📊 Agent Workflows

1. **Query Processing**: Safety checks → RAG retrieval → Evaluation
2. **Quality Control**: Factuality + completeness + tone assessment
3. **Retry Logic**: Automatically improves low-confidence responses
4. **Memory Management**: Summarizes conversations to maintain context efficiency

## 🎯 Confidence Scoring

- **≥0.6**: Accept answer
- **0.4–0.6**: Accept with disclaimer
- **<0.4**: Escalate or retry

## 📝 License

Medical content from professional reference materials only. System designed for educational and informational purposes.
