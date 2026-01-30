"""
Planner Agent - Orchestrates all other agents and controls execution flow.

This is the ONLY agent that makes routing decisions. It never generates
medical content directly.
"""

from typing import Dict, List, Optional
from enum import Enum
import re

class PlannerDecision(Enum):
    """Possible actions the planner can take."""
    QUERY_RAG = "query_rag"
    RETRY_RAG = "retry_rag"
    SUMMARIZE = "summarize"
    ESCALATE = "escalate"
    REFUSE = "refuse"
    CLARIFY = "clarify"

class PlannerAgent:
    """
    Central orchestrator that controls all agent interactions.
    
    Responsibilities:
    - Safety checks (emergency detection, dangerous procedures)
    - Agent routing (which agents to call, in what order)
    - Retry logic (when to re-query RAG)
    - Escalation (when to refuse or show disclaimers)
    - Summarization triggers
    """
    
    # Safety keywords
    EMERGENCY_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die",
        "overdose", "self-harm", "cutting myself",
        "heart attack", "stroke", "can't breathe", "chest pain severe",
        "unconscious", "seizure", "bleeding heavily"
    ]
    
    DANGEROUS_PROCEDURES = [
        "perform surgery", "remove stitches myself", "self surgery",
        "extract tooth", "drain abscess myself", "remove appendix",
        "set broken bone", "perform biopsy"
    ]
    
    # Confidence thresholds
    CONFIDENCE_ACCEPT = 0.6
    CONFIDENCE_RETRY = 0.4
    
    # Summarization settings
    SUMMARIZE_EVERY_N_TURNS = 5
    
    def __init__(self, rag_agent, evaluator_agent, summarizer_agent):
        """Initialize planner with references to all agents."""
        self.rag_agent = rag_agent
        self.evaluator_agent = evaluator_agent
        self.summarizer_agent = summarizer_agent
        
    def process_query(
        self, 
        query: str, 
        conversation_history: List[Dict],
        conversation_summary: Optional[str] = None
    ) -> Dict:
        """
        Main entry point - orchestrate agents to respond to user query.
        
        Args:
            query: User's question
            conversation_history: Previous messages
            conversation_summary: Compressed conversation context
            
        Returns:
            {
                "status": "success" | "escalated" | "refused" | "clarification_needed",
                "response": str,
                "confidence": float,
                "sources": list,
                "evaluation": dict,
                "reasoning": dict,
                "summary": str (optional)
            }
        """
        
        # STEP 1: Safety checks (highest priority)
        safety_result = self._check_safety(query)
        if safety_result:
            return safety_result
        
        # STEP 2: Check if we need to summarize first
        turn_count = len(conversation_history) + 1
        if self._should_summarize(turn_count):
            summary = self.summarizer_agent.summarize(conversation_history)
            conversation_summary = summary["summary"]
        
        # STEP 3: Query RAG agent
        rag_result = self.rag_agent.query(
            query=query,
            conversation_summary=conversation_summary
        )
        
        # STEP 4: Evaluate RAG response
        evaluation = self.evaluator_agent.evaluate(
            query=query,
            answer=rag_result["answer"],
            sources=rag_result["sources"],
            retrieval_confidence=rag_result["retrieval_confidence"]
        )
        
        # STEP 5: Decide on action based on evaluation
        decision = self._make_decision(query, rag_result, evaluation, attempt=1)
        
        # STEP 6: Handle retry if needed
        if decision["action"] == PlannerDecision.RETRY_RAG:
            # Refine query and try again
            refined_query = self._refine_query(query, evaluation)
            rag_result = self.rag_agent.query(
                query=refined_query,
                conversation_summary=conversation_summary,
                top_k=8  # Retrieve more chunks on retry
            )
            
            # Re-evaluate
            evaluation = self.evaluator_agent.evaluate(
                query=query,
                answer=rag_result["answer"],
                sources=rag_result["sources"],
                retrieval_confidence=rag_result["retrieval_confidence"]
            )
            evaluation["retry_attempted"] = True
            
            # Make final decision
            decision = self._make_decision(query, rag_result, evaluation, attempt=2)
        
        # STEP 7: Format response based on decision
        return self._format_response(
            decision=decision,
            query=query,
            rag_result=rag_result,
            evaluation=evaluation,
            summary=conversation_summary
        )
    
    def _check_safety(self, query: str) -> Optional[Dict]:
        """
        Check for emergency or dangerous queries.
        
        Returns None if safe, otherwise returns escalation response.
        """
        query_lower = query.lower()
        
        # Check for emergency keywords
        if any(kw in query_lower for kw in self.EMERGENCY_KEYWORDS):
            return {
                "status": "escalated",
                "response": """🚨 **EMERGENCY DETECTED**

                If you are experiencing a medical emergency, please:
                - Call emergency services immediately (911 in US, 999 in UK, 112 in EU)
                - Go to the nearest emergency room
                - Contact a crisis helpline

                **Crisis Resources:**
                - National Suicide Prevention Lifeline (US): 988
                - Crisis Text Line: Text HOME to 741741
                - International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

                I'm not able to provide emergency medical assistance. Please seek immediate professional help.""",
                "confidence": 0.0,
                "reasoning": {"action": "emergency_escalation", "trigger": "emergency_keywords"}
            }
        
        # Check for dangerous self-procedures
        if any(proc in query_lower for proc in self.DANGEROUS_PROCEDURES):
            return {
                "status": "refused",
                "response": """⚠️ **SAFETY WARNING**

                I cannot provide instructions for medical procedures that should only be performed by licensed healthcare professionals. Attempting to perform such procedures yourself could result in serious harm.

                Please consult with a qualified healthcare provider who can:
                - Properly assess your condition
                - Perform procedures safely with proper equipment
                - Provide appropriate follow-up care

                If this is urgent, please visit an urgent care center or emergency room.""",
                "confidence": 0.0,
                "reasoning": {"action": "safety_refusal", "trigger": "dangerous_procedure"}
            }
        
        return None
    
    def _should_summarize(self, turn_count: int) -> bool:
        """Check if conversation should be summarized."""
        return turn_count > 0 and turn_count % self.SUMMARIZE_EVERY_N_TURNS == 0
    
    def _make_decision(
        self,
        query: str,
        rag_result: Dict,
        evaluation: Dict,
        attempt: int
    ) -> Dict:
        """
        Decide what action to take based on RAG and evaluation results.
        
        Returns:
            {
                "action": PlannerDecision,
                "reasoning": str,
                "needs_disclaimer": bool
            }
        """
        confidence = evaluation["overall_confidence"]
        hallucination_risk = evaluation["hallucination_risk"]
        
        # If no relevant documents found
        if not rag_result["contains_answer"]:
            return {
                "action": PlannerDecision.ESCALATE,
                "reasoning": "No relevant documents found in medical book",
                "needs_disclaimer": True
            }
        
        # If high hallucination risk, never accept
        if hallucination_risk == "high":
            if attempt == 1:
                return {
                    "action": PlannerDecision.RETRY_RAG,
                    "reasoning": "High hallucination risk detected, attempting retry"
                }
            else:
                return {
                    "action": PlannerDecision.ESCALATE,
                    "reasoning": "High hallucination risk persists after retry",
                    "needs_disclaimer": True
                }
        
        # Low confidence handling
        if confidence < self.CONFIDENCE_RETRY:
            if attempt == 1:
                return {
                    "action": PlannerDecision.RETRY_RAG,
                    "reasoning": f"Confidence {confidence:.2f} below retry threshold {self.CONFIDENCE_RETRY}"
                }
            else:
                return {
                    "action": PlannerDecision.ESCALATE,
                    "reasoning": f"Confidence {confidence:.2f} still low after retry",
                    "needs_disclaimer": True
                }
        
        # Medium confidence - accept with disclaimer
        if confidence < self.CONFIDENCE_ACCEPT:
            return {
                "action": PlannerDecision.QUERY_RAG,
                "reasoning": f"Medium confidence {confidence:.2f}, accepting with disclaimer",
                "needs_disclaimer": True
            }
        
        # High confidence - accept
        needs_disclaimer = self._requires_disclaimer(query)
        return {
            "action": PlannerDecision.QUERY_RAG,
            "reasoning": f"High confidence {confidence:.2f}",
            "needs_disclaimer": needs_disclaimer
        }
    
    def _requires_disclaimer(self, query: str) -> bool:
        """Check if query requires medical disclaimer even with good answer."""
        query_lower = query.lower()
        
        # Diagnostic queries
        diagnostic_patterns = [
            r"do i have", r"am i", r"is this", r"could this be",
            r"symptoms of", r"signs of", r"diagnosis"
        ]
        if any(re.search(pattern, query_lower) for pattern in diagnostic_patterns):
            return True
        
        # Treatment queries
        treatment_patterns = [
            r"should i take", r"how much", r"dosage", r"treatment for",
            r"cure for", r"medication for"
        ]
        if any(re.search(pattern, query_lower) for pattern in treatment_patterns):
            return True
        
        return False
    
    def _refine_query(self, original_query: str, evaluation: Dict) -> str:
        """Refine query based on evaluation feedback for retry."""
        issues = evaluation.get("issues_found", [])
        
        # Add context based on missing information
        expansions = []
        if "incomplete" in str(issues).lower():
            expansions.append("Include detailed explanation")
        if "missing" in str(issues).lower():
            expansions.append("with causes and symptoms")
        
        if expansions:
            return f"{original_query}. {' and '.join(expansions)}."
        
        return original_query
    
    def _format_response(
        self,
        decision: Dict,
        query: str,
        rag_result: Dict,
        evaluation: Dict,
        summary: Optional[str]
    ) -> Dict:
        """Format final response based on decision."""
        
        # Handle escalation
        if decision["action"] == PlannerDecision.ESCALATE:
            return {
                "status": "escalated",
                "response": self._generate_escalation_message(rag_result, evaluation),
                "confidence": evaluation["overall_confidence"],
                "reasoning": decision
            }
        
        # Handle normal response
        response = rag_result["answer"]
        
        # Add disclaimer if needed
        if decision.get("needs_disclaimer", False):
            disclaimer = """\n\n⚠️ **Important**: This information is for educational purposes only. Always consult a healthcare professional for medical advice, diagnosis, or treatment."""
            response = response + disclaimer
        
        return {
            "status": "success",
            "response": response,
            "confidence": evaluation["overall_confidence"],
            "sources": rag_result["sources"],
            "evaluation": evaluation,
            "reasoning": decision,
            "summary": summary
        }
    
    def _generate_escalation_message(self, rag_result: Dict, evaluation: Dict) -> str:
        """Generate appropriate escalation message."""
        if not rag_result["contains_answer"]:
            return """I don't have reliable information about this topic in the medical reference book. 

For accurate information, please:
- Consult with a healthcare professional
- Visit reputable medical websites (Mayo Clinic, NIH, CDC)
- Contact your doctor's office for specific medical advice"""
        
        return f"""I found some information but I'm not confident enough (confidence: {evaluation['overall_confidence']:.1%}) to provide a reliable answer on this medical topic.

For your safety, please consult with a qualified healthcare provider who can:
- Review your specific situation
- Provide personalized medical advice
- Answer your questions with certainty

**Why I'm uncertain**: {', '.join(evaluation.get('issues_found', ['Insufficient information']))}"""