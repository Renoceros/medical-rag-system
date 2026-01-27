"""
Evaluator Agent - Assesses quality and safety of RAG-generated answers.

This agent NEVER modifies or generates new medical content.
"""

from typing import Dict, List
import re
import os
from anthropic import Anthropic

class EvaluatorAgent:
    """
    Evaluates RAG outputs for factuality, completeness, and safety.
    
    Responsibilities:
    - Check if answer is grounded in sources
    - Detect hallucinations or unsupported claims
    - Assess completeness and tone
    - Produce structured evaluation (not new content)
    """
    
    def __init__(self):
        """Initialize evaluator with LLM access for assessment."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
    
    def evaluate(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        retrieval_confidence: float
    ) -> Dict:
        """
        Evaluate answer quality and safety.
        
        Args:
            query: Original user question
            answer: RAG-generated answer
            sources: Retrieved document chunks
            retrieval_confidence: Score from vector search
            
        Returns:
            {
                "overall_confidence": float (0-1),
                "factuality_score": float (0-1),
                "completeness_score": float (0-1),
                "tone_appropriateness": float (0-1),
                "hallucination_risk": "low" | "medium" | "high",
                "issues_found": [str],
                "recommendation": "accept" | "retry" | "escalate"
            }
        """
        
        # STEP 1: Check for obvious issues
        quick_checks = self._quick_safety_checks(answer, sources)
        if quick_checks["critical_issue"]:
            return {
                "overall_confidence": 0.0,
                "factuality_score": 0.0,
                "completeness_score": 0.0,
                "tone_appropriateness": 0.0,
                "hallucination_risk": "high",
                "issues_found": quick_checks["issues"],
                "recommendation": "escalate"
            }
        
        # STEP 2: Deep evaluation using LLM
        eval_result = self._llm_evaluation(query, answer, sources)
        
        # STEP 3: Compute overall confidence
        overall_confidence = self._compute_overall_confidence(
            eval_result,
            retrieval_confidence
        )
        
        # STEP 4: Determine recommendation
        recommendation = self._make_recommendation(overall_confidence, eval_result)
        
        return {
            "overall_confidence": overall_confidence,
            "factuality_score": eval_result["factuality_score"],
            "completeness_score": eval_result["completeness_score"],
            "tone_appropriateness": eval_result["tone_score"],
            "hallucination_risk": eval_result["hallucination_risk"],
            "issues_found": eval_result["issues"],
            "recommendation": recommendation
        }
    
    def _quick_safety_checks(self, answer: str, sources: List[Dict]) -> Dict:
        """Fast heuristic checks for obvious issues."""
        issues = []
        critical = False
        
        # Check 1: Is answer too short?
        if len(answer) < 20:
            issues.append("Answer is too brief")
        
        # Check 2: Does answer claim no information when sources exist?
        if sources and "don't have information" in answer.lower():
            issues.append("Claims no information despite relevant sources")
        
        # Check 3: Does answer contain dangerous advice patterns?
        dangerous_patterns = [
            r"definitely have",
            r"you should immediately",
            r"diagnose you with",
            r"guaranteed to cure"
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, answer.lower()):
                issues.append(f"Contains unsafe definitive language: {pattern}")
                critical = True
        
        return {
            "critical_issue": critical,
            "issues": issues
        }
    
    def _llm_evaluation(self, query: str, answer: str, sources: List[Dict]) -> Dict:
        """Use LLM to deeply evaluate answer quality."""
        
        # Format sources for evaluation
        sources_text = "\n\n".join([
            f"Source {i+1}: {s['text']}"
            for i, s in enumerate(sources)
        ])
        
        eval_prompt = f"""You are an expert evaluator for medical information systems. Evaluate the quality and safety of this answer.

[ORIGINAL QUESTION]
{query}

[RETRIEVED SOURCES FROM MEDICAL BOOK]
{sources_text}

[GENERATED ANSWER]
{answer}

Evaluate the answer on these criteria and return a JSON response:

1. **Factuality** (0.0-1.0): Are all statements in the answer grounded in the provided sources? Check each claim.

2. **Completeness** (0.0-1.0): Does the answer address all parts of the question adequately?

3. **Tone** (0.0-1.0): Is the tone appropriately cautious for medical information? Does it avoid overconfidence?

4. **Hallucination Risk** (low/medium/high): 
   - low: All statements traceable to sources
   - medium: Some inferential leaps or weak support
   - high: Clear statements not in sources

5. **Issues Found**: List any specific problems (e.g., "claims symptom X not mentioned in sources")

Return ONLY a JSON object with this structure:
{{
    "factuality_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "tone_score": 0.0-1.0,
    "hallucination_risk": "low"|"medium"|"high",
    "issues": ["issue1", "issue2"],
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.0,  # Deterministic evaluation
                messages=[
                    {"role": "user", "content": eval_prompt}
                ]
            )
            
            # Parse JSON response
            import json
            eval_text = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                return eval_data
            else:
                # Fallback if parsing fails
                return self._default_evaluation()
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._default_evaluation()
    
    def _default_evaluation(self) -> Dict:
        """Fallback evaluation if LLM call fails."""
        return {
            "factuality_score": 0.5,
            "completeness_score": 0.5,
            "tone_score": 0.5,
            "hallucination_risk": "medium",
            "issues": ["Could not complete automatic evaluation"],
            "reasoning": "Evaluation service unavailable"
        }
    
    def _compute_overall_confidence(
        self,
        eval_result: Dict,
        retrieval_confidence: float
    ) -> float:
        """
        Compute weighted overall confidence score.
        
        Formula:
        confidence = 0.4 * factuality + 0.2 * completeness + 0.1 * tone + 0.3 * retrieval
        
        Rationale: Factuality is most important, followed by retrieval quality.
        """
        weights = {
            "factuality": 0.4,
            "completeness": 0.2,
            "tone": 0.1,
            "retrieval": 0.3
        }
        
        # Penalize high hallucination risk
        hallucination_penalty = {
            "low": 0.0,
            "medium": 0.15,
            "high": 0.5
        }
        penalty = hallucination_penalty.get(eval_result["hallucination_risk"], 0.3)
        
        confidence = (
            weights["factuality"] * eval_result["factuality_score"] +
            weights["completeness"] * eval_result["completeness_score"] +
            weights["tone"] * eval_result["tone_score"] +
            weights["retrieval"] * retrieval_confidence
        ) * (1 - penalty)
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _make_recommendation(self, confidence: float, eval_result: Dict) -> str:
        """Decide whether to accept, retry, or escalate."""
        
        # Always escalate high hallucination risk
        if eval_result["hallucination_risk"] == "high":
            return "escalate"
        
        # Confidence-based thresholds
        if confidence >= 0.6:
            return "accept"
        elif confidence >= 0.4:
            return "retry"
        else:
            return "escalate"