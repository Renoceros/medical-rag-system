"""
Summarizer Agent - Compresses conversation history for memory management.

This agent NEVER generates new medical facts, only compresses existing dialogue.
"""

from typing import Dict, List
import os
from anthropic import Anthropic

class SummarizerAgent:
    """
    Compresses conversation history to preserve context while reducing tokens.
    
    Responsibilities:
    - Extract key topics and medical terms from conversation
    - Preserve user constraints (conditions, allergies, medications)
    - Maintain conversation intent and context
    - Compress dialogue into concise summary
    """
    
    MAX_SUMMARY_LENGTH = 500  # Target summary length in words
    
    def __init__(self):
        """Initialize summarizer with LLM access."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
    
    def summarize(self, conversation_history: List[Dict]) -> Dict:
        """
        Compress conversation history into summary.
        
        Args:
            conversation_history: List of {"role": "user"|"assistant", "content": str}
            
        Returns:
            {
                "summary": str,
                "key_topics": [str],
                "user_constraints": [str],  # allergies, conditions mentioned
                "intent": str  # what user is trying to learn/achieve
            }
        """
        
        if not conversation_history:
            return {
                "summary": "",
                "key_topics": [],
                "user_constraints": [],
                "intent": ""
            }
        
        # Format conversation for summarization
        conversation_text = self._format_conversation(conversation_history)
        
        # Generate summary using LLM
        summary_result = self._generate_summary(conversation_text)
        
        return summary_result
    
    def _format_conversation(self, history: List[Dict]) -> str:
        """Format conversation history as text."""
        lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"]
            lines.append(f"{role}: {content}")
        
        return "\n\n".join(lines)
    
    def _generate_summary(self, conversation_text: str) -> Dict:
        """Use LLM to generate structured summary."""
        
        prompt = f"""Summarize this medical conversation concisely. Focus on:
1. Key medical topics discussed
2. User's health constraints mentioned (conditions, allergies, medications)
3. User's intent or what they're trying to learn

[CONVERSATION]
{conversation_text}

Provide a JSON response with:
{{
    "summary": "2-3 sentence concise summary of conversation",
    "key_topics": ["topic1", "topic2"],
    "user_constraints": ["constraint1", "constraint2"],
    "intent": "what user wants to know/achieve"
}}

Keep the summary under {self.MAX_SUMMARY_LENGTH} words. Return ONLY the JSON object."""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse JSON
            import json
            import re
            summary_text = response.content[0].text
            json_match = re.search(r'\{.*\}', summary_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._default_summary(conversation_text)
                
        except Exception as e:
            print(f"Summarization error: {e}")
            return self._default_summary(conversation_text)
    
    def _default_summary(self, conversation_text: str) -> Dict:
        """Fallback summary if LLM call fails."""
        # Simple extraction of first/last messages
        lines = conversation_text.split("\n\n")
        first_user_msg = next((line for line in lines if line.startswith("User:")), "")
        
        return {
            "summary": f"Conversation about medical topics. Started with: {first_user_msg[:100]}...",
            "key_topics": [],
            "user_constraints": [],
            "intent": "Seeking medical information"
        }