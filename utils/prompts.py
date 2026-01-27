from typing import List
"""
Prompt templates and utilities.
"""

MEDICAL_DISCLAIMER = """
⚠️ **IMPORTANT**: This information is for educational purposes only and should not be used for self-diagnosis or treatment. Always consult a qualified healthcare provider for medical advice.
"""

EMERGENCY_RESPONSE = """🚨 **EMERGENCY DETECTED**

If you are experiencing a medical emergency, please:
- Call emergency services immediately (119 in Indonesia, 911 in US, 999 in UK, 112 in EU)
- Go to the nearest emergency room
- Contact a crisis helpline

**Crisis Resources:**
- National Suicide Prevention Lifeline (US): 988
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

I'm not able to provide emergency medical assistance. Please seek immediate professional help.
"""

SAFETY_REFUSAL = """⚠️ **SAFETY WARNING**

I cannot provide instructions for medical procedures that should only be performed by licensed healthcare professionals. Attempting to perform such procedures yourself could result in serious harm.

Please consult with a qualified healthcare provider who can:
- Properly assess your condition
- Perform procedures safely with proper equipment
- Provide appropriate follow-up care

If this is urgent, please visit an urgent care center or emergency room.
"""

def format_low_confidence_message(confidence: float, issues: List[str]) -> str:
    """Generate message for low-confidence responses."""
    return f"""I found some information but I'm not confident enough (confidence: {confidence:.1%}) to provide a reliable answer on this medical topic.

For your safety, please consult with a qualified healthcare provider who can:
- Review your specific situation
- Provide personalized medical advice
- Answer your questions with certainty

**Why I'm uncertain**: {', '.join(issues)}"""