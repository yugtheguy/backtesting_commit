import google.generativeai as genai
import os
import logging

logger = logging.getLogger(__name__)

# System prompt to define the chatbot's persona and domain knowledge
SYSTEM_PROMPT = """
You are a helpful AI assistant for a financial backtesting platform. Your name is 'TradeBuddy'.
Your purpose is to assist users with questions related to financial trading, backtesting, and the platform's features.
Be concise and direct. Do not engage in topics outside of finance and trading.
DO NOT provide financial advice. Always state that your responses are for informational and educational purposes only.
"""

def get_gemini_response(message: str, chat_history: list) -> str:
    """
    Interacts with the Gemini API to get a conversational response.
    
    Args:
        message: The user's new message.
        chat_history: A list of previous messages in the conversation.
        
    Returns:
        The text response from the chatbot.
    """
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
        
        # Format history for the Gemini API
        formatted_history = []
        for msg in chat_history:
            formatted_history.append({'role': msg.sender, 'parts': [{'text': msg.message_text}]})
        
        # Start a new chat session with the formatted history
        chat = model.start_chat(history=formatted_history)
        
        # Send the user's new message
        response = chat.send_message(message)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "I'm sorry, I'm having trouble connecting right now. Please try again later."
