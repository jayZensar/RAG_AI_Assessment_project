"""Terminal chat loop for the YouTube RAG Chatbot.

Run with:
    python cli.py
Type 'exit' to quit.
"""
from src.rag_pipeline import YouTubeRagChatbot


def main() -> None:
    bot = YouTubeRagChatbot()
    bot.setup()
    bot.chat_loop()


if __name__ == "__main__":
    main()
