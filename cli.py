"""Command-line client.

    python cli.py           # chat; each reply shows Azure's sentiment verdict
    python cli.py --debug   # also show matched intent, confidence and source

Set MicrosoftAIServiceEndpoint and MicrosoftAPIKey first (see README.md);
without them the bot still runs, just without sentiment analysis.
"""

import sys

from chatbot import ChatBot
from chatbot.config import DefaultConfig
from chatbot.sentiment import SentimentService


def main() -> None:
    debug = "--debug" in sys.argv
    service = SentimentService()
    bot, session = ChatBot(sentiment=service), {}
    print("TradBot (type 'help' for capabilities, 'quit' to exit)")
    print(DefaultConfig.describe())
    if not service.available:
        print(f"Sentiment analysis disabled: {service.disabled_reason}")
    while True:
        try:
            msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        reply = bot.respond(msg, session)
        print(f"Bot: {reply.text}")
        s = reply.sentiment
        # show the verdict, or a runtime error (e.g. wrong key); skip the
        # repeated "not configured" note already printed at startup
        if s is not None and (s.ok or s.error != service.disabled_reason):
            print(f"     [Azure sentiment: {s.summary()}]")
        if debug:
            print(f"     [intent={reply.intent} confidence={reply.confidence:.2f} source={reply.source}]")
        if reply.intent == "goodbye":
            break
    if service.calls:
        print(f"(Azure AI Language calls this session: {service.calls})")


if __name__ == "__main__":
    main()
