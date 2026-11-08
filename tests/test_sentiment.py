"""Tests for the Azure AI Language integration.

A fake client stands in for azure.ai.textanalytics.TextAnalyticsClient, so
these tests need no key, no network, and use none of the free-tier quota.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import ChatBot  # noqa: E402
from chatbot.sentiment import SentimentService  # noqa: E402


def doc(label, pos, neu, neg):
    """Shaped like the SDK's AnalyzeSentimentResult."""
    return SimpleNamespace(is_error=False, sentiment=label,
                           confidence_scores=SimpleNamespace(positive=pos, neutral=neu, negative=neg))


class FakeClient:
    """Answers by keyword, records every call, can be told to fail."""

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def analyze_sentiment(self, documents, language=None):
        self.calls.append(documents[0])
        if self.fail_with:
            raise self.fail_with
        text = documents[0].lower()
        if any(w in text for w in ("hate", "terrible", "awful", "frustrat")):
            return [doc("negative", 0.02, 0.05, 0.93)]
        if any(w in text for w in ("love", "great", "awesome", "enjoy")):
            return [doc("positive", 0.96, 0.03, 0.01)]
        if "but" in text:
            return [doc("mixed", 0.45, 0.10, 0.45)]
        return [doc("neutral", 0.05, 0.90, 0.05)]


class HttpError(Exception):
    """Shaped like azure.core.exceptions.HttpResponseError."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class ServiceRequestError(Exception):
    """Same class name as the SDK's network error."""


def bot_with(client):
    return ChatBot(seed=0, sentiment=SentimentService(endpoint="https://x", key="k", client=client))


class TestSentimentService(unittest.TestCase):
    def test_labels_and_scores(self):
        s = SentimentService(endpoint="https://x", key="k", client=FakeClient())
        r = s.analyze("I love this class")
        self.assertTrue(r.ok)
        self.assertEqual(r.label, "positive")
        self.assertAlmostEqual(r.scores["positive"], 0.96)
        self.assertEqual(s.analyze("the weather is cloudy").label, "neutral")
        self.assertEqual(s.analyze("good but slow").label, "mixed")

    def test_not_configured(self):
        s = SentimentService(endpoint="", key="")
        self.assertFalse(s.available)
        self.assertIn("not configured", s.analyze("hi").error)

    def test_errors_are_explained_not_raised(self):
        cases = {401: "key", 404: "endpoint", 429: "limit"}
        for status, word in cases.items():
            with self.subTest(status=status):
                r = SentimentService(endpoint="https://x", key="k", client=FakeClient(HttpError(status))).analyze("hi")
                self.assertFalse(r.ok)
                self.assertIn(word, r.error)
        r = SentimentService(endpoint="https://x", key="k", client=FakeClient(ServiceRequestError())).analyze("hi")
        self.assertIn("could not reach Azure", r.error)

    def test_key_never_appears_in_messages(self):
        s = SentimentService(endpoint="https://x", key="SECRET-KEY-123", client=FakeClient(HttpError(401)))
        self.assertNotIn("SECRET-KEY-123", s.analyze("hi").summary())


class TestBotWithSentiment(unittest.TestCase):
    def test_every_valid_message_gets_sentiment(self):
        bot = bot_with(FakeClient())
        r = bot.respond("tell me a joke", {})
        self.assertEqual(r.intent, "joke")
        self.assertEqual(r.sentiment.label, "neutral")

    def test_negative_user_gets_empathy(self):
        r = bot_with(FakeClient()).respond("I hate this, what time is it", {})
        self.assertEqual(r.intent, "time")
        self.assertTrue(r.text.startswith("I'm sorry you're feeling that way."))

    def test_fallback_uses_sentiment(self):
        r = bot_with(FakeClient()).respond("purple elephant banana awful", {})
        self.assertEqual(r.source, "fallback")
        self.assertIn("frustrating", r.text)

    def test_sentiment_command(self):
        client = FakeClient()
        r = bot_with(client).respond("sentiment I really enjoyed this week", {})
        self.assertEqual(r.intent, "analyze_sentiment")
        self.assertIn("POSITIVE", r.text)
        self.assertEqual(client.calls, ["I really enjoyed this week"])  # command word not sent

    def test_malformed_input_never_calls_azure(self):
        client = FakeClient()
        bot = bot_with(client)
        for bad in ["", "   ", "!!!???", "qwrtzp sdfghjk", "x" * 600]:
            bot.respond(bad, {})
        self.assertEqual(client.calls, [])

    def test_bot_keeps_working_when_azure_fails(self):
        bot = bot_with(FakeClient(HttpError(401)))
        r = bot.respond("what time is it", {})
        self.assertEqual(r.intent, "time")
        self.assertIn("It's", r.text)
        self.assertFalse(r.sentiment.ok)
        self.assertIn("couldn't analyze", bot.respond("sentiment I love it", {}).text)

    def test_capabilities_mention_azure(self):
        self.assertIn("Azure AI Language", ChatBot().respond("help", {}).text)


if __name__ == "__main__":
    unittest.main()
