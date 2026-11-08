"""Unit tests: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import ChatBot  # noqa: E402
from chatbot.nlu import IntentResult, NLUProvider  # noqa: E402


class TestIntents(unittest.TestCase):
    def setUp(self):
        self.bot = ChatBot(seed=0)

    def intent(self, text, session=None):
        return self.bot.respond(text, session if session is not None else {}).intent

    def test_multiple_prompts(self):
        cases = {
            "hello": "greeting", "how are you": "how_are_you", "who are you": "bot_identity",
            "what time is it": "time", "what is the date": "date", "tell me a joke": "joke",
            "thanks": "thanks", "bye": "goodbye", "chatbot lifecycle": "chatbot_lifecycle",
            "explain your algorithm": "how_do_you_work",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.intent(text), expected)

    def test_capabilities_list(self):
        reply = self.bot.respond("help", {})
        self.assertEqual(reply.intent, "capabilities")
        self.assertIn("Here's what I can do", reply.text)
        self.assertGreaterEqual(reply.text.count("\n"), 5)

    def test_typo_tolerance(self):
        self.assertEqual(self.intent("tell me a jokke"), "joke")
        self.assertEqual(self.intent("heyyyyy"), "greeting")
        self.assertEqual(self.intent("can you tell the tme"), "time")

    def test_name_memory(self):
        session = {}
        self.bot.respond("my name is ming", session)
        self.assertEqual(self.bot.respond("what is my name", session).text, "Your name is Ming.")

    def test_math(self):
        self.assertTrue(self.bot.respond("calculate (3+4)*2", {}).text.endswith("= 14"))
        self.assertTrue(self.bot.respond("what is 2^10", {}).text.endswith("= 1024"))
        self.assertIn("zero", self.bot.respond("10/0", {}).text)
        self.assertIn("malformed", self.bot.respond("calc 2++", {}).text)
        # huge exponents are refused instead of freezing the server
        self.assertNotIn("=", self.bot.respond("calc 9^999999", {}).text)

    def test_echo_and_reverse(self):
        self.assertEqual(self.bot.respond("echo hi there", {}).text, "You said: hi there")
        self.assertEqual(self.bot.respond("reverse abc", {}).text, "cba")


class TestMalformedInput(unittest.TestCase):
    def setUp(self):
        self.bot = ChatBot(seed=0)

    def source(self, raw):
        return self.bot.respond(raw, {}).source

    def test_validator(self):
        self.assertEqual(self.source(""), "validator:empty")
        self.assertEqual(self.source("    \n\t"), "validator:empty")
        self.assertEqual(self.source(None), "validator:empty")
        self.assertEqual(self.source("?!?!..."), "validator:no_words")
        self.assertEqual(self.source("a" * 501 + " b"), "validator:too_long")
        self.assertEqual(self.source("qwrtzp sdfghjk"), "validator:gibberish")

    def test_markup_is_stripped(self):
        self.assertEqual(self.bot.respond("<script>hi</script>", {}).intent, "greeting")

    def test_unknown_then_capabilities(self):
        session = {}
        first = self.bot.respond("purple elephant banana", session)
        second = self.bot.respond("blue giraffe sandwich", session)
        self.assertEqual(first.source, "fallback")
        self.assertIn("Here's what I can do", second.text)


class TestClarification(unittest.TestCase):
    class FakeNLU(NLUProvider):
        def predict(self, text):
            return IntentResult("joke", 0.3, "fake")

    def test_yes_confirms_pending_intent(self):
        bot = ChatBot(nlu=self.FakeNLU(), seed=0)
        session = {}
        self.assertIn("Did you want to hear a joke", bot.respond("something vague", session).text)
        self.assertEqual(bot.respond("yes", session).intent, "joke")

    def test_no_rejects_pending_intent(self):
        bot = ChatBot(nlu=self.FakeNLU(), seed=0)
        session = {}
        bot.respond("something vague", session)
        self.assertIn("my mistake", bot.respond("no", session).text)


class TestExtensibility(unittest.TestCase):
    def test_custom_provider_plugs_in(self):
        """Any NLUProvider (e.g. a cloud AI service) can replace the local one."""

        class CloudStub(NLUProvider):
            name = "cloud-stub"

            def predict(self, text):
                return IntentResult("thanks", 0.92, self.name)

        reply = ChatBot(nlu=CloudStub()).respond("anything at all", {})
        self.assertEqual((reply.intent, reply.source), ("thanks", "cloud-stub"))


if __name__ == "__main__":
    unittest.main()
