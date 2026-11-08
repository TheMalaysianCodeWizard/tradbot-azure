"""Natural-language-understanding (NLU) layer.

The bot asks an ``NLUProvider`` to turn a cleaned message into an
``IntentResult``. Two local, traditional providers are included:

* ``RegexProvider``   - hand-written regular expressions (high precision)
* ``TfidfProvider``   - TF-IDF + cosine similarity against example phrases

``CascadeProvider`` chains them: rules first, statistics second.

The same interface is the extension point for cloud "AI as a service"
offerings. ``AzureCLUProvider`` shows where Azure AI Language (Conversational
Language Understanding) would plug in; it is only used when the matching
environment variables are set, so the bot runs fully offline by default.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class IntentResult:
    tag: str | None
    confidence: float
    source: str  # which provider produced the result
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class NLUProvider(ABC):
    """Interface every NLU back end must implement."""

    name = "base"

    @abstractmethod
    def predict(self, text: str) -> IntentResult:
        ...


# ---------------------------------------------------------------------------
# 1. Rule-based matching
# ---------------------------------------------------------------------------
class RegexProvider(NLUProvider):
    name = "regex"

    def __init__(self, intents: list[dict]):
        self.rules = []
        for intent in intents:
            for pattern in intent.get("patterns", []):
                self.rules.append((intent["tag"], re.compile(pattern, re.IGNORECASE)))

    def predict(self, text: str) -> IntentResult:
        for tag, rx in self.rules:
            if rx.search(text):
                return IntentResult(tag, 1.0, self.name)
        return IntentResult(None, 0.0, self.name)


# ---------------------------------------------------------------------------
# 2. Statistical matching (TF-IDF), with a pure-Python fallback
# ---------------------------------------------------------------------------
class TfidfProvider(NLUProvider):
    name = "tfidf"

    def __init__(self, intents: list[dict]):
        self.examples: list[str] = []
        self.labels: list[str] = []
        for intent in intents:
            for ex in intent.get("examples", []):
                self.examples.append(ex.lower())
                self.labels.append(intent["tag"])

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Word unigrams+bigrams capture phrasing; char n-grams make the
            # matcher tolerant of small typos ("helo", "jokke").
            self._word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            self._Xw = self._word.fit_transform(self.examples)
            self._Xc = self._char.fit_transform(self.examples)
            self.backend = "scikit-learn"
        except ImportError:  # keeps the bot usable if scikit-learn is missing
            self.backend = "difflib"

    def _scores(self, text: str) -> list[float]:
        if self.backend == "scikit-learn":
            from sklearn.metrics.pairwise import cosine_similarity

            sw = cosine_similarity(self._word.transform([text]), self._Xw)[0]
            sc = cosine_similarity(self._char.transform([text]), self._Xc)[0]
            return list(0.5 * sw + 0.5 * sc)
        return [SequenceMatcher(None, text, ex).ratio() for ex in self.examples]

    def predict(self, text: str) -> IntentResult:
        scores = self._scores(text.lower())
        best: dict[str, float] = {}
        for label, score in zip(self.labels, scores):
            best[label] = max(best.get(label, 0.0), float(score))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return IntentResult(None, 0.0, self.name)
        tag, conf = ranked[0]
        return IntentResult(tag, conf, self.name, ranked[1:3])


# ---------------------------------------------------------------------------
# 3. Cascade: rules first, then statistics
# ---------------------------------------------------------------------------
class CascadeProvider(NLUProvider):
    name = "cascade"

    def __init__(self, providers: list[NLUProvider]):
        self.providers = providers

    def predict(self, text: str) -> IntentResult:
        best = IntentResult(None, 0.0, self.name)
        for provider in self.providers:
            result = provider.predict(text)
            if result.tag and result.confidence >= 1.0:
                return result  # an exact rule hit wins immediately
            if result.confidence > best.confidence:
                best = result
        return best


# ---------------------------------------------------------------------------
# 4. Extension point: a cloud AI service (not active unless configured)
# ---------------------------------------------------------------------------
class AzureCLUProvider(NLUProvider):
    """Azure AI Language - Conversational Language Understanding.

    Set AZURE_LANGUAGE_ENDPOINT, AZURE_LANGUAGE_KEY, AZURE_CLU_PROJECT and
    AZURE_CLU_DEPLOYMENT, and install ``azure-ai-language-conversations``.
    The CLU project's intent names should match the tags in intents.json so
    the existing handlers and responses are reused unchanged.
    """

    name = "azure-clu"

    def __init__(self):
        from azure.ai.language.conversations import ConversationAnalysisClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore

        self.client = ConversationAnalysisClient(
            os.environ["AZURE_LANGUAGE_ENDPOINT"], AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"])
        )
        self.project = os.environ["AZURE_CLU_PROJECT"]
        self.deployment = os.environ["AZURE_CLU_DEPLOYMENT"]

    def predict(self, text: str) -> IntentResult:
        result = self.client.analyze_conversation(
            task={
                "kind": "Conversation",
                "analysisInput": {
                    "conversationItem": {"id": "1", "participantId": "user", "text": text}
                },
                "parameters": {"projectName": self.project, "deploymentName": self.deployment},
            }
        )
        prediction = result["result"]["prediction"]
        top = prediction["topIntent"]
        conf = next((i["confidenceScore"] for i in prediction["intents"] if i["category"] == top), 0.0)
        return IntentResult(top, conf, self.name)


def build_default_provider(intents: list[dict]) -> NLUProvider:
    """Local cascade by default; prepend Azure CLU when it is configured."""
    providers: list[NLUProvider] = [RegexProvider(intents), TfidfProvider(intents)]
    if os.environ.get("AZURE_LANGUAGE_ENDPOINT") and os.environ.get("AZURE_LANGUAGE_KEY"):
        try:
            providers.insert(0, AzureCLUProvider())
        except Exception as exc:  # fall back to local NLU rather than crash
            print(f"[nlu] Azure CLU unavailable, using local NLU only: {exc}")
    return CascadeProvider(providers)
