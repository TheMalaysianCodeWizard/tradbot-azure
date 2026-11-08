"""Sentiment analysis through Azure AI Language (an AI-as-a-service offering).

The bot sends each valid user message to Azure's sentiment endpoint and gets
back a label (positive, neutral, negative, mixed) with confidence scores.
Every failure - missing settings, missing library, wrong key, quota used up,
no network - is caught here and turned into a SentimentResult with an error
message, so the chatbot keeps working without sentiment instead of crashing.

Reference: Azure Text Analytics client library for Python
https://learn.microsoft.com/en-us/python/api/overview/azure/ai-textanalytics-readme
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DefaultConfig

LABELS = ("positive", "neutral", "negative", "mixed")


@dataclass
class SentimentResult:
    label: str | None = None                  # positive / neutral / negative / mixed
    scores: dict = field(default_factory=dict)  # {"positive": 0.97, "neutral": 0.02, "negative": 0.01}
    error: str | None = None                  # set when Azure could not be used

    @property
    def ok(self) -> bool:
        return self.label is not None and self.error is None

    def summary(self) -> str:
        if not self.ok:
            return f"sentiment unavailable ({self.error})"
        s = self.scores
        return (f"{self.label} (positive {s.get('positive', 0):.2f}, "
                f"neutral {s.get('neutral', 0):.2f}, negative {s.get('negative', 0):.2f})")


class SentimentService:
    """Thin wrapper around azure.ai.textanalytics.TextAnalyticsClient."""

    def __init__(self, endpoint: str | None = None, key: str | None = None, client=None,
                 timeout: float | None = None):
        self.endpoint = endpoint if endpoint is not None else DefaultConfig.ENDPOINT_URI
        self.key = key if key is not None else DefaultConfig.API_KEY
        self.timeout = timeout if timeout is not None else DefaultConfig.AZURE_TIMEOUT
        self.client = client
        self.disabled_reason: str | None = None
        self.calls = 0
        if self.client is None:
            self._connect()

    def _connect(self) -> None:
        if not (self.endpoint and self.key):
            self.disabled_reason = "Azure settings not configured"
            return
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            self.disabled_reason = "azure-ai-textanalytics is not installed (pip install azure-ai-textanalytics)"
            return
        try:
            self.client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
                retry_total=1,                    # fail fast instead of retrying for a minute
                connection_timeout=self.timeout,
                read_timeout=self.timeout,
            )
        except Exception as exc:  # e.g. malformed endpoint URL
            self.disabled_reason = f"could not create Azure client: {exc}"

    @property
    def available(self) -> bool:
        return self.client is not None and self.disabled_reason is None

    def analyze(self, text: str) -> SentimentResult:
        if not self.available:
            return SentimentResult(error=self.disabled_reason or "Azure client unavailable")
        try:
            self.calls += 1
            doc = self.client.analyze_sentiment([text], language="en")[0]
        except Exception as exc:
            return SentimentResult(error=_explain(exc))
        if getattr(doc, "is_error", False):
            return SentimentResult(error=f"Azure rejected the text: {getattr(doc.error, 'message', doc.error)}")
        cs = doc.confidence_scores
        scores = {"positive": float(cs.positive), "neutral": float(cs.neutral), "negative": float(cs.negative)}
        label = doc.sentiment if doc.sentiment in LABELS else max(scores, key=scores.get)
        return SentimentResult(label=label, scores=scores)


def _explain(exc: Exception) -> str:
    """Turn SDK exceptions into short, user-safe explanations (never the key)."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 401 or name == "ClientAuthenticationError":
        return "Azure rejected the key (401) - check MicrosoftAPIKey"
    if status == 403:
        return "access denied by Azure (403)"
    if status == 404:
        return "endpoint not found (404) - check MicrosoftAIServiceEndpoint"
    if status == 429:
        return "Azure free-tier limit reached (429) - try again later"
    if name in ("ServiceRequestError", "ServiceResponseError", "ConnectionError", "TimeoutError"):
        return "could not reach Azure (network or timeout)"
    return f"Azure error: {name}"
