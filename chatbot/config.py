"""Runtime configuration.

ADDED for the AI-service assignment: the Azure AI Language endpoint and key
are read from environment variables so the secret key never appears in the
source code or in the GitHub repository. Variable names follow the course
instructions.

    PowerShell:  $env:MicrosoftAIServiceEndpoint = "https://<name>.cognitiveservices.azure.com/"
                 $env:MicrosoftAPIKey = "<key from Keys and Endpoint>"
    cmd.exe:     SET MicrosoftAIServiceEndpoint=https://<name>.cognitiveservices.azure.com/
                 SET MicrosoftAPIKey=<key>
"""

import os


class DefaultConfig:
    """Settings the bot needs at startup."""

    # Azure AI Language (sentiment analysis)
    ENDPOINT_URI = os.environ.get("MicrosoftAIServiceEndpoint", "").strip()
    API_KEY = os.environ.get("MicrosoftAPIKey", "").strip()

    # Seconds to wait for Azure before giving up and answering without sentiment.
    AZURE_TIMEOUT = float(os.environ.get("TradBotAzureTimeout", "8"))

    @classmethod
    def azure_configured(cls) -> bool:
        return bool(cls.ENDPOINT_URI and cls.API_KEY)

    @classmethod
    def describe(cls) -> str:
        """Safe-to-print summary: shows the endpoint but never the key."""
        if not cls.azure_configured():
            missing = [n for n, v in (("MicrosoftAIServiceEndpoint", cls.ENDPOINT_URI),
                                      ("MicrosoftAPIKey", cls.API_KEY)) if not v]
            return "Azure AI Language: not configured (missing " + ", ".join(missing) + ")"
        return f"Azure AI Language: {cls.ENDPOINT_URI} (key set, hidden)"
