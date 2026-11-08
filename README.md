# TradBot + Azure AI Language: a chatbot connected to an AI service

A traditional, rule-based chatbot extended with **sentiment analysis from
Azure AI Language**, a cloud AI-as-a-service offering. The bot still answers from
its own rules and knowledge base. Azure tells it whether each message
sounds positive, neutral, negative or mixed, and the bot uses that to adjust
its tone and to answer a new `sentiment <text>` command.

## Where this code came from (and why it was copied)

This project is a **copy of TradBot**, the traditional chatbot I built for the
*Prototype Simple Traditional Chatbot* assignment
(https://github.com/TheMalaysianCodeWizard/traditional-chatbot). The
instructions suggest starting from the course EchoBot sample and allow a
different bot as the starting point. I copied TradBot because it already met the base
requirements (multiple prompts, a capability list, malformed-input handling),
so this project could focus on connecting it to an AI service. The original
repository stays unchanged as the record of the earlier assignment.

**Changed or added for this assignment:**

| File | Change |
|---|---|
| `chatbot/config.py` | **New.** `DefaultConfig` reads the Azure endpoint and key from environment variables |
| `chatbot/sentiment.py` | **New.** Wraps Azure's `TextAnalyticsClient`; turns every Azure failure into a safe message |
| `chatbot/engine.py` | Calls Azure for each valid message, adds empathy for negative messages, sentiment-aware fallbacks, `sentiment <text>` command |
| `chatbot/intents.json`, `chatbot/skills.py` | New `analyze_sentiment` intent; capability list mentions Azure |
| `cli.py`, `web_preview.py` | Show Azure's verdict under each reply and the connection status at startup |
| `tests/test_sentiment.py` | **New.** 11 tests using a fake Azure client (no key or network needed) |
| `requirements.txt` | Adds `azure-ai-textanalytics` |

Removed from the copy because they don't apply here: the Hugging Face static
version (a browser page must never hold a secret key), the Gradio app and the
parity tools.

## How a message is processed

```
user text
  -> validate & clean            -> malformed? explain (no Azure call, saves quota)
  -> regex rules / TF-IDF        -> pick an intent (or clarify / fall back)
  -> Azure AI Language           -> sentiment label + confidence scores
  -> reply from knowledge base   -> softened if the user sounds negative;
                                    fallbacks phrased by sentiment
```

If Azure is not configured, unreachable, rejects the key, or the free tier
(5,000 text records per 30 days) runs out, the bot says sentiment is
unavailable and **keeps working**.

## Setup (Windows PowerShell)

1. Create an Azure AI **Language** resource (Free F0 tier) and open
   **Keys and Endpoint**.
2. In the project folder:

```powershell
pip install -r requirements.txt
$env:MicrosoftAIServiceEndpoint = "https://<your-resource>.cognitiveservices.azure.com/"
$env:MicrosoftAPIKey = "<paste KEY 1 here>"
python cli.py               # terminal chat
python web_preview.py       # browser chat at http://127.0.0.1:8000
python -m unittest discover -s tests -v
```

In `cmd.exe` use `SET MicrosoftAPIKey=<key>` instead. The variables last
until the window is closed.

**Never commit the key.** It lives only in environment variables; it is not in
any file in this repository.

## Try these messages

| Message | What happens |
|---|---|
| `help` | Capability list (includes sentiment analysis) |
| `sentiment I really enjoyed this week` | Azure's label and confidence scores |
| `I hate waiting, what time is it` | Time, prefixed with an empathetic line (negative) |
| `purple elephant banana` | Fallback reply, worded by sentiment |
| empty message, `!!!???` | Malformed-input message; no Azure call |
