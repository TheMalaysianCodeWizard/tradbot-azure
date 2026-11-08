"""Handlers for intents whose answer must be computed rather than canned."""

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Windows has no built-in time-zone database; the "tzdata" package supplies
# it. Without it, fall back to the computer's local time instead of crashing.
try:
    TIMEZONE = ZoneInfo("America/Chicago")
    TZ_LABEL = "US Central time"
except ZoneInfoNotFoundError:
    TIMEZONE = None
    TZ_LABEL = "this computer's local time"

CAPABILITIES = [
    ("Greetings and small talk", "hi, how are you, thanks, bye"),
    ("Tell you about myself", "who are you, how do you work"),
    ("Remember your name", "my name is Sam  /  what is my name"),
    ("Tell the time and date", "what time is it, what is the date"),
    ("Do arithmetic", "calculate (3 + 4) * 2, what is 2^10"),
    ("Tell a joke", "tell me a joke"),
    ("Echo or reverse text", "echo hello  /  reverse hello"),
    ("Explain chatbot concepts", "chatbot lifecycle, traditional vs LLM"),
    ("Analyze sentiment with Azure AI Language", "sentiment I really enjoyed this week"),
    ("Show this list", "help"),
]


def capabilities(_text: str, _session: dict) -> str:
    lines = ["Here's what I can do:"]
    for i, (skill, example) in enumerate(CAPABILITIES, 1):
        lines.append(f"{i}. {skill} - try: \"{example}\"")
    return "\n".join(lines)


def set_name(text: str, session: dict) -> str:
    m = re.search(r"(?:my name is|call me|i am called|i'm called)\s+([a-z][a-z' -]{0,30})", text, re.I)
    if not m:
        return "I didn't catch your name. Try: \"my name is Sam\"."
    name = m.group(1).strip().split()[0].capitalize()
    session["name"] = name
    return f"Nice to meet you, {name}! I'll remember that for this conversation."


def get_name(_text: str, session: dict) -> str:
    if session.get("name"):
        return f"Your name is {session['name']}."
    return "I don't know your name yet. Tell me with \"my name is ...\"."


def time(_text: str, _session: dict) -> str:
    return f"It's {datetime.now(TIMEZONE).strftime('%I:%M %p').lstrip('0')} ({TZ_LABEL})."


def date(_text: str, _session: dict) -> str:
    return f"Today is {datetime.now(TIMEZONE).strftime('%A, %B %d, %Y')}."


# --- safe arithmetic -------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_WORDS = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "x": "*",
          "divided by": "/", "over": "/", "mod": "%", "to the power of": "**"}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def math(text: str, _session: dict) -> str:
    expr = re.sub(r"^(calculate|compute|calc|math|what is|what's)\b\s*", "", text.strip(), flags=re.I)
    expr = expr.rstrip("?=. ").lower()
    for word, sym in sorted(_WORDS.items(), key=lambda kv: -len(kv[0])):
        expr = re.sub(rf"\b{word}\b", sym, expr)
    expr = expr.replace("^", "**")
    if not expr or not re.fullmatch(r"[\d\s.+\-*/()%]+", expr):
        return "I can only do arithmetic with numbers and + - * / % ^ ( ). Example: calculate (3 + 4) * 2"
    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return "Dividing by zero isn't defined, so I can't calculate that."
    except (SyntaxError, ValueError):
        return "That expression looks malformed. Check the brackets and operators, e.g. calculate (3 + 4) * 2"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 6)
    return f"{expr.replace('**', '^')} = {value}"


def _payload(text: str, command: str) -> str:
    return re.sub(rf"^{command}\s*:?\s*", "", text.strip(), flags=re.I)


def echo(text: str, _session: dict) -> str:
    payload = _payload(text, "echo")
    return f"You said: {payload}" if payload else "Give me something to echo, e.g. \"echo hello\"."


def reverse(text: str, _session: dict) -> str:
    payload = _payload(text, "reverse")
    return payload[::-1] if payload else "Give me something to reverse, e.g. \"reverse hello\"."


HANDLERS = {
    "capabilities": capabilities, "set_name": set_name, "get_name": get_name,
    "time": time, "date": date, "math": math, "echo": echo, "reverse": reverse,
}
