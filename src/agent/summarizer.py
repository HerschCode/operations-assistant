"""
Summarize older conversation turns before they overflow the retention window.

Called from routes.py when a conversation crosses SUMMARIZE_THRESHOLD stored turns.
Uses a fast, single Anthropic call (claude-haiku, no tools, no history) to distill the
oldest SUMMARIZE_BATCH_SIZE turns into 2-4 sentences. The summary is stored in the
SQLite summaries table and prepended as synthetic context by get_history() so future
turns have it without re-replaying every past message.

Design choice: we use the cheapest capable Haiku model here, not the main agent model,
because compilation accuracy matters less than cost for a background summary; a rough
précis of what was discussed is enough to orient the model on follow-up turns.
"""
import os

_SUMMARIZE_PROMPT = """\
The following are older turns from an ongoing conversation with an operations assistant. \
Summarize them in 2-4 sentences so a reader has the essential context (what was asked, what \
was found, any key numbers or policy references) without needing to re-read every turn. \
Be specific — "cycle time for electronics was 14.2 days" is more useful than "cycle time was discussed."

Conversation turns:
{turns_text}

Summary (2-4 sentences):"""


def summarize_turns(
    turns: list[dict],
    client=None,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Summarize a list of {"role": ..., "content": ...} turns into 2-4 sentences.

    Never raises: returns a fallback string if the LLM call fails so callers can
    treat summarization failure as a soft, non-fatal degradation.
    """
    if not turns:
        return ""

    if client is None:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    turns_text = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in turns
    )
    prompt = _SUMMARIZE_PROMPT.format(turns_text=turns_text)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as exc:
        return f"[summary unavailable: {exc}]"
