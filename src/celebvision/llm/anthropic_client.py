import json
from celebvision.interfaces import MentionExtraction
from celebvision.models import SpokenMention, KeywordHit

_PROMPT = """You are extracting celebrity mentions and keyword hits from a transcript.

Only report a celebrity if it matches one of the watchlist identities (by name or alias).
Return STRICT JSON, no prose, matching this schema:
{{"mentions": [{{"name": str, "canonical_id": str, "confidence": float 0..1,
  "evidence_span": str}}],
 "keyword_hits": [{{"keyword": str, "count": int, "spans": [str]}}]}}

Watchlist (canonical_id | name | aliases):
{watchlist}

Keywords to count: {keywords}

Transcript:
{text}
"""


class AnthropicLLMClient:
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client = None

    def _load(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key) if self._api_key \
                else Anthropic()
        return self._client

    def _parse_response(self, payload: dict, keywords: list[str]) -> MentionExtraction:
        mentions = [SpokenMention(**m) for m in payload.get("mentions", [])]
        hits = [KeywordHit(**h) for h in payload.get("keyword_hits", [])]
        return MentionExtraction(mentions=mentions, keyword_hits=hits)

    def extract_mentions(self, transcript_text, watchlist, keywords) -> MentionExtraction:
        wl_lines = "\n".join(f"{cid} | {name} | {', '.join(aliases)}"
                             for cid, name, aliases in watchlist)
        prompt = _PROMPT.format(watchlist=wl_lines or "(empty)",
                                keywords=", ".join(keywords) or "(none)",
                                text=transcript_text)
        client = self._load()
        msg = client.messages.create(
            model=self._model, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        payload = json.loads(raw)
        return self._parse_response(payload, keywords)
