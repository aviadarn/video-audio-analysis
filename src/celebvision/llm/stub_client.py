from celebvision.interfaces import MentionExtraction
from celebvision.models import SpokenMention, KeywordHit


class StubLLMClient:
    def extract_mentions(self, transcript_text, watchlist, keywords) -> MentionExtraction:
        low = transcript_text.lower()
        mentions = []
        for canonical_id, name, aliases in watchlist:
            needles = [name.split()[0].lower()] + [a.lower() for a in aliases]
            if any(n in low for n in needles):
                mentions.append(SpokenMention(
                    name=name, canonical_id=canonical_id, confidence=0.99,
                    evidence_span=transcript_text))
        hits = []
        for kw in keywords:
            c = low.count(kw.lower())
            if c > 0:
                hits.append(KeywordHit(keyword=kw, count=c, spans=[transcript_text]))
        return MentionExtraction(mentions=mentions, keyword_hits=hits)
