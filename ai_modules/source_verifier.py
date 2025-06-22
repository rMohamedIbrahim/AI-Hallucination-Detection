from urllib.parse import urlparse
import re
from typing import Dict

class SourceVerifier:
    """Verifies and ranks external sources."""

    def __init__(self):
        self.trust_scores = {
            "wikipedia.org": 0.8,
            ".edu": 0.9,
            ".gov": 0.9,
            ".org": 0.7,
            "news.google.com": 0.8,
            "scholar.google.com": 0.9
        }

    def calculate_trust_score(self, url: str, content: str) -> float:
        """Calculate trust score for a source."""
        domain = urlparse(url).netloc.lower()
        base_score = next((score for domain_part, score in self.trust_scores.items()
                          if domain_part in domain), 0.5)

        factors = {
            "length": min(len(content.split()) / 500, 1.0) if content else 0.0,
            "citations": len(re.findall(r'\[\d+\]|\[citation needed\]', content)) > 0 if content else False,
            "dates": len(re.findall(r'\b\d{4}\b', content)) > 0 if content else False,
            "links": len(re.findall(r'http[s]?://', content)) > 0 if content else False
        }

        quality_score = sum(factors.values()) / len(factors)
        return (base_score * 0.7 + quality_score * 0.3)