"""PHI redaction: strip patient identifiers before text reaches the model.

Two layers, deliberately:

1. **Deterministic patterns** for structured identifiers — MRN, member ID,
   SSN, phone, email, claim number. Regexes do not have a bad day; for
   structured data they beat a statistical model on recall.
2. **Presidio's named-entity recognition** for the unstructured leftovers —
   people's names, dates, locations — which patterns cannot catch.

Redaction is applied to the *input* on the way in, and the same detectors are
run over the *output* on the way back as a leak check. Nothing here stores the
values it finds: audit records get entity types and counts, never the PHI
itself, because a log of what you redacted is still a log of PHI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# Structured identifiers, checked before the NER model runs.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("MRN", re.compile(r"\b(?:MRN|mrn)[:\s#]*([A-Z]{0,3}\d{6,10})\b")),
    ("MEMBER_ID", re.compile(r"\b(?:member(?:\s+id)?|subscriber)[:\s#]*([A-Z]{2,4}\d{6,12})\b", re.I)),
    ("CLAIM_ID", re.compile(r"\b(?:claim)[:\s#]*([A-Z]{0,3}\d{8,12})\b", re.I)),
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE_NUMBER", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("DATE_OF_BIRTH", re.compile(r"\b(?:DOB|dob|date of birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    # Calendar dates only. Durations ("8 weeks of PT") are clinically load-bearing
    # and must survive redaction, which is why DATE_TIME is not sent to the NER
    # model: it cannot tell "August 2026" from "6 weeks" reliably enough.
    ("DATE", re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    (
        "DATE",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
        ),
    ),
)

# Entities Presidio should look for, beyond what the patterns already cover.
NER_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "LOCATION",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "CREDIT_CARD",
    "IP_ADDRESS",
)

# Presidio scores 0-1; below this we ignore the finding.
MIN_SCORE = 0.5


@dataclass
class RedactionResult:
    """Redacted text plus a PHI-free summary of what was removed."""

    text: str
    entity_counts: dict[str, int] = field(default_factory=dict)

    @property
    def found_phi(self) -> bool:
        return bool(self.entity_counts)

    def summary(self) -> str:
        if not self.entity_counts:
            return "none"
        return ", ".join(
            f"{entity}x{count}" for entity, count in sorted(self.entity_counts.items())
        )


@lru_cache(maxsize=1)
def _analyzer():
    """Build the Presidio analyzer (slow to start, so built once)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def _apply_patterns(text: str, counts: dict[str, int]) -> str:
    for label, pattern in PATTERNS:
        text, hits = pattern.subn(f"<{label}>", text)
        if hits:
            counts[label] = counts.get(label, 0) + hits
    return text


def _apply_ner(text: str, counts: dict[str, int]) -> str:
    results = [
        result
        for result in _analyzer().analyze(
            text=text, entities=list(NER_ENTITIES), language="en"
        )
        if result.score >= MIN_SCORE
    ]

    # Replace from the end so earlier offsets stay valid.
    for result in sorted(results, key=lambda r: r.start, reverse=True):
        if "<" in text[result.start : result.end]:
            continue  # already redacted by a pattern
        text = f"{text[: result.start]}<{result.entity_type}>{text[result.end :]}"
        counts[result.entity_type] = counts.get(result.entity_type, 0) + 1

    return text


def redact(text: str, use_ner: bool = True) -> RedactionResult:
    """Return the text with identifiers replaced by <TYPE> placeholders."""
    counts: dict[str, int] = {}
    redacted = _apply_patterns(text, counts)
    if use_ner:
        redacted = _apply_ner(redacted, counts)
    return RedactionResult(text=redacted, entity_counts=counts)


def scan(text: str, use_ner: bool = True) -> dict[str, int]:
    """Detect identifiers without rewriting — used as an output leak check."""
    return redact(text, use_ner=use_ner).entity_counts
