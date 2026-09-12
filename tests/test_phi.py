"""PHI redaction: what must be removed, and what must survive."""

from __future__ import annotations

import pytest

from clinical_rag.phi import redact, scan

SECRETS = [
    ("MRN 4471923", "MRN"),
    ("ssn 123-45-6789", "US_SSN"),
    ("615-555-0142", "PHONE_NUMBER"),
    ("j.smith@example.com", "EMAIL_ADDRESS"),
    ("DOB 3/4/1961", "DATE_OF_BIRTH"),
    ("member ID BCX449281", "MEMBER_ID"),
]


@pytest.mark.parametrize("text,entity", SECRETS)
def test_identifiers_are_removed(text: str, entity: str) -> None:
    result = redact(f"Question about {text} please")
    assert entity in result.entity_counts
    # the raw value must not survive anywhere in the output
    value = text.split()[-1]
    assert value not in result.text


def test_person_name_is_removed() -> None:
    result = redact("Is an MRI covered for John Smith?")
    assert "John" not in result.text
    assert result.entity_counts.get("PERSON") == 1


def test_durations_survive_redaction() -> None:
    """Over-redaction is a failure too: '8 weeks' decides the imaging policy."""
    result = redact("Patient had 8 weeks of conservative therapy")
    assert "8 weeks" in result.text


def test_clean_question_is_untouched() -> None:
    question = "What is the appeal window for a denied specialty drug?"
    result = redact(question)
    assert result.text == question
    assert not result.found_phi


def test_counts_never_contain_values() -> None:
    result = redact("MRN 4471923 for John Smith")
    serialised = str(result.entity_counts)
    assert "4471923" not in serialised
    assert "John" not in serialised


def test_scan_detects_leak_in_output() -> None:
    assert scan("Call the member at 615-555-0142", use_ner=False) == {
        "PHONE_NUMBER": 1
    }
