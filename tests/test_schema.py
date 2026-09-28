"""
Unit tests for preference dataset schemas.
"""

import pytest
from dataset.schema import PreferenceChoice, DimensionRating, AnnotationRecord, PreferenceItem


def test_preference_choice_enum():
    assert PreferenceChoice.A.value == "A"
    assert PreferenceChoice.B.value == "B"
    assert PreferenceChoice.TIE.value == "tie"


def test_annotation_record_serialization():
    ann = AnnotationRecord(
        annotator_id="ann_1",
        preference=PreferenceChoice.A,
        rationale="Clear winner",
        confidence=5,
        ratings_a=DimensionRating(truthfulness=5.0, safety=5.0, instruction_following=5.0, helpfulness=5.0),
        ratings_b=DimensionRating(truthfulness=1.0, safety=5.0, instruction_following=2.0, helpfulness=1.0),
        error_tags_b=["factual_error"],
        is_simulated=False
    )
    d = ann.to_dict()
    assert d["annotator_id"] == "ann_1"
    assert d["preference"] == "A"
    assert d["ratings_a"]["truthfulness"] == 5.0
    assert d["error_tags_b"] == ["factual_error"]


def test_preference_item_deserialization():
    raw = {
        "id": "p1",
        "prompt": "What is 2+2?",
        "response_a": "4",
        "response_b": "5",
        "domain": "Math",
        "consensus_preference": "A",
        "annotations": [
            {
                "annotator_id": "ann_1",
                "preference": "A",
                "rationale": "Correct math",
                "confidence": 5,
                "ratings_a": {"truthfulness": 5.0},
                "ratings_b": {"truthfulness": 1.0},
                "error_tags_b": ["math_error"],
                "is_simulated": False
            }
        ]
    }
    item = PreferenceItem.from_dict(raw)
    assert item.id == "p1"
    assert item.consensus_preference == PreferenceChoice.A
    assert len(item.annotations) == 1
    assert item.annotations[0].preference == PreferenceChoice.A
