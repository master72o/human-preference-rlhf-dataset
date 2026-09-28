"""
Data Schemas for Human Preference & RLHF Dataset.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class PreferenceChoice(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"


@dataclass
class DimensionRating:
    truthfulness: float = 5.0  # 1.0 to 5.0 scale
    safety: float = 5.0
    instruction_following: float = 5.0
    helpfulness: float = 5.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AnnotationRecord:
    annotator_id: str
    preference: PreferenceChoice
    rationale: str
    confidence: int  # 1 to 5 scale
    ratings_a: DimensionRating = field(default_factory=DimensionRating)
    ratings_b: DimensionRating = field(default_factory=DimensionRating)
    error_tags_a: List[str] = field(default_factory=list)
    error_tags_b: List[str] = field(default_factory=list)
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "preference": self.preference.value if isinstance(self.preference, PreferenceChoice) else self.preference,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "ratings_a": self.ratings_a.to_dict() if isinstance(self.ratings_a, DimensionRating) else self.ratings_a,
            "ratings_b": self.ratings_b.to_dict() if isinstance(self.ratings_b, DimensionRating) else self.ratings_b,
            "error_tags_a": self.error_tags_a,
            "error_tags_b": self.error_tags_b,
            "is_simulated": self.is_simulated,
        }


@dataclass
class PreferenceItem:
    id: str
    prompt: str
    response_a: str
    response_b: str
    annotations: List[AnnotationRecord]
    domain: str = "general"
    consensus_preference: Optional[PreferenceChoice] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceItem":
        annotations = []
        for ann in data.get("annotations", []):
            pref = PreferenceChoice(ann["preference"])
            r_a = DimensionRating(**ann.get("ratings_a", {})) if isinstance(ann.get("ratings_a"), dict) else DimensionRating()
            r_b = DimensionRating(**ann.get("ratings_b", {})) if isinstance(ann.get("ratings_b"), dict) else DimensionRating()
            annotations.append(
                AnnotationRecord(
                    annotator_id=ann["annotator_id"],
                    preference=pref,
                    rationale=ann.get("rationale", ""),
                    confidence=ann.get("confidence", 5),
                    ratings_a=r_a,
                    ratings_b=r_b,
                    error_tags_a=ann.get("error_tags_a", []),
                    error_tags_b=ann.get("error_tags_b", []),
                    is_simulated=ann.get("is_simulated", False),
                )
            )

        cons = PreferenceChoice(data["consensus_preference"]) if data.get("consensus_preference") else None

        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response_a=data["response_a"],
            response_b=data["response_b"],
            annotations=annotations,
            domain=data.get("domain", "general"),
            consensus_preference=cons,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "domain": self.domain,
            "consensus_preference": self.consensus_preference.value if self.consensus_preference else None,
            "annotations": [ann.to_dict() for ann in self.annotations],
            "metadata": self.metadata,
        }
