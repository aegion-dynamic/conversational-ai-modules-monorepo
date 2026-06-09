from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Part:
    """Represents a part from the catalogue."""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartMatch:
    """Represents a catalogue search match returned by a retriever."""

    part_id: Optional[str] = None
    part: Optional[Part] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that a match can be linked back to a part."""

        if self.part_id is None and self.part is None:
            raise ValueError("PartMatch requires either part_id or part.")

    @property
    def resolved_part_id(self) -> str:
        """Return the part identifier regardless of whether the part is hydrated."""

        if self.part is not None:
            return self.part.id
        if self.part_id is None:
            raise ValueError("PartMatch does not contain a part id.")
        return self.part_id

    def with_part(self, part: Part) -> "PartMatch":
        """Return a copy of this match with hydrated part details."""

        return PartMatch(part_id=part.id, part=part, score=self.score, reason=self.reason, metadata=self.metadata)


@dataclass(frozen=True)
class AgentResponse:
    """Response produced by the parts catalogue agent."""

    message: str
    matches: Sequence[PartMatch] = field(default_factory=tuple)


@dataclass(frozen=True)
class WhatsAppInboundMessage:
    """Normalized inbound WhatsApp message."""

    sender: str
    text: str
    message_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
