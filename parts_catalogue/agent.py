from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from parts_catalogue.models import AgentResponse, Part, PartMatch, WhatsAppInboundMessage
from parts_catalogue.repository import PartRepository
from parts_catalogue.retrieval import PartRetriever


class PartsCatalogueAgent:
    """AI agent that answers natural-language part lookup requests."""

    def __init__(
        self,
        retriever: PartRetriever,
        repository: Optional[PartRepository] = None,
        default_limit: int = 5,
    ):
        """Initialize the agent.

        Args:
            retriever (PartRetriever): Black-box vector retrieval dependency.
            repository (Optional[PartRepository]): Optional store for hydrating part rows.
            default_limit (int): Default number of matches to return.
        """

        if default_limit < 1:
            raise ValueError("default_limit must be at least 1.")

        self.retriever = retriever
        self.repository = repository
        self.default_limit = default_limit

    def handle_message(
        self,
        message_text: str,
        sender: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AgentResponse:
        """Answer a plain inbound user message.

        Args:
            message_text (str): User's WhatsApp text.
            sender (Optional[str]): Sender identifier.
            context (Optional[Mapping[str, Any]]): Optional request metadata.

        Returns:
            AgentResponse: User-facing text and structured matches.
        """

        request_context = dict(context or {})
        if sender is not None:
            request_context["sender"] = sender

        description = message_text.strip()
        if not description:
            return AgentResponse(message="Please describe the part you are looking for.")

        matches = self.search_parts(description=description, limit=self.default_limit, context=request_context)
        return AgentResponse(message=self.format_matches(matches), matches=matches)

    def handle_whatsapp_message(self, inbound_message: WhatsAppInboundMessage) -> AgentResponse:
        """Answer a normalized WhatsApp message."""

        return self.handle_message(
            message_text=inbound_message.text,
            sender=inbound_message.sender,
            context=inbound_message.metadata,
        )

    def search_parts(
        self,
        description: str,
        limit: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[PartMatch]:
        """Find parts that match a user description.

        Args:
            description (str): User's natural-language description.
            limit (Optional[int]): Maximum number of matches.
            context (Optional[Mapping[str, Any]]): Request metadata.

        Returns:
            Sequence[PartMatch]: Hydrated matches when a repository is configured.
        """

        resolved_limit = limit or self.default_limit
        if resolved_limit < 1:
            raise ValueError("limit must be at least 1.")

        matches = tuple(self.retriever.find_parts(description=description, limit=resolved_limit, context=context or {}))
        if self.repository is None:
            return matches

        return self._hydrate_matches(matches)

    def format_matches(self, matches: Sequence[PartMatch]) -> str:
        """Format matches for a WhatsApp response."""

        if not matches:
            return "I could not find a close part match. Please add more detail, such as size, material, or part number."

        lines = ["Closest parts I found:"]
        for index, match in enumerate(matches, start=1):
            lines.append(self._format_match(index, match))
        return "\n".join(lines)

    def _hydrate_matches(self, matches: Sequence[PartMatch]) -> Sequence[PartMatch]:
        """Hydrate matches with Postgres part details while preserving retrieval order."""

        part_ids_to_load = [match.resolved_part_id for match in matches if match.part is None]
        hydrated_parts = self.repository.fetch_parts(part_ids_to_load)

        hydrated_matches = []
        for match in matches:
            if match.part is not None:
                hydrated_matches.append(match)
                continue

            part = hydrated_parts.get(match.resolved_part_id)
            hydrated_matches.append(match.with_part(part) if part is not None else match)

        return tuple(hydrated_matches)

    def _format_match(self, index: int, match: PartMatch) -> str:
        """Format a single match for compact mobile display."""

        part = match.part or Part(id=match.resolved_part_id)
        name = part.name or f"Part {part.id}"
        detail_parts = [f"{index}. {name}", f"ID: {part.id}"]

        if part.description:
            detail_parts.append(part.description)
        if match.score is not None:
            detail_parts.append(f"score: {match.score:.2f}")
        if match.reason and match.reason != part.description:
            detail_parts.append(match.reason)

        return " - ".join(detail_parts)
