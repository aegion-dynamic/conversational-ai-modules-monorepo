from typing import Mapping, Sequence

from parts_catalogue.agent import PartsCatalogueAgent
from parts_catalogue.models import Part, PartMatch, WhatsAppInboundMessage
from parts_catalogue.repository import PartRepository
from parts_catalogue.retrieval import CallablePartRetriever


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def find_parts(self, description: str, limit: int = 5, context=None) -> Sequence[PartMatch]:
        self.calls.append({"description": description, "limit": limit, "context": context})
        return (
            PartMatch(part_id="BRG-100", score=0.94, reason="bearing match"),
            PartMatch(part=Part(id="BLT-200", name="M8 Bolt", description="Stainless steel bolt"), score=0.81),
        )


class FakeRepository:
    def __init__(self):
        self.loaded_part_ids = []

    def fetch_parts(self, part_ids: Sequence[str]) -> Mapping[str, Part]:
        self.loaded_part_ids.extend(part_ids)
        return {
            "BRG-100": Part(id="BRG-100", name="20mm Bearing", description="Sealed stainless bearing"),
        }


def test_agent_hydrates_part_ids_and_formats_response():
    retriever = FakeRetriever()
    repository: PartRepository = FakeRepository()
    agent = PartsCatalogueAgent(retriever=retriever, repository=repository, default_limit=2)

    response = agent.handle_message(" stainless bearing 20mm ", sender="whatsapp:+15550001111")

    assert retriever.calls == [
        {
            "description": "stainless bearing 20mm",
            "limit": 2,
            "context": {"sender": "whatsapp:+15550001111"},
        }
    ]
    assert repository.loaded_part_ids == ["BRG-100"]
    assert response.matches[0].part is not None
    assert "20mm Bearing" in response.message
    assert "M8 Bolt" in response.message


def test_agent_returns_prompt_for_empty_message_without_retrieval():
    retriever = FakeRetriever()
    agent = PartsCatalogueAgent(retriever=retriever)

    response = agent.handle_message("  ")

    assert response.message == "Please describe the part you are looking for."
    assert response.matches == ()
    assert retriever.calls == []


def test_agent_handles_normalized_whatsapp_message():
    retriever = FakeRetriever()
    agent = PartsCatalogueAgent(retriever=retriever, default_limit=1)

    inbound_message = WhatsAppInboundMessage(
        sender="whatsapp:+15550001111",
        text="rubber gasket",
        message_id="message-1",
        metadata={"provider": "twilio"},
    )
    response = agent.handle_whatsapp_message(inbound_message)

    assert response.matches[0].resolved_part_id == "BRG-100"
    assert retriever.calls[0]["context"] == {"provider": "twilio", "sender": "whatsapp:+15550001111"}


def test_callable_retriever_maps_black_box_results():
    def existing_retrieval_function(query: str, top_k: int):
        assert query == "hex screw"
        assert top_k == 3
        return [{"sku": "SCR-10", "name": "Hex Screw", "description": "10mm hex screw", "similarity": "0.88"}]

    retriever = CallablePartRetriever(existing_retrieval_function)

    matches = retriever.find_parts("hex screw", limit=3)

    assert len(matches) == 1
    assert matches[0].resolved_part_id == "SCR-10"
    assert matches[0].part is not None
    assert matches[0].score == 0.88
