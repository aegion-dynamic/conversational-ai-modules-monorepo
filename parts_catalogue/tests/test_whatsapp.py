import pytest

from parts_catalogue.agent import PartsCatalogueAgent
from parts_catalogue.models import PartMatch
from parts_catalogue.whatsapp import (
    WhatsAppPartsCatalogueHandler,
    WhatsAppPayloadError,
    WhatsAppResponseFormatter,
    WhatsAppWebhookParser,
)


class StaticRetriever:
    def find_parts(self, description: str, limit: int = 5, context=None):
        return (PartMatch(part_id="VALVE-1", reason=f"matched {description}"),)


def test_parse_twilio_form_and_format_twiml():
    inbound_message = WhatsAppWebhookParser.parse_twilio_form(
        {"From": "whatsapp:+15550001111", "Body": "valve < 10mm", "MessageSid": "SM123"}
    )

    assert inbound_message.sender == "whatsapp:+15550001111"
    assert inbound_message.text == "valve < 10mm"
    assert inbound_message.message_id == "SM123"

    twiml = WhatsAppResponseFormatter.to_twilio_twiml("Use A < B")

    assert "<Message>Use A &lt; B</Message>" in twiml


def test_handle_twilio_form_returns_catalogue_response():
    handler = WhatsAppPartsCatalogueHandler(PartsCatalogueAgent(StaticRetriever()))

    twiml = handler.handle_twilio_form({"From": "whatsapp:+15550001111", "Body": "valve"})

    assert "Closest parts I found" in twiml
    assert "VALVE-1" in twiml


def test_parse_meta_payload_and_format_response():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "15550001111",
                                    "id": "wamid.123",
                                    "text": {"body": "pump impeller"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    handler = WhatsAppPartsCatalogueHandler(PartsCatalogueAgent(StaticRetriever()))
    response_payload = handler.handle_meta_payload(payload)

    assert response_payload["messaging_product"] == "whatsapp"
    assert response_payload["to"] == "15550001111"
    assert "VALVE-1" in response_payload["text"]["body"]


def test_parse_meta_payload_rejects_invalid_payload():
    with pytest.raises(WhatsAppPayloadError):
        WhatsAppWebhookParser.parse_meta_payload({"entry": []})
