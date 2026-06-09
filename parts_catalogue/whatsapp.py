from __future__ import annotations

from html import escape
from typing import Any, Mapping, Optional

from parts_catalogue.agent import PartsCatalogueAgent
from parts_catalogue.models import AgentResponse, WhatsAppInboundMessage


class WhatsAppPayloadError(ValueError):
    """Raised when a WhatsApp webhook payload cannot be parsed."""


class WhatsAppWebhookParser:
    """Parses provider-specific WhatsApp webhook payloads."""

    @staticmethod
    def parse_twilio_form(form_data: Mapping[str, Any]) -> WhatsAppInboundMessage:
        """Parse a Twilio WhatsApp inbound message form.

        Args:
            form_data (Mapping[str, Any]): Form fields from Twilio.

        Returns:
            WhatsAppInboundMessage: Normalized inbound message.
        """

        sender = _require_string(form_data, "From")
        text = _require_string(form_data, "Body")
        message_id = _optional_string(form_data, "MessageSid")
        return WhatsAppInboundMessage(sender=sender, text=text, message_id=message_id, metadata=dict(form_data))

    @staticmethod
    def parse_meta_payload(payload: Mapping[str, Any]) -> WhatsAppInboundMessage:
        """Parse a Meta WhatsApp Cloud API webhook payload."""

        try:
            value = payload["entry"][0]["changes"][0]["value"]
            message = value["messages"][0]
            text = message["text"]["body"]
            sender = message["from"]
            message_id = message.get("id")
        except (KeyError, IndexError, TypeError) as exc:
            raise WhatsAppPayloadError("Invalid Meta WhatsApp payload.") from exc

        if not isinstance(sender, str) or not isinstance(text, str):
            raise WhatsAppPayloadError("Meta WhatsApp payload is missing sender or text.")

        return WhatsAppInboundMessage(sender=sender, text=text, message_id=message_id, metadata=dict(payload))


class WhatsAppResponseFormatter:
    """Formats agent responses for WhatsApp providers."""

    @staticmethod
    def to_twilio_twiml(message: str) -> str:
        """Build a TwiML response body for Twilio webhooks."""

        return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{escape(message)}</Message></Response>'

    @staticmethod
    def to_meta_message(to: str, message: str, preview_url: bool = False) -> Mapping[str, Any]:
        """Build a Meta WhatsApp Cloud API text message payload."""

        return {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": preview_url, "body": message},
        }


class WhatsAppPartsCatalogueHandler:
    """Coordinates webhook parsing, catalogue lookup, and provider response formatting."""

    def __init__(self, agent: PartsCatalogueAgent):
        """Initialize the handler.

        Args:
            agent (PartsCatalogueAgent): Parts catalogue agent.
        """

        self.agent = agent

    def handle_twilio_form(self, form_data: Mapping[str, Any]) -> str:
        """Handle a Twilio form payload and return TwiML."""

        response = self.agent.handle_whatsapp_message(WhatsAppWebhookParser.parse_twilio_form(form_data))
        return WhatsAppResponseFormatter.to_twilio_twiml(response.message)

    def handle_meta_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Handle a Meta payload and return an outbound text payload."""

        inbound_message = WhatsAppWebhookParser.parse_meta_payload(payload)
        response = self.agent.handle_whatsapp_message(inbound_message)
        return WhatsAppResponseFormatter.to_meta_message(to=inbound_message.sender, message=response.message)

    def handle_inbound_message(self, inbound_message: WhatsAppInboundMessage) -> AgentResponse:
        """Handle an already-normalized inbound WhatsApp message."""

        return self.agent.handle_whatsapp_message(inbound_message)


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    """Read a required string value."""

    value = _optional_string(mapping, key)
    if value is None:
        raise WhatsAppPayloadError(f"Missing required WhatsApp field: {key}")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> Optional[str]:
    """Read an optional string value."""

    value = mapping.get(key)
    if value is None:
        return None
    return str(value)
