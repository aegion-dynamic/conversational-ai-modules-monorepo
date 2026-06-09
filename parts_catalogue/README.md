# Parts Catalogue Agent

This package provides a Python agent for WhatsApp-driven parts catalogue lookup.
It intentionally treats vector search as a black box: plug in any existing
retrieval library through `PartRetriever` or `CallablePartRetriever`.

## Components

- `PartsCatalogueAgent`: Orchestrates user messages, retrieval, optional
  Postgres hydration, and response formatting.
- `PartRetriever`: Protocol implemented by the existing vector DB retrieval
  library.
- `CallablePartRetriever`: Adapter for existing retrieval functions that return
  dicts, `Part`, or `PartMatch` objects.
- `PostgresPartsRepository`: Optional hydrator that loads full part rows from a
  Postgres catalogue when retrieval returns only part IDs.
- `WhatsAppPartsCatalogueHandler`: Provider-neutral WhatsApp handler with Twilio
  and Meta payload helpers.

## Example

```python
from parts_catalogue import (
    CallablePartRetriever,
    PartsCatalogueAgent,
    PostgresPartsRepository,
    WhatsAppPartsCatalogueHandler,
)
from parts_catalogue.config import load_postgres_repository_config


def existing_vector_search(description: str, limit: int):
    # Call your existing retrieval library here. The implementation is opaque
    # to the agent as long as results include an id/part_id/sku or full part.
    return [{"part_id": "BRG-100", "score": 0.94, "reason": "Matches bore and bearing keywords"}]


retriever = CallablePartRetriever(existing_vector_search)
repository = PostgresPartsRepository(load_postgres_repository_config())
agent = PartsCatalogueAgent(retriever=retriever, repository=repository)
handler = WhatsAppPartsCatalogueHandler(agent)

# Twilio webhook example:
twiml = handler.handle_twilio_form({"From": "whatsapp:+15550001111", "Body": "stainless bearing 20mm"})
```

## Environment variables for Postgres hydration

```sh
PARTS_CATALOGUE_POSTGRES_HOST=localhost
PARTS_CATALOGUE_POSTGRES_PORT=5432
PARTS_CATALOGUE_POSTGRES_USER=postgres
PARTS_CATALOGUE_POSTGRES_PASSWORD=password
PARTS_CATALOGUE_POSTGRES_DATABASE=parts
PARTS_CATALOGUE_PARTS_TABLE=parts
PARTS_CATALOGUE_PART_ID_COLUMN=id
PARTS_CATALOGUE_PART_NAME_COLUMN=name
PARTS_CATALOGUE_PART_DESCRIPTION_COLUMN=description
```
