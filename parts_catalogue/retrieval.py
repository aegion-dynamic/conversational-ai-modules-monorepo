from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from parts_catalogue.models import Part, PartMatch


class PartRetriever(Protocol):
    """Black-box interface for finding parts from a natural-language description."""

    def find_parts(
        self,
        description: str,
        limit: int = 5,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[PartMatch]:
        """Find the closest catalogue parts for a description.

        Args:
            description (str): Natural-language part description from the user.
            limit (int): Maximum number of matches to return.
            context (Optional[Mapping[str, Any]]): Optional caller/channel context.

        Returns:
            Sequence[PartMatch]: Closest matching parts.
        """


class RetrievalError(RuntimeError):
    """Raised when the black-box retrieval integration cannot return matches."""


def map_object_to_part_match(result: Any) -> PartMatch:
    """Map a library-specific retrieval result to a PartMatch.

    Args:
        result (Any): Result from the underlying retrieval library.

    Returns:
        PartMatch: Normalized part match.

    Raises:
        ValueError: If the result cannot be converted.
    """

    if isinstance(result, PartMatch):
        return result

    if isinstance(result, Part):
        return PartMatch(part=result)

    if isinstance(result, Mapping):
        part = _extract_part(result)
        part_id = _first_present(result, ("part_id", "id", "sku", "lookup_key", "lookup_key_column_value"))
        score = _coerce_optional_float(_first_present(result, ("score", "similarity", "distance")))
        reason = _coerce_optional_str(_first_present(result, ("reason", "explanation", "data", "description")))

        if part is None and part_id is None:
            raise ValueError(f"Could not find a part id in retrieval result: {result}")

        return PartMatch(
            part_id=str(part_id) if part_id is not None else None,
            part=part,
            score=score,
            reason=reason,
            metadata=dict(result),
        )

    part_id = getattr(result, "part_id", None) or getattr(result, "id", None) or getattr(result, "sku", None)
    part = getattr(result, "part", None)
    if part is not None and not isinstance(part, Part):
        part = _extract_part(part)

    if part is None and part_id is None:
        raise ValueError(f"Could not convert retrieval result of type {type(result)!r} to PartMatch.")

    return PartMatch(
        part_id=str(part_id) if part_id is not None else None,
        part=part,
        score=_coerce_optional_float(getattr(result, "score", None)),
        reason=_coerce_optional_str(getattr(result, "reason", None)),
        metadata={},
    )


class CallablePartRetriever:
    """Adapter for an existing retrieval function or callable object.

    The wrapped callable remains a black box. This adapter only passes the user query,
    requested result limit, and optional context when those arguments are supported.
    """

    def __init__(
        self,
        retrieve: Callable[..., Iterable[Any]],
        result_mapper: Callable[[Any], PartMatch] = map_object_to_part_match,
    ):
        """Initialize the adapter.

        Args:
            retrieve (Callable[..., Iterable[Any]]): Existing retrieval function.
            result_mapper (Callable[[Any], PartMatch]): Mapper for raw results.
        """

        self.retrieve = retrieve
        self.result_mapper = result_mapper

    def find_parts(
        self,
        description: str,
        limit: int = 5,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[PartMatch]:
        """Find matching parts through the wrapped callable."""

        try:
            raw_results = self._call_retrieve(description=description, limit=limit, context=context or {})
            return tuple(self.result_mapper(result) for result in raw_results)
        except Exception as exc:
            raise RetrievalError("Part retrieval failed.") from exc

    def _call_retrieve(self, description: str, limit: int, context: Mapping[str, Any]) -> Iterable[Any]:
        """Call the wrapped retriever with the arguments its signature supports."""

        try:
            signature = inspect.signature(self.retrieve)
        except (TypeError, ValueError):
            return self.retrieve(description, limit)

        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        kwargs = {}

        query_arg_name = _find_supported_name(parameters, ("description", "query", "text", "message"))
        if query_arg_name is not None:
            kwargs[query_arg_name] = description

        limit_arg_name = _find_supported_name(parameters, ("limit", "top_k", "k", "n_results"))
        if limit_arg_name is not None:
            kwargs[limit_arg_name] = limit

        context_arg_name = _find_supported_name(parameters, ("context", "metadata"))
        if context_arg_name is not None:
            kwargs[context_arg_name] = context

        if accepts_kwargs:
            kwargs.setdefault("description", description)
            kwargs.setdefault("limit", limit)
            kwargs.setdefault("context", context)

        if kwargs:
            return self.retrieve(**kwargs)

        return self.retrieve(description, limit)


def _extract_part(value: Any) -> Optional[Part]:
    """Extract a Part from a mapping or object."""

    if isinstance(value, Part):
        return value

    if isinstance(value, Mapping):
        raw_part = value.get("part")
        if raw_part is not None:
            extracted_part = _extract_part(raw_part)
            if extracted_part is not None:
                return extracted_part

        part_id = _first_present(value, ("part_id", "id", "sku", "lookup_key", "lookup_key_column_value"))
        if part_id is None:
            return None

        name = _coerce_optional_str(_first_present(value, ("name", "part_name", "title")))
        description = _coerce_optional_str(_first_present(value, ("description", "data", "summary")))
        return Part(id=str(part_id), name=name, description=description, attributes=dict(value))

    part_id = getattr(value, "part_id", None) or getattr(value, "id", None) or getattr(value, "sku", None)
    if part_id is None:
        return None

    return Part(
        id=str(part_id),
        name=_coerce_optional_str(getattr(value, "name", None)),
        description=_coerce_optional_str(getattr(value, "description", None)),
        attributes={},
    )


def _first_present(values: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Return the first non-empty value in a mapping."""

    for name in names:
        if name in values and values[name] not in (None, ""):
            return values[name]
    return None


def _find_supported_name(parameters: Mapping[str, inspect.Parameter], names: Sequence[str]) -> Optional[str]:
    """Find an argument name supported by a callable signature."""

    for name in names:
        if name in parameters:
            return name
    return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    """Coerce a value to float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> Optional[str]:
    """Coerce a value to string when present."""

    if value is None:
        return None
    return str(value)
