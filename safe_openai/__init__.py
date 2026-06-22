"""Hardened drop-in wrapper for OpenAI streaming chat completions.

This module intentionally intercepts raw SSE bytes before OpenAI's stream
parser calls ``ServerSentEvent.json()`` and before Pydantic validates parsed
chunks. It is designed for Python 3.10+ and the modern ``openai-python`` SDK.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import warnings
from dataclasses import dataclass
from http.client import RemoteDisconnected
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Optional, TypeVar, Union, cast, overload

import httpx
import openai
from openai import OpenAI, Stream
from openai._streaming import ServerSentEvent
from openai.resources.chat.completions import Completions as ChatCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk

try:
    import httpcore
except Exception:  # pragma: no cover - httpx normally depends on httpcore.
    httpcore = None  # type: ignore[assignment]

T = TypeVar("T")
StreamFailureMode = Literal["repair_and_stop", "stop", "raise", "repair_or_raise"]
RepairCallback = Callable[["RepairEvent"], None]

__version__ = "0.1.0"

_DELTA_NULL_RE = re.compile(r'("delta"\s*:\s*)null\b')
_TRAILING_COMMA_BEFORE_CLOSE_RE = re.compile(r",(\s*[}\]])")
_MODEL_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_TESTED_OPENAI_MINORS = ("2.29.", "2.43.")
_DEFAULT_MAX_REPAIR_BYTES = 64_000
_DEFAULT_MAX_REPAIR_DEPTH = 256
_REASONING_FIELD_NAMES = (
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "reasoning_tokens",
    "thought",
    "thoughts",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairEvent:
    """Structured signal emitted when a truncated stream payload is repaired."""

    original_payload: str
    repaired_payload: str
    exception_type: str
    exception_message: str
    source: str
    repaired_bytes: int


def _network_stream_exceptions() -> tuple[type[BaseException], ...]:
    exceptions: list[type[BaseException]] = [
        httpx.ReadTimeout,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        httpx.ReadError,
        RemoteDisconnected,
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        OSError,
    ]

    if httpcore is not None:
        for name in ("RemoteProtocolError", "ReadError", "ReadTimeout", "NetworkError"):
            exc = getattr(httpcore, name, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                exceptions.append(exc)

    return tuple(dict.fromkeys(exceptions))


_NETWORK_STREAM_EXCEPTIONS = _network_stream_exceptions()


def build_tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Build portable low-level TCP keep-alive ``setsockopt`` triples.

    Linux exposes ``TCP_KEEPIDLE`` while macOS exposes ``TCP_KEEPALIVE`` for
    the idle timer. Unsupported options are skipped instead of failing client
    construction. Windows generally accepts ``SO_KEEPALIVE`` here but controls
    timers through IOCTL APIs, so timer options will usually be absent.
    """

    options: list[tuple[int, int, int]] = []

    def add(level: int, option_name: str, value: int) -> None:
        option = getattr(socket, option_name, None)
        if isinstance(option, int):
            options.append((level, option, value))

    try:
        options.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
    except AttributeError:
        pass

    tcp_level = getattr(socket, "IPPROTO_TCP", None)
    if isinstance(tcp_level, int):
        if hasattr(socket, "TCP_KEEPIDLE"):
            add(tcp_level, "TCP_KEEPIDLE", 5)
        elif hasattr(socket, "TCP_KEEPALIVE"):
            add(tcp_level, "TCP_KEEPALIVE", 5)

        add(tcp_level, "TCP_KEEPINTVL", 2)
        add(tcp_level, "TCP_KEEPCNT", 3)

    return options


def build_keepalive_http_client(
    *,
    timeout: httpx.Timeout | float | None = None,
    limits: httpx.Limits | None = None,
    http2: bool = False,
    verify: httpx.VerifyTypes = True,
    trust_env: bool = True,
) -> httpx.Client:
    """Create the ``httpx.Client`` injected into ``openai.OpenAI``."""

    transport = httpx.HTTPTransport(
        verify=verify,
        http1=True,
        http2=http2,
        limits=limits or httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0),
        trust_env=trust_env,
        socket_options=build_tcp_keepalive_socket_options(),
    )
    return httpx.Client(
        transport=transport,
        timeout=timeout if timeout is not None else httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        trust_env=trust_env,
    )


def _validate_stream_failure_mode(mode: str) -> None:
    if mode not in {"repair_and_stop", "stop", "raise", "repair_or_raise"}:
        raise ValueError(
            "stream_failure_mode must be one of: repair_and_stop, stop, raise, repair_or_raise"
        )


def _warn_if_untested_openai_version() -> None:
    version = getattr(openai, "__version__", "")
    if version and not version.startswith(_TESTED_OPENAI_MINORS):
        warnings.warn(
            "safe_openai 0.1.0 was tested against openai 2.29.x and 2.43.x. "
            f"You are running openai {version}. Private stream internals may have changed; "
            "run compatibility tests before production use.",
            RuntimeWarning,
            stacklevel=3,
        )


def _strip_data_prefix(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("data:"):
        return stripped.partition(":")[2].lstrip()
    return stripped


def sanitize_sse_payload(payload: str) -> str:
    """Sanitize one SSE data payload before JSON parsing."""

    stripped = payload.strip()
    return _DELTA_NULL_RE.sub(r'\1{"content": ""}', stripped)


def _json_string_state(text: str, *, max_depth: int = _DEFAULT_MAX_REPAIR_DEPTH) -> tuple[bool, bool, list[str]]:
    """Return ``(in_string, dangling_escape, missing_closers)`` for a JSON prefix.

    The stack tracks only structural delimiters outside strings. This makes the
    repair deterministic: strings are closed first, then the still-open JSON
    containers are closed from the innermost container outward.
    """

    in_string = False
    escape = False
    closers: list[str] = []

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            closers.append("}")
        elif char == "[":
            closers.append("]")
        elif char in ("}", "]") and closers and closers[-1] == char:
            closers.pop()

        if len(closers) > max_depth:
            raise ValueError(f"JSON repair depth exceeded {max_depth}")

    return in_string, escape, closers


def repair_truncated_json_payload(
    payload: str,
    *,
    max_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
    max_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
) -> str | None:
    """Predictively repair a truncated JSON payload.

    The repair is intentionally conservative:

    1. Sanitize known schema poison pills such as ``"delta": null``.
    2. If the prefix ends in a dangling key/value separator, inject the
       smallest structurally valid value.
    3. Close an unterminated JSON string, preserving a dangling final
       backslash by completing it as an escaped backslash.
    4. Close unmatched ``{``/``[`` containers from the innermost level out.
    5. Validate with ``json.loads``. Invalid repairs are discarded.
    """

    if len(payload.encode("utf-8", errors="replace")) > max_bytes:
        return None

    candidate = sanitize_sse_payload(_strip_data_prefix(payload))
    if not candidate or candidate == "[DONE]":
        return None

    if candidate.endswith("}"):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    candidate = candidate.rstrip()

    if candidate.endswith(":"):
        if re.search(r'"delta"\s*:\s*$', candidate):
            candidate += '{"content": ""}'
        else:
            candidate += "null"
    try:
        in_string, dangling_escape, closers = _json_string_state(candidate, max_depth=max_depth)
    except ValueError:
        return None
    if in_string:
        if dangling_escape:
            candidate += '\\"'
        else:
            candidate += '"'

    while closers:
        candidate += closers.pop()

    candidate = _TRAILING_COMMA_BEFORE_CLOSE_RE.sub(r"\1", candidate)

    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return candidate


class SafeLineIterator(Iterator[str]):
    """Iterate raw HTTPX stream chunks as sanitized SSE lines.

    The iterator accepts either byte chunks, string chunks, or pre-split lines.
    Empty lines and SSE comments such as ``: keep-alive`` are silently consumed,
    as requested. Consumers that need SSE event boundaries should use
    ``SafeSSEEventIterator``, which preserves boundaries internally while using
    the same sanitization rules.
    """

    def __init__(
        self,
        source: Iterable[bytes | str],
        *,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
        max_repair_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
    ) -> None:
        self._source = iter(source)
        self._buffer = ""
        self._closed = False
        self._repair_emitted = False
        self._stream_failure_mode = stream_failure_mode
        self._on_stream_repair = on_stream_repair
        self._max_repair_bytes = max_repair_bytes
        self._max_repair_depth = max_repair_depth
        self.last_valid_line: str | None = None
        self.last_partial_line: str | None = None

    def __iter__(self) -> "SafeLineIterator":
        return self

    def __next__(self) -> str:
        while True:
            if self._closed:
                raise StopIteration

            try:
                line = self._readline()
            except _NETWORK_STREAM_EXCEPTIONS as exc:
                self._closed = True
                if self._stream_failure_mode == "raise":
                    raise
                if self._repair_emitted:
                    raise StopIteration
                if self._stream_failure_mode == "stop":
                    raise StopIteration

                self.last_partial_line = self._buffer or self.last_partial_line
                original = self._buffer or self.last_partial_line or self.last_valid_line or ""
                repaired = repair_truncated_json_payload(
                    original,
                    max_bytes=self._max_repair_bytes,
                    max_depth=self._max_repair_depth,
                )
                self._buffer = ""
                self._repair_emitted = True
                if repaired is None:
                    if self._stream_failure_mode == "repair_or_raise":
                        raise exc
                    raise StopIteration
                self._emit_repair_event(original, repaired, exc, "SafeLineIterator")
                return f"data: {repaired}"

            if line is None:
                self._closed = True
                raise StopIteration

            sanitized = self._sanitize_line(line)
            if sanitized is None:
                continue

            self.last_valid_line = sanitized
            return sanitized

    def _readline(self) -> str | None:
        while "\n" not in self._buffer:
            try:
                chunk = next(self._source)
            except StopIteration:
                if not self._buffer:
                    return None
                line = self._buffer
                self._buffer = ""
                self.last_partial_line = line
                return line

            self._buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk

        line, self._buffer = self._buffer.split("\n", 1)
        return line.rstrip("\r")

    @staticmethod
    def _sanitize_line(line: str) -> str | None:
        stripped = line.strip()
        if not stripped or stripped == ": keep-alive" or stripped.startswith(":"):
            return None
        if stripped.startswith("data:"):
            prefix, _, value = stripped.partition(":")
            return f"{prefix}: {sanitize_sse_payload(value)}"
        return sanitize_sse_payload(stripped)

    def _emit_repair_event(self, original: str, repaired: str, exc: BaseException, source: str) -> None:
        event = RepairEvent(
            original_payload=original,
            repaired_payload=repaired,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            source=source,
            repaired_bytes=len(repaired.encode("utf-8", errors="replace")),
        )
        _LOGGER.warning(
            "Repaired truncated OpenAI SSE payload after %s in %s",
            event.exception_type,
            source,
            extra={"safe_openai_repair": event},
        )
        if self._on_stream_repair is not None:
            self._on_stream_repair(event)


class SafeSSEEventIterator(Iterator[ServerSentEvent]):
    """Decode sanitized SSE events and repair the final event on network loss."""

    def __init__(
        self,
        source: Iterable[bytes | str],
        *,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
        max_repair_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
    ) -> None:
        self._source = iter(source)
        self._buffer = ""
        self._event: str | None = None
        self._data: list[str] = []
        self._id: str | None = None
        self._retry: int | None = None
        self._queue: list[ServerSentEvent] = []
        self._closed = False
        self._repair_attempted = False
        self._stream_failure_mode = stream_failure_mode
        self._on_stream_repair = on_stream_repair
        self._max_repair_bytes = max_repair_bytes
        self._max_repair_depth = max_repair_depth
        self.last_valid_line: str | None = None
        self.last_partial_line: str | None = None

    def __iter__(self) -> "SafeSSEEventIterator":
        return self

    def __next__(self) -> ServerSentEvent:
        while not self._queue:
            if self._closed:
                raise StopIteration
            self._pump()
        return self._queue.pop(0)

    def _pump(self) -> None:
        try:
            line = self._readline()
        except _NETWORK_STREAM_EXCEPTIONS as exc:
            self._closed = True
            if self._stream_failure_mode == "raise":
                raise
            if self._stream_failure_mode == "stop":
                return
            self.last_partial_line = self._buffer or self.last_partial_line
            repaired = self._repair_current_event(exc)
            if repaired is not None:
                self._queue.append(repaired)
                return
            if self._stream_failure_mode == "repair_or_raise":
                raise exc
            return

        if line is None:
            self._closed = True
            if self._data:
                self._queue.append(self._emit_event())
            return

        stripped = line.strip()
        if not stripped or stripped == ": keep-alive" or stripped.startswith(":"):
            if not stripped and self._data:
                self._queue.append(self._emit_event())
            return

        field_name, separator, value = stripped.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field_name == "event":
            self._event = value
        elif field_name == "data":
            data = sanitize_sse_payload(value)
            self._data.append(data)
            self.last_valid_line = data
        elif field_name == "id":
            if "\0" not in value:
                self._id = value
        elif field_name == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass

    def _readline(self) -> str | None:
        while "\n" not in self._buffer:
            try:
                chunk = next(self._source)
            except StopIteration:
                if not self._buffer:
                    return None
                line = self._buffer
                self._buffer = ""
                self.last_partial_line = line
                return line

            self._buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk

        line, self._buffer = self._buffer.split("\n", 1)
        return line.rstrip("\r")

    def _emit_event(self) -> ServerSentEvent:
        event = ServerSentEvent(event=self._event, data="\n".join(self._data), id=self._id, retry=self._retry)
        self._event = None
        self._data = []
        self._retry = None
        return event

    def _repair_current_event(self, exc: BaseException) -> ServerSentEvent | None:
        if self._repair_attempted:
            return None
        self._repair_attempted = True

        payload = "\n".join(self._data).strip()
        if not payload:
            payload = _strip_data_prefix((self.last_partial_line or self.last_valid_line or "").strip())

        if not payload or payload == "[DONE]":
            return None

        try:
            json.loads(payload)
        except json.JSONDecodeError:
            pass
        else:
            return None

        repaired = repair_truncated_json_payload(
            payload,
            max_bytes=self._max_repair_bytes,
            max_depth=self._max_repair_depth,
        )
        if repaired is None:
            return None

        self._event = None
        self._data = []
        self._retry = None
        self._emit_repair_event(payload, repaired, exc, "SafeSSEEventIterator")
        return ServerSentEvent(event=None, data=repaired, id=self._id, retry=None)

    def _emit_repair_event(self, original: str, repaired: str, exc: BaseException, source: str) -> None:
        event = RepairEvent(
            original_payload=original,
            repaired_payload=repaired,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            source=source,
            repaired_bytes=len(repaired.encode("utf-8", errors="replace")),
        )
        _LOGGER.warning(
            "Repaired truncated OpenAI SSE payload after %s in %s",
            event.exception_type,
            source,
            extra={"safe_openai_repair": event},
        )
        if self._on_stream_repair is not None:
            self._on_stream_repair(event)


class SafeStream(Stream[T], Generic[T]):
    """OpenAI ``Stream`` subclass with raw SSE protection and reasoning flattening."""

    def __init__(
        self,
        source: Stream[T],
        *,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
        max_repair_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
    ) -> None:
        required = ("response", "_cast_to", "_client")
        missing = [name for name in required if not hasattr(source, name)]
        if missing:
            joined = ", ".join(missing)
            raise TypeError(f"Unsupported openai.Stream implementation; missing attributes: {joined}")

        _validate_stream_failure_mode(stream_failure_mode)
        self.response = source.response
        self._cast_to = source._cast_to
        self._client = source._client
        self._options = getattr(source, "_options", None)
        self._decoder = getattr(self._client, "_make_sse_decoder", lambda: None)()
        self._flatten_reasoning = flatten_reasoning
        self._stream_failure_mode = stream_failure_mode
        self._on_stream_repair = on_stream_repair
        self._max_repair_bytes = max_repair_bytes
        self._max_repair_depth = max_repair_depth
        self._iterator = self.__stream__()

    def _iter_events(self) -> Iterator[ServerSentEvent]:
        yield from SafeSSEEventIterator(
            self.response.iter_bytes(),
            stream_failure_mode=self._stream_failure_mode,
            on_stream_repair=self._on_stream_repair,
            max_repair_bytes=self._max_repair_bytes,
            max_repair_depth=self._max_repair_depth,
        )

    def __stream__(self) -> Iterator[T]:
        for item in super().__stream__():
            yield self._flatten_chunk(item) if self._flatten_reasoning else item

    def _flatten_chunk(self, item: T) -> T:
        model = getattr(item, "model", None)
        if isinstance(model, str) and not model.startswith(_MODEL_REASONING_PREFIXES):
            return item

        choices = getattr(item, "choices", None)
        if not choices:
            return item

        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            reasoning_text = _extract_reasoning_text(delta)
            if not reasoning_text:
                continue

            if content:
                if reasoning_text in content:
                    continue
                setattr(delta, "content", f"{reasoning_text}{content}")
            else:
                setattr(delta, "content", reasoning_text)

        return item


def _extract_reasoning_text(delta: Any) -> str | None:
    values: list[str] = []

    for field in _REASONING_FIELD_NAMES:
        value = _get_field(delta, field)
        if value is None:
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            for nested_name in ("content", "text", "summary"):
                nested = value.get(nested_name)
                if nested:
                    values.append(str(nested))

    joined = "".join(values).strip()
    return joined or None


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    direct = getattr(obj, name, None)
    if direct is not None:
        return direct

    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)

    return None


class SafeChatCompletions(ChatCompletions):
    """``chat.completions`` resource that wraps streaming responses."""

    def __init__(
        self,
        client: OpenAI,
        *,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
        max_repair_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
    ) -> None:
        super().__init__(client)
        self._flatten_reasoning = flatten_reasoning
        self._stream_failure_mode = stream_failure_mode
        self._on_stream_repair = on_stream_repair
        self._max_repair_bytes = max_repair_bytes
        self._max_repair_depth = max_repair_depth

    @overload
    def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> SafeStream[ChatCompletionChunk]:
        ...

    @overload
    def create(
        self, *args: Any, stream: Optional[Literal[False]] = False, **kwargs: Any
    ) -> ChatCompletion:
        ...

    def create(
        self, *args: Any, stream: Optional[bool] = None, **kwargs: Any
    ) -> Union[ChatCompletion, SafeStream[ChatCompletionChunk]]:
        if stream is not None:
            kwargs["stream"] = stream

        result = super().create(*args, **kwargs)
        if kwargs.get("stream") is True:
            return SafeStream(
                cast(Stream[ChatCompletionChunk], result),
                flatten_reasoning=self._flatten_reasoning,
                stream_failure_mode=self._stream_failure_mode,
                on_stream_repair=self._on_stream_repair,
                max_repair_bytes=self._max_repair_bytes,
                max_repair_depth=self._max_repair_depth,
            )
        return cast(ChatCompletion, result)


class SafeOpenAI(OpenAI):
    """Drop-in ``OpenAI`` client hardened for streaming Chat Completions."""

    def __init__(
        self,
        *args: Any,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = _DEFAULT_MAX_REPAIR_BYTES,
        max_repair_depth: int = _DEFAULT_MAX_REPAIR_DEPTH,
        http_client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None:
        _validate_stream_failure_mode(stream_failure_mode)
        _warn_if_untested_openai_version()
        self._owns_safe_http_client = http_client is None
        self._safe_flatten_reasoning = flatten_reasoning
        self._safe_stream_failure_mode = stream_failure_mode
        self._safe_on_stream_repair = on_stream_repair
        self._safe_max_repair_bytes = max_repair_bytes
        self._safe_max_repair_depth = max_repair_depth
        super().__init__(*args, http_client=http_client or build_keepalive_http_client(), **kwargs)
        self._install_safe_chat_completions()

    def _install_safe_chat_completions(self) -> None:
        chat_resource = self.chat
        chat_resource.__dict__["completions"] = SafeChatCompletions(
            self,
            flatten_reasoning=self._safe_flatten_reasoning,
            stream_failure_mode=self._safe_stream_failure_mode,
            on_stream_repair=self._safe_on_stream_repair,
            max_repair_bytes=self._safe_max_repair_bytes,
            max_repair_depth=self._safe_max_repair_depth,
        )


__all__ = [
    "SafeOpenAI",
    "SafeChatCompletions",
    "SafeLineIterator",
    "SafeSSEEventIterator",
    "SafeStream",
    "RepairEvent",
    "RepairCallback",
    "StreamFailureMode",
    "__version__",
    "build_keepalive_http_client",
    "build_tcp_keepalive_socket_options",
    "repair_truncated_json_payload",
    "sanitize_sse_payload",
]
