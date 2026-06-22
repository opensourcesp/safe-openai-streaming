from __future__ import annotations

import json
from http.client import RemoteDisconnected
from typing import Iterator

import httpx
import pytest
from openai import Stream
from openai.types.chat import ChatCompletionChunk

from safe_openai import (
    RepairEvent,
    SafeLineIterator,
    SafeOpenAI,
    SafeSSEEventIterator,
    SafeStream,
    __version__,
    build_tcp_keepalive_socket_options,
    repair_truncated_json_payload,
    sanitize_sse_payload,
)


def _chunk_payload(delta: object, *, model: str = "gpt-4o-mini") -> dict[str, object]:
    return {
        "id": "chunk_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta}],
    }


def _stream_items(response: httpx.Response, **safe_stream_kwargs: object) -> list[ChatCompletionChunk]:
    client = SafeOpenAI(api_key="test-key")
    try:
        source = Stream(cast_to=ChatCompletionChunk, response=response, client=client)
        return list(SafeStream(source, **safe_stream_kwargs))
    finally:
        client.close()


def _response_from_body(body: bytes) -> httpx.Response:
    request = httpx.Request("GET", "https://api.openai.test/chat/completions")
    return httpx.Response(200, request=request, content=body)


class FailingByteStream(httpx.SyncByteStream):
    def __init__(self, exc: BaseException, prefix: bytes) -> None:
        self._exc = exc
        self._prefix = prefix

    def __iter__(self) -> Iterator[bytes]:
        yield self._prefix
        raise self._exc


def _response_from_failure(exc: BaseException, prefix: bytes) -> httpx.Response:
    request = httpx.Request("GET", "https://api.openai.test/chat/completions")
    return httpx.Response(200, request=request, stream=FailingByteStream(exc, prefix))


def test_version_is_release() -> None:
    assert __version__ == "0.1.0"


def test_socket_options_are_well_formed() -> None:
    options = build_tcp_keepalive_socket_options()
    assert options
    assert all(len(option) == 3 for option in options)
    assert all(isinstance(value, int) for option in options for value in option)


def test_delta_null_is_sanitized_before_pydantic_validation() -> None:
    payload = _chunk_payload(None)
    body = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()

    items = _stream_items(_response_from_body(body))

    assert len(items) == 1
    assert items[0].choices[0].delta.content == ""


def test_keep_alive_comments_are_ignored() -> None:
    payload = _chunk_payload({"content": "ok"})
    body = f": keep-alive\n\ndata: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()

    items = _stream_items(_response_from_body(body))

    assert len(items) == 1
    assert items[0].choices[0].delta.content == "ok"


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timeout"),
        httpx.NetworkError("network error"),
        httpx.RemoteProtocolError("peer closed connection"),
        RemoteDisconnected("remote disconnected"),
    ],
)
def test_network_cut_repairs_last_partial_chunk(exc: BaseException) -> None:
    prefix = (
        b'data: {"id":"chunk_1","object":"chat.completion.chunk","created":0,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"abc'
    )
    repairs: list[RepairEvent] = []

    items = _stream_items(
        _response_from_failure(exc, prefix),
        on_stream_repair=repairs.append,
    )

    assert len(items) == 1
    assert items[0].choices[0].delta.content == "abc"
    assert len(repairs) == 1
    assert repairs[0].repaired_payload.endswith('"}}]}')


def test_failure_mode_stop_suppresses_repair() -> None:
    prefix = b'data: {"choices":[{"delta":{"content":"abc'

    items = _stream_items(
        _response_from_failure(httpx.RemoteProtocolError("cut"), prefix),
        stream_failure_mode="stop",
    )

    assert items == []


def test_failure_mode_raise_propagates_original_exception() -> None:
    prefix = b'data: {"choices":[{"delta":{"content":"abc'

    with pytest.raises(httpx.RemoteProtocolError):
        _stream_items(
            _response_from_failure(httpx.RemoteProtocolError("cut"), prefix),
            stream_failure_mode="raise",
        )


def test_failure_mode_repair_or_raise_raises_when_irreparable() -> None:
    prefix = b'data: {"choices":[{"delta"'

    with pytest.raises(httpx.RemoteProtocolError):
        _stream_items(
            _response_from_failure(httpx.RemoteProtocolError("cut"), prefix),
            stream_failure_mode="repair_or_raise",
        )


def test_repair_payload_respects_max_bytes() -> None:
    payload = '{"choices":[{"delta":{"content":"' + ("x" * 100) + '"'

    repaired = repair_truncated_json_payload(payload, max_bytes=16)

    assert repaired is None


def test_repair_payload_balances_strings_and_containers() -> None:
    payload = '{"choices":[{"delta":{"content":"hello'

    repaired = repair_truncated_json_payload(payload)

    assert repaired == '{"choices":[{"delta":{"content":"hello"}}]}'
    assert json.loads(repaired)["choices"][0]["delta"]["content"] == "hello"


def test_safe_line_iterator_repairs_after_network_cut() -> None:
    class Lines:
        def __iter__(self) -> Iterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"abc'
            raise httpx.NetworkError("cut")

    lines = list(SafeLineIterator(Lines()))

    assert lines == ['data: {"choices":[{"delta":{"content":"abc"}}]}']


def test_safe_sse_iterator_sanitizes_delta_null() -> None:
    events = list(SafeSSEEventIterator(['data: {"choices":[{"delta":null}]}\n\n']))

    assert json.loads(events[0].data)["choices"][0]["delta"] == {"content": ""}


def test_reasoning_flattening_copies_extra_fields_to_content() -> None:
    payload = _chunk_payload({"content": "answer", "thought": "reason:"}, model="o3-mini")
    body = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()

    items = _stream_items(_response_from_body(body), flatten_reasoning=True)

    assert items[0].choices[0].delta.content == "reason:answer"


def test_sanitize_sse_payload_rewrites_delta_null() -> None:
    assert sanitize_sse_payload('{"choices":[{"delta":null}]}') == (
        '{"choices":[{"delta":{"content": ""}}]}'
    )
