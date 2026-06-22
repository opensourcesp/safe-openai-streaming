# Safe OpenAI Streaming Client

Production-oriented Python wrapper for `openai-python` that hardens streaming Chat Completions against common real-world failure modes.

Current package version: `0.1.0`.

- Some NAT and firewall idle connection drops during long `stream=True` responses.
- `delta: null` payloads that can break downstream Pydantic validation.
- Truncated Server-Sent Events after a mid-stream network cut.
- Optional reasoning-field flattening for legacy consumers that only read `choices[0].delta.content`.

The wrapper is designed as a near drop-in replacement for `openai.OpenAI`:

```python
from safe_openai import SafeOpenAI

client = SafeOpenAI(api_key="YOUR_API_KEY", flatten_reasoning=True)

stream = client.chat.completions.create(
    model="o3-mini",
    messages=[{"role": "user", "content": "Explain TCP keep-alive in one paragraph."}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Why This Exists

The official OpenAI Python SDK already handles normal streaming well. The hard cases appear in production environments with long-lived connections:

- Cloud NATs, corporate proxies, mobile networks, and load balancers may silently drop idle TCP connections.
- A stream can be interrupted after bytes have been received but before the final JSON object is complete.
- Some experimental or reasoning-model payloads may include fields older consumers do not expect.
- If malformed data reaches the SDK parser, application code can see a hard exception instead of a clean stream termination.

`SafeOpenAI` adds a defensive layer below the SDK parser. It intercepts raw SSE bytes, sanitizes known bad shapes, and attempts one conservative final repair when the network dies mid-event.

## Features

### TCP Keep-Alive Transport

`SafeOpenAI` injects a custom `httpx.Client` backed by `httpx.HTTPTransport(socket_options=...)`. These options help reduce the likelihood of idle TCP connection drops and can detect dead peers sooner, but they cannot override every proxy, corporate firewall, NAT gateway, or load balancer policy.

Configured options:

| Option | Value | Purpose |
| --- | ---: | --- |
| `SOL_SOCKET / SO_KEEPALIVE` | `1` | Enable TCP keep-alive probes. |
| `TCP_KEEPIDLE` or `TCP_KEEPALIVE` | `5` | Start probing after 5 seconds of inactivity. |
| `TCP_KEEPINTVL` | `2` | Send probes every 2 seconds. |
| `TCP_KEEPCNT` | `3` | Treat the connection as dead after 3 failed probes. |

Unsupported socket options are skipped safely. This makes the client tolerant across Linux, macOS, and Windows.

### Raw SSE Interception

The wrapper replaces `client.chat.completions` with `SafeChatCompletions`. When `create(..., stream=True)` is used, the returned SDK `Stream` is wrapped in `SafeStream`.

`SafeStream` overrides the event iterator used before `ServerSentEvent.json()` runs. This is the key safety point: malformed SSE payloads are handled before the OpenAI SDK converts them into typed response chunks.

### Line and Event Sanitization

The raw stream layer:

- Ignores empty lines.
- Ignores SSE comments such as `: keep-alive`.
- Rewrites `"delta": null` and `"delta":null` to `"delta": {"content": ""}`.
- Preserves normal SDK chunk parsing after sanitization.

### Predictive JSON Repair

If a stream fails with `httpx.ReadTimeout`, `httpx.NetworkError`, `httpx.RemoteProtocolError`, `httpx.ReadError`, matching `httpcore` read/protocol errors, `http.client.RemoteDisconnected`, or common socket-level disconnect errors, the iterator can recover the last partial SSE payload.

The repair algorithm is intentionally conservative:

1. Strip an optional `data:` prefix.
2. Apply known schema sanitizers such as `delta: null`.
3. If the prefix ends after a key/value separator, inject the smallest safe JSON value.
4. Walk the JSON prefix once while tracking string state, escape state, and open containers.
5. Close an unterminated string if needed.
6. Close unmatched `{` and `[` containers from the innermost container outward.
7. Remove trailing commas before structural closers.
8. Validate the repaired candidate with `json.loads`.
9. Emit the repaired event once, then terminate the stream cleanly.

If validation fails, no synthetic chunk is emitted. The default policy still ends the stream cleanly instead of crashing the caller.

The failure policy is configurable:

| Mode | Behavior |
| --- | --- |
| `repair_and_stop` | Default. Try one repair, emit it if valid, then stop cleanly. |
| `stop` | Do not repair. Stop cleanly on stream transport failure. |
| `raise` | Propagate the original stream transport exception. |
| `repair_or_raise` | Try one repair. If no valid repair exists, re-raise the original exception. |

When a repair is emitted, the library logs a warning and can call a user-provided callback with a structured `RepairEvent`.

### Reasoning Flattening

Some reasoning-capable models or intermediary systems may expose reasoning-like fields outside `delta.content`, for example:

- `reasoning`
- `reasoning_content`
- `reasoning_text`
- `reasoning_tokens`
- `thought`
- `thoughts`

With `flatten_reasoning=True`, `SafeStream` copies recognized reasoning-like text into `choices[*].delta.content` for model names that look reasoning-oriented, such as `o1`, `o3`, `o4`, and `gpt-5`.

This is an opt-in best-effort compatibility shim for older streaming consumers that only concatenate `delta.content`. It only duplicates fields already present in the streamed payload; it does not expose hidden reasoning, guarantee future reasoning field names, or define a stable reasoning API.

## Installation

Install from PyPI:

```bash
pip install safe-openai-streaming
```

PyPI project:

https://pypi.org/project/safe-openai-streaming/

Install from the project directory during development:

```bash
pip install -e ".[dev]"
```

Then import the public module:

```python
from safe_openai import SafeOpenAI
```

Recommended runtime:

- Python `3.10+`
- `openai>=2.29,<2.44`
- `httpx` with `HTTPTransport.socket_options` support

The implementation was locally checked against:

- Python `3.11`
- `openai==2.29.0`
- `openai==2.43.0`
- `httpx==0.27.0`
- `httpx==0.28.1`

The dependency range intentionally pins the OpenAI SDK minor version because this wrapper intercepts private stream internals.

## Basic Usage

Replace:

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
```

with:

```python
from safe_openai import SafeOpenAI

client = SafeOpenAI(api_key="YOUR_API_KEY")
```

Non-streaming calls continue to pass through normally:

```python
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello."}],
)

print(completion.choices[0].message.content)
```

Streaming calls are protected:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a short story."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

## Failure Policy and Repair Observability

```python
from safe_openai import RepairEvent, SafeOpenAI


def on_repair(event: RepairEvent) -> None:
    print(
        "repaired stream payload",
        event.exception_type,
        event.repaired_bytes,
    )


client = SafeOpenAI(
    api_key="YOUR_API_KEY",
    stream_failure_mode="repair_and_stop",
    on_stream_repair=on_repair,
    max_repair_bytes=64_000,
    max_repair_depth=256,
)
```

Use `stream_failure_mode="raise"` for workflows where silently ending a broken stream is unacceptable.

## Custom HTTP Client

You can still provide your own `httpx.Client`. If you do, `SafeOpenAI` will use it as-is and will not inject the keep-alive transport.

```python
import httpx
from safe_openai import SafeOpenAI, build_keepalive_http_client

http_client = build_keepalive_http_client(
    timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
    http2=False,
)

client = SafeOpenAI(api_key="YOUR_API_KEY", http_client=http_client)
```

## Local Smoke Tests

Compile the package:

```bash
python -m compileall safe_openai
```

Run the full test suite:

```bash
python -m pytest
```

Run the live OpenAI smoke test explicitly:

```bash
set OPENAI_API_KEY=...
python -m pytest -m integration
```

The live smoke test is opt-in because it performs a real streaming request, requires network access, and may incur API usage.

Build the source distribution and wheel:

```bash
python -m build
```

Check `delta: null` sanitization:

```python
from safe_openai import sanitize_sse_payload

payload = '{"choices":[{"delta":null}]}'
print(sanitize_sse_payload(payload))
```

Expected output:

```json
{"choices":[{"delta":{"content": ""}}]}
```

Check JSON repair:

```python
from safe_openai import repair_truncated_json_payload

partial = '{"choices":[{"delta":{"content":"hello'
print(repair_truncated_json_payload(partial))
```

Expected output:

```json
{"choices":[{"delta":{"content":"hello"}}]}
```

## API Reference

### `SafeOpenAI`

Drop-in subclass of `openai.OpenAI`.

```python
SafeOpenAI(
    *args,
    flatten_reasoning: bool = False,
    stream_failure_mode: StreamFailureMode = "repair_and_stop",
    on_stream_repair: RepairCallback | None = None,
    max_repair_bytes: int = 64000,
    max_repair_depth: int = 256,
    http_client: httpx.Client | None = None,
    **kwargs,
)
```

Use this in place of `OpenAI`.

### `SafeChatCompletions`

Replacement for `client.chat.completions`.

- Non-streaming `create(...)` returns the normal `ChatCompletion`.
- Streaming `create(..., stream=True)` returns `SafeStream[ChatCompletionChunk]`.

### `SafeStream`

Wraps the official SDK `Stream` and overrides raw SSE event iteration.

### `SafeSSEEventIterator`

Decodes raw bytes or strings into sanitized `ServerSentEvent` objects.

### `SafeLineIterator`

Lower-level raw line iterator for callers that want sanitized lines directly.

### `repair_truncated_json_payload`

Attempts a single conservative repair of a truncated JSON payload and validates the result with `json.loads`.

The repair function accepts `max_bytes` and `max_depth` limits to prevent unbounded processing of corrupt payloads.

### `RepairEvent`

Dataclass passed to `on_stream_repair` when a repaired final SSE payload is emitted.

### `build_keepalive_http_client`

Builds an `httpx.Client` with a keep-alive-enabled `HTTPTransport`.

### `build_tcp_keepalive_socket_options`

Returns portable socket option triples suitable for `httpx.HTTPTransport(socket_options=...)`.

## Operational Guidance

Use this wrapper when:

- You run streaming completions behind NAT, proxies, VPNs, serverless gateways, or mobile networks and want to reduce the likelihood or impact of idle connection drops.
- You want best-effort recovery of the last partial chunk after a network cut.
- You want streaming callers to see clean termination instead of parser crashes for known recoverable cases.

Avoid treating repaired chunks as cryptographic truth. A repaired chunk is a best-effort salvage of bytes already received. It can preserve user-visible tokens, but it cannot recover bytes that never arrived.

For high-integrity workflows, log when a repaired final event is emitted and consider re-running the request if exact completion is required.

`SafeStream` objects are single-consumer iterators. Do not consume the same stream concurrently from multiple threads, tasks, or callbacks. If multiple consumers need the output, read from `SafeStream` in one place and fan out parsed chunks through your own queue or pub/sub layer.

## Compatibility Notes

This module relies on private internals of `openai-python`, including stream attributes such as `response`, `_cast_to`, `_client`, and `_options`.

That is necessary to intercept SSE data before SDK parsing, but it means SDK upgrades should be tested before production rollout. The package supports `openai>=2.29,<2.44` and is explicitly tested against `2.29.x` and `2.43.x`. If you run an untested minor version inside that range, `SafeOpenAI` will warn at runtime.

## Failure Semantics

When the network fails mid-stream:

- If the current partial event can be repaired and validated, one final chunk is emitted.
- If it cannot be repaired, no synthetic chunk is emitted.
- In both cases, the iterator terminates cleanly.

The wrapper does not retry the request automatically. Automatic retries during streaming can duplicate tokens and produce ambiguous output. If retries are needed, implement them at the application level with idempotency and transcript handling.

## License

MIT. See `LICENSE`.
