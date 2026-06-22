from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Optional, TypeVar, Union, overload

import httpx
from openai import OpenAI, Stream
from openai._streaming import ServerSentEvent
from openai.resources.chat.completions import Completions as ChatCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk

T = TypeVar("T")
StreamFailureMode = Literal["repair_and_stop", "stop", "raise", "repair_or_raise"]
RepairCallback = Callable[["RepairEvent"], None]

__version__: str

@dataclass(frozen=True)
class RepairEvent:
    original_payload: str
    repaired_payload: str
    exception_type: str
    exception_message: str
    source: str
    repaired_bytes: int

def build_tcp_keepalive_socket_options() -> list[tuple[int, int, int]]: ...

def build_keepalive_http_client(
    *,
    timeout: httpx.Timeout | float | None = None,
    limits: httpx.Limits | None = None,
    http2: bool = False,
    verify: httpx.VerifyTypes = True,
    trust_env: bool = True,
) -> httpx.Client: ...

def sanitize_sse_payload(payload: str) -> str: ...

def repair_truncated_json_payload(
    payload: str,
    *,
    max_bytes: int = 64000,
    max_depth: int = 256,
) -> str | None: ...

class SafeLineIterator(Iterator[str]):
    last_valid_line: str | None
    last_partial_line: str | None
    def __init__(
        self,
        source: Iterable[bytes | str],
        *,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = 64000,
        max_repair_depth: int = 256,
    ) -> None: ...
    def __iter__(self) -> SafeLineIterator: ...
    def __next__(self) -> str: ...

class SafeSSEEventIterator(Iterator[ServerSentEvent]):
    last_valid_line: str | None
    last_partial_line: str | None
    def __init__(
        self,
        source: Iterable[bytes | str],
        *,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = 64000,
        max_repair_depth: int = 256,
    ) -> None: ...
    def __iter__(self) -> SafeSSEEventIterator: ...
    def __next__(self) -> ServerSentEvent: ...

class SafeStream(Stream[T], Generic[T]):
    def __init__(
        self,
        source: Stream[T],
        *,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = 64000,
        max_repair_depth: int = 256,
    ) -> None: ...

class SafeChatCompletions(ChatCompletions):
    def __init__(
        self,
        client: OpenAI,
        *,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = 64000,
        max_repair_depth: int = 256,
    ) -> None: ...
    @overload
    def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> SafeStream[ChatCompletionChunk]: ...
    @overload
    def create(self, *args: Any, stream: Optional[Literal[False]] = False, **kwargs: Any) -> ChatCompletion: ...

class SafeOpenAI(OpenAI):
    def __init__(
        self,
        *args: Any,
        flatten_reasoning: bool = False,
        stream_failure_mode: StreamFailureMode = "repair_and_stop",
        on_stream_repair: RepairCallback | None = None,
        max_repair_bytes: int = 64000,
        max_repair_depth: int = 256,
        http_client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None: ...
