import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union

from software_factory.models import LLMResponse, ToolCall

DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

ToolSpec = Dict[str, Any]
Message = Dict[str, Any]


class LLMClient(Protocol):
    """Chat interface used by every stage.

    Messages use the Bedrock Converse content-block format:
    ``{"role": "user"|"assistant", "content": [{"text": ...} | {"toolUse": ...} | {"toolResult": ...}]}``.
    Tool specs are ``{"name", "description", "input_schema"}`` dicts.
    """

    model_name: str

    def chat(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        ...


class BedrockClient:
    """AWS Bedrock client using the Converse API (tool use supported)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        client: Any = None,
        max_retries: int = 8,
    ):
        self.model_id = model_id or os.environ.get("FACTORY_BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
                config=Config(
                    read_timeout=600, retries={"max_attempts": max_retries, "mode": "adaptive"}
                ),
            )
        self.client = client

    @property
    def model_name(self) -> str:
        return f"bedrock/{self.model_id}"

    def chat(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "system": [{"text": system}],
            "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature},
        }
        if tools:
            kwargs["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t["name"],
                            "description": t["description"],
                            "inputSchema": {"json": t["input_schema"]},
                        }
                    }
                    for t in tools
                ]
            }
        response = self.client.converse(**kwargs)
        content = response["output"]["message"]["content"]
        text = "".join(block["text"] for block in content if "text" in block)
        tool_calls = [
            ToolCall(
                id=block["toolUse"]["toolUseId"],
                name=block["toolUse"]["name"],
                input=block["toolUse"].get("input") or {},
            )
            for block in content
            if "toolUse" in block
        ]
        usage = response.get("usage", {})
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.get("stopReason", "end_turn"),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )


Responder = Callable[[str, List[Message], List[ToolSpec]], LLMResponse]


class MockLLM:
    """Deterministic LLM for tests and dry runs.

    ``responder`` is either a sequence of responses returned in order, or a callable
    ``(system, messages, tools) -> LLMResponse``. When a sequence is exhausted, an empty
    text response is returned.
    """

    def __init__(
        self, responder: Union[Sequence[LLMResponse], Responder, None] = None, model_name="mock"
    ):
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []
        if responder is None:
            responder = default_mock_responder
        self._responder = responder if callable(responder) else None
        self._queue = list(responder) if not callable(responder) else []

    def chat(self, system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
        self.calls.append(
            {"system": system, "messages": json.loads(json.dumps(messages)), "tools": tools}
        )
        if self._responder is not None:
            return self._responder(system, messages, tools)
        if self._queue:
            return self._queue.pop(0)
        return LLMResponse(text="")


def tool_call(name: str, **tool_input: Any) -> LLMResponse:
    """Build a response containing a single tool call (handy for scripted mocks)."""
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, input=tool_input)],
        stop_reason="tool_use",
    )


def default_mock_responder(system: str, messages: List[Message], tools: List[ToolSpec]) -> LLMResponse:
    """Walks every stage without editing code: useful to smoke-test the plumbing."""
    names = {t["name"] for t in tools}
    used_tools = any(
        "toolUse" in block for m in messages if m["role"] == "assistant" for block in m["content"]
    )
    if "submit_plan" in names:
        if not used_tools:
            return tool_call("list_files", path=".")
        return tool_call("submit_plan", summary="Mock plan: no analysis performed.", files=[], steps=[])
    if "submit_changes" in names:
        return tool_call("submit_changes", summary="Mock LLM made no changes.")
    if "submit_verification" in names:
        return tool_call(
            "submit_verification", passed=False, summary="Mock LLM does not verify.", commands=[]
        )
    return LLMResponse(text="")
