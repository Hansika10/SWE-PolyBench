from unittest.mock import Mock

from software_factory.llm import BedrockClient, MockLLM, default_mock_responder, tool_call
from software_factory.models import LLMResponse
from software_factory.stages import SUBMIT_CHANGES, SUBMIT_PLAN

TOOLS = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]


def test_bedrock_request_and_response_parsing():
    client = Mock()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Looking."},
                    {"toolUse": {"toolUseId": "t1", "name": "bash", "input": {"command": "ls"}}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    llm = BedrockClient(model_id="my-model", client=client, max_tokens=100)
    messages = [{"role": "user", "content": [{"text": "hi"}]}]

    response = llm.chat("sys", messages, TOOLS)

    kwargs = client.converse.call_args.kwargs
    assert kwargs["modelId"] == "my-model"
    assert kwargs["system"] == [{"text": "sys"}]
    assert kwargs["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.0}
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"] == {
        "name": "bash",
        "description": "run",
        "inputSchema": {"json": {"type": "object"}},
    }
    assert response.text == "Looking."
    assert response.tool_calls[0].id == "t1"
    assert response.tool_calls[0].input == {"command": "ls"}
    assert (response.input_tokens, response.output_tokens) == (10, 5)
    assert llm.model_name == "bedrock/my-model"


def test_bedrock_model_id_from_env(monkeypatch):
    monkeypatch.setenv("FACTORY_BEDROCK_MODEL_ID", "env-model")
    assert BedrockClient(client=Mock()).model_id == "env-model"


def test_mock_llm_sequence_then_empty():
    llm = MockLLM([LLMResponse(text="a")])
    assert llm.chat("s", [], []).text == "a"
    assert llm.chat("s", [], []).text == ""
    assert len(llm.calls) == 2


def test_default_mock_responder_walks_stages():
    first = default_mock_responder("s", [], TOOLS + [SUBMIT_PLAN])
    assert first.tool_calls[0].name == "list_files"
    history = [{"role": "assistant", "content": [{"toolUse": {"toolUseId": "x"}}]}]
    assert default_mock_responder("s", history, [SUBMIT_PLAN]).tool_calls[0].name == "submit_plan"
    assert default_mock_responder("s", [], [SUBMIT_CHANGES]).tool_calls[0].name == "submit_changes"


def test_tool_call_helper():
    response = tool_call("bash", command="ls")
    assert response.tool_calls[0].name == "bash"
    assert response.tool_calls[0].input == {"command": "ls"}
