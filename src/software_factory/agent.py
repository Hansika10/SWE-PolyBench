from typing import Any, Dict, List

from loguru import logger

from software_factory.llm import LLMClient, ToolSpec
from software_factory.models import AgentRun
from software_factory.tools import ToolExecutor

NUDGE = "Continue working with the tools. When you are done, call `{finish}`."


def run_agent(
    llm: LLMClient,
    *,
    stage: str,
    system: str,
    task: str,
    executor: ToolExecutor,
    finish_tool: ToolSpec,
    max_steps: int,
) -> AgentRun:
    """Tool-use loop: call the model, execute its tool calls, stop when it calls ``finish_tool``."""
    finish_name = finish_tool["name"]
    tools = executor.specs() + [finish_tool]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": [{"text": task}]}]
    input_tokens = output_tokens = 0

    for step in range(1, max_steps + 1):
        response = llm.chat(system, messages, tools)
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens

        content: List[Dict[str, Any]] = []
        if response.text.strip():
            content.append({"text": response.text})
        for call in response.tool_calls:
            content.append({"toolUse": {"toolUseId": call.id, "name": call.name, "input": call.input}})
        messages.append({"role": "assistant", "content": content or [{"text": "(no output)"}]})

        if not response.tool_calls:
            messages.append({"role": "user", "content": [{"text": NUDGE.format(finish=finish_name)}]})
            continue

        results = []
        final_output = None
        for call in response.tool_calls:
            if call.name == finish_name:
                final_output = call.input
                output, ok = "Submitted.", True
            else:
                output, ok = executor.execute(call.name, call.input)
                logger.debug(f"[{stage}] {call.name}({str(call.input)[:120]}) ok={ok}")
            results.append(
                {
                    "toolResult": {
                        "toolUseId": call.id,
                        "content": [{"text": output or "(no output)"}],
                        "status": "success" if ok else "error",
                    }
                }
            )
        messages.append({"role": "user", "content": results})

        if final_output is not None:
            return AgentRun(stage, True, final_output, step, messages, input_tokens, output_tokens)

    logger.warning(f"[{stage}] reached max_steps={max_steps} without calling {finish_name}")
    return AgentRun(stage, False, {}, max_steps, messages, input_tokens, output_tokens)
