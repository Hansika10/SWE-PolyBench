from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Issue:
    """The agent-visible view of a task. Never carries gold patches or grading tests."""

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str = ""
    language: str = ""
    hints_text: str = ""
    dockerfile: str = ""
    issue_url: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CommandResult:
    exit_code: int
    output: str
    timed_out: bool = False


@dataclass
class AgentRun:
    stage: str
    finished: bool
    final_output: Dict[str, Any]
    steps: int
    messages: List[Dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Plan:
    summary: str
    files: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    passed: bool
    summary: str
    commands: List[str] = field(default_factory=list)


@dataclass
class FactoryResult:
    instance_id: str
    status: str
    model_patch: str = ""
    plan: Optional[Plan] = None
    verification: Optional[VerificationReport] = None
    iterations: int = 0
    pr_title: str = ""
    pr_body: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0

    def to_prediction(self, model_name: str) -> Dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_patch": self.model_patch,
            "model_name_or_path": model_name,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
