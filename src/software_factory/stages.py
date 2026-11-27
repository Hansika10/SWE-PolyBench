from typing import List, Optional

from software_factory.agent import run_agent
from software_factory.llm import LLMClient
from software_factory.models import AgentRun, Issue, Plan, VerificationReport
from software_factory.tools import ALL_TOOLS, READ_ONLY_TOOLS, ToolExecutor
from software_factory.workspace import SCRATCH_DIR, Workspace

_COMMON = """You are one stage of an autonomous software factory that resolves GitHub issues.
The repository is checked out at {root} (language: {language}). Commands run from that directory.
Work only from the issue text and the repository itself. Be efficient with tool calls."""

PLAN_SYSTEM = (
    _COMMON
    + """

You are the PLANNING stage. Explore the code to find the root cause of the issue and decide the
smallest correct change. Do not modify any files (bash is for read-only commands only).
When done, call `submit_plan` with a short summary, the files to change, and concrete steps."""
)

CODE_SYSTEM = (
    _COMMON
    + """

You are the CODING stage. Implement the plan with a minimal, focused change to the source code.
Follow the existing code style. Do not modify existing tests unless the issue requires it.
Put throwaway scripts in {scratch}/ (it is excluded from the final patch).
When done, call `submit_changes` with a summary of what you changed."""
)

VERIFY_SYSTEM = (
    _COMMON
    + """

You are the VERIFICATION stage. Check that the current changes resolve the issue without regressions:
1. Write a minimal test or script that reproduces the behavior described in the issue
   (in {scratch}/ or in the repo's test suite) and run it.
2. Find out how this repo runs its tests and run the existing tests closest to the changed code.
Do not edit non-test source files: if the fix is wrong or incomplete, report it.
Call `submit_verification` with passed=true only if the issue is fixed and related tests pass.
Otherwise set passed=false and explain precisely what is wrong so the coding stage can fix it."""
)

SUBMIT_PLAN = {
    "name": "submit_plan",
    "description": "Submit the implementation plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Root cause and intended fix."},
            "files": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "files", "steps"],
    },
}

SUBMIT_CHANGES = {
    "name": "submit_changes",
    "description": "Signal that the code change is complete.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}

SUBMIT_VERIFICATION = {
    "name": "submit_verification",
    "description": "Report verification results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "summary": {"type": "string"},
            "commands": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["passed", "summary"],
    },
}


def _issue_block(issue: Issue) -> str:
    text = f"<issue repo=\"{issue.repo}\">\n{issue.problem_statement.strip()}\n</issue>"
    if issue.hints_text and issue.hints_text.strip():
        text += f"\n<hints>\n{issue.hints_text.strip()}\n</hints>"
    return text


def _plan_block(plan: Plan) -> str:
    steps = "\n".join(f"- {s}" for s in plan.steps) or "- (none)"
    files = ", ".join(plan.files) or "(not specified)"
    return f"<plan>\nSummary: {plan.summary}\nFiles: {files}\nSteps:\n{steps}\n</plan>"


def _system(template: str, workspace: Workspace, issue: Issue) -> str:
    return template.format(root=workspace.root, language=issue.language or "unknown", scratch=SCRATCH_DIR)


def _as_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value else []


def plan_stage(
    llm: LLMClient, issue: Issue, workspace: Workspace, max_steps: int, max_output_chars: int
) -> "tuple[Plan, AgentRun]":
    run = run_agent(
        llm,
        stage="plan",
        system=_system(PLAN_SYSTEM, workspace, issue),
        task=f"{_issue_block(issue)}\n\nInvestigate the repository and submit a plan.",
        executor=ToolExecutor(workspace, READ_ONLY_TOOLS, max_output_chars),
        finish_tool=SUBMIT_PLAN,
        max_steps=max_steps,
    )
    out = run.final_output
    plan = Plan(
        summary=str(out.get("summary") or "No plan submitted; investigate and fix the issue directly."),
        files=_as_list(out.get("files")),
        steps=_as_list(out.get("steps")),
    )
    return plan, run


def code_stage(
    llm: LLMClient,
    issue: Issue,
    workspace: Workspace,
    plan: Plan,
    feedback: Optional[str],
    max_steps: int,
    max_output_chars: int,
) -> AgentRun:
    task = f"{_issue_block(issue)}\n\n{_plan_block(plan)}\n\n"
    if feedback:
        current = workspace.diff()
        task += (
            f"A previous attempt did not pass verification.\n<feedback>\n{feedback}\n</feedback>\n"
            f"<current_diff>\n{current or '(no changes)'}\n</current_diff>\n\n"
            "Revise the changes to address the feedback."
        )
    else:
        task += "Implement the plan."
    return run_agent(
        llm,
        stage="code",
        system=_system(CODE_SYSTEM, workspace, issue),
        task=task,
        executor=ToolExecutor(workspace, ALL_TOOLS, max_output_chars),
        finish_tool=SUBMIT_CHANGES,
        max_steps=max_steps,
    )


def verify_stage(
    llm: LLMClient,
    issue: Issue,
    workspace: Workspace,
    plan: Plan,
    patch: str,
    max_steps: int,
    max_output_chars: int,
) -> "tuple[VerificationReport, AgentRun]":
    task = (
        f"{_issue_block(issue)}\n\n{_plan_block(plan)}\n\n"
        f"<current_diff>\n{patch}\n</current_diff>\n\nVerify these changes."
    )
    run = run_agent(
        llm,
        stage="verify",
        system=_system(VERIFY_SYSTEM, workspace, issue),
        task=task,
        executor=ToolExecutor(workspace, ALL_TOOLS, max_output_chars),
        finish_tool=SUBMIT_VERIFICATION,
        max_steps=max_steps,
    )
    out = run.final_output
    report = VerificationReport(
        passed=bool(out.get("passed")) if run.finished else False,
        summary=str(out.get("summary") or "Verification did not complete."),
        commands=_as_list(out.get("commands")),
    )
    return report, run
