"""Software Factory: an issue-to-patch agent pipeline scored on SWE-PolyBench.

Stages: intake -> plan -> code -> verify (iterate) -> PR draft / prediction -> score.
"""

from software_factory.llm import BedrockClient, LLMClient, MockLLM
from software_factory.models import FactoryResult, Issue, Plan, VerificationReport
from software_factory.pipeline import FactoryConfig, SoftwareFactory
from software_factory.workspace import DockerWorkspace, LocalWorkspace, Workspace

__all__ = [
    "BedrockClient",
    "DockerWorkspace",
    "FactoryConfig",
    "FactoryResult",
    "Issue",
    "LLMClient",
    "LocalWorkspace",
    "MockLLM",
    "Plan",
    "SoftwareFactory",
    "VerificationReport",
    "Workspace",
]
