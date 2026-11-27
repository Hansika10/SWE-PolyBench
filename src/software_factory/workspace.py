import io
import posixpath
import re
import shlex
import subprocess
import tarfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from software_factory.models import CommandResult, Issue

SCRATCH_DIR = ".factory_scratch"

_GIT = "git -c core.fileMode=false -c core.autocrlf=false"

TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|__mocks__|specs?|testing|test-?utils?|fixtures)/"
    r"|(^|/)src/test/"
    r"|(^|/)test_[^/]*\.py$"
    r"|_tests?\.py$"
    r"|(^|/)conftest\.py$"
    r"|\.(test|spec)\.[cm]?[jt]sx?$"
    r"|Tests?\.java$"
)


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path))


def _split_patch(patch: str) -> List[str]:
    chunks = re.split(r"(?m)^(?=diff --git )", patch)
    return [c for c in chunks if c.startswith("diff --git ")]


def _patch_path(chunk: str) -> str:
    match = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
    return match.group(2) if match else ""


def filter_patch(patch: str, keep_test_changes: bool = False) -> str:
    """Drop scratch files (always) and test files (unless ``keep_test_changes``) from a diff.

    The benchmark applies its own test patch, so agent-written tests would only cause conflicts.
    """
    kept = []
    for chunk in _split_patch(patch):
        path = _patch_path(chunk)
        if path.startswith(f"{SCRATCH_DIR}/"):
            continue
        if not keep_test_changes and is_test_path(path):
            continue
        kept.append(chunk)
    return "".join(kept)


def patch_files(patch: str) -> List[str]:
    return [_patch_path(c) for c in _split_patch(patch)]


class Workspace(ABC):
    """A checked-out repository the agents can read, edit and run commands in."""

    root: str = ""
    baseline_tree: Optional[str] = None

    def start(self) -> None:
        """Prepare the workspace (e.g. pull image and start a container)."""

    def close(self) -> None:
        """Release resources."""

    @abstractmethod
    def run(self, command: str, timeout: int = 300) -> CommandResult:
        ...

    @abstractmethod
    def read_file(self, path: str) -> str:
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        ...

    def resolve(self, path: str) -> str:
        return path if posixpath.isabs(path) else posixpath.normpath(posixpath.join(self.root, path))

    def _snapshot_tree(self) -> str:
        result = self.run(
            f"{_GIT} add -A >/dev/null 2>&1 && {_GIT} write-tree && {_GIT} reset -q >/dev/null 2>&1",
            timeout=600,
        )
        tree = result.output.strip().splitlines()[-1] if result.output.strip() else ""
        if result.exit_code != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", tree):
            raise RuntimeError(f"Failed to snapshot git tree: {result.output[-500:]}")
        return tree

    def setup_baseline(self) -> None:
        """Record the starting tree (without committing) so later diffs contain only agent edits."""
        self.run(f"mkdir -p {SCRATCH_DIR}")
        self.baseline_tree = self._snapshot_tree()

    def diff(self) -> str:
        if self.baseline_tree is None:
            raise RuntimeError("setup_baseline() must be called before diff()")
        current = self._snapshot_tree()
        result = self.run(f"{_GIT} diff --no-color --no-ext-diff {self.baseline_tree} {current}", timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {result.output[-500:]}")
        patch = result.output
        if patch and not patch.endswith("\n"):
            patch += "\n"
        return patch


class LocalWorkspace(Workspace):
    """A git checkout on the host filesystem. Used for live GitHub issues and tests."""

    def __init__(self, root: str):
        self.root = str(Path(root).expanduser().resolve())
        self.baseline_tree = None

    def run(self, command: str, timeout: int = 300) -> CommandResult:
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=self.root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            return CommandResult(proc.returncode, proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            return CommandResult(124, f"{partial}\n[command timed out after {timeout}s]", True)

    def read_file(self, path: str) -> str:
        return Path(self.resolve(path)).read_text(errors="replace")

    def write_file(self, path: str, content: str) -> None:
        target = Path(self.resolve(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


class DockerWorkspace(Workspace):
    """The SWE-PolyBench instance container: same image the evaluator uses, at the base commit."""

    def __init__(
        self,
        issue: Issue,
        client=None,
        delete_image: bool = False,
        repo_path: str = "~/polybench_repos",
        repair_native_packages: Optional[List[str]] = None,
    ):
        self.issue = issue
        self._client = client
        self.delete_image = delete_image
        self.repo_path = repo_path
        self.repair_native_packages = repair_native_packages
        self.image_id = f"polybench_{issue.language.lower()}_{issue.instance_id.lower()}"
        self.container = None
        self.baseline_tree = None
        self._ghcr_image: Optional[str] = None

    @property
    def client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env(timeout=720)
        return self._client

    def _ensure_image(self) -> None:
        from poly_bench_evaluation.docker_utils import DockerManager
        from poly_bench_evaluation.repo_utils import RepoManager

        manager = DockerManager(image_id=self.image_id, delete_image=False, client=self.client)
        if manager.check_image_local(local_image_name=self.image_id):
            return
        if manager.try_pull_prebuilt_image(self.issue.instance_id):
            self._ghcr_image = manager.ghcr_image_name
            return
        logger.info(f"Building image {self.image_id} locally")
        if self.issue.language != "Python":
            DockerManager(
                image_id=f"polybench_{self.issue.language.lower()}_base",
                delete_image=False,
                client=self.client,
            ).build_base_image(language=self.issue.language)
        repo_manager = RepoManager(repo_name=self.issue.repo, repo_path=self.repo_path)
        repo_manager.clone_repo()
        repo_manager.checkout_commit(commit_hash=self.issue.base_commit)
        assert repo_manager.tmp_repo_dir is not None, "Repo not properly cloned."
        if manager.docker_build(repo_manager.tmp_repo_dir, self.issue.dockerfile) != 0:
            raise RuntimeError(
                f"Docker build failed for {self.issue.instance_id}: "
                + "\n".join(manager.build_logs[-20:])
            )

    def start(self) -> None:
        self._ensure_image()
        image = self.client.images.get(self.image_id)
        self.root = str(image.attrs["Config"]["WorkingDir"] or "/testbed")
        self.container = self.client.containers.run(
            image=self.image_id,
            command="tail -f /dev/null",
            detach=True,
            tty=True,
            working_dir=self.root,
            name=f"factory_{self.image_id}_{uuid.uuid4().hex[:8]}",
        )
        if self.repair_native_packages:
            from poly_bench_evaluation.docker_utils import DockerManager

            manager = DockerManager(image_id=self.image_id, delete_image=False, client=self.client)
            manager.container = self.container
            try:
                manager.repair_native_imports(packages=self.repair_native_packages)
            finally:
                manager.container = None

    def run(self, command: str, timeout: int = 300) -> CommandResult:
        assert self.container is not None, "Workspace not started"
        wrapped = f"timeout -s KILL {int(timeout)} bash -c {shlex.quote(command)}"
        result = self.container.exec_run(["bash", "-c", wrapped], workdir=self.root, user="root")
        output = (result.output or b"").decode("utf-8", errors="replace")
        timed_out = result.exit_code == 137
        if timed_out:
            output += f"\n[command killed after {timeout}s]"
        return CommandResult(result.exit_code, output, timed_out)

    def read_file(self, path: str) -> str:
        result = self.run(f"cat {shlex.quote(self.resolve(path))}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.output.strip())
        return result.output

    def write_file(self, path: str, content: str) -> None:
        assert self.container is not None, "Workspace not started"
        target = self.resolve(path)
        directory, name = posixpath.split(target)
        self.run(f"mkdir -p {shlex.quote(directory)}")
        data = content.encode()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        if not self.container.put_archive(directory, stream.getvalue()):
            raise RuntimeError(f"Failed to write {target}")

    def close(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception as e:
                logger.debug(f"Failed to remove container: {e}")
            self.container = None
        if self.delete_image:
            for tag in filter(None, [self.image_id, self._ghcr_image]):
                try:
                    self.client.images.remove(tag, force=True)
                except Exception as e:
                    logger.debug(f"Failed to remove image {tag}: {e}")


WorkspaceFactory = Callable[[Issue], Workspace]
