"""Sequential evaluation runner.

Ties together Task + Simulator + Agent into an episode loop:
1. Load task -> get episodes, simulator key, action space
2. Start simulator subprocess
3. For each episode:
   a. Reset simulator with format_reset_config(episode)
   b. Loop: agent.act(observation) -> simulator.step(action) until done or max_steps
   c. Evaluate: task.evaluate_episode(episode, trajectory)
   d. Save per-episode metrics + trajectory.jsonl + images
4. Aggregate metrics into summary.json

Output directory structure:
    <output_dir>/<task_name>/<run_id>/
        config.json
        summary.json
        episodes/
            000_<episode_id>/
                result.json
                trajectory.jsonl
                step_0000.png, step_0001.png, ...
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from easi.core.episode import StepResult
from easi.evaluation.metrics import aggregate_metrics
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _sanitize_dirname(name: str) -> str:
    """Replace characters unsafe for directory names."""
    return re.sub(r'[^\w\-.]', '_', name)


class EvaluationRunner:
    """Sequential evaluation runner."""

    def __init__(
        self,
        task_name: str,
        agent_type: str = "dummy",
        output_dir: Path | str = "./logs",
        data_dir: Path | str = "./datasets",
        llm_base_url: str | None = None,
        agent_seed: int | None = None,
        backend: str | None = None,
        model: str = "default",
        port: int = 8080,
        llm_kwargs_raw: str | None = None,
    ):
        self.task_name = task_name
        self.agent_type = agent_type
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.llm_base_url = llm_base_url
        self.agent_seed = agent_seed
        self.backend = backend
        self.model = model
        self.port = port
        self.llm_kwargs_raw = llm_kwargs_raw
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _resolve_llm_backend(self) -> tuple[str | None, str | None]:
        """Resolve which LLM backend to use.

        Returns (backend, base_url):
            - (None, None) for dummy agent
            - ("legacy", url) for --llm-url without --backend
            - (backend_name, url_or_none) for --backend
        """
        if self.agent_type == "dummy":
            return None, None

        if self.backend:
            return self.backend, self.llm_base_url

        if self.llm_base_url:
            return "legacy", self.llm_base_url

        raise ValueError(
            f"Agent '{self.agent_type}' requires --backend or --llm-url. "
            f"Use --backend vllm|openai|anthropic|gemini or --llm-url <url>."
        )

    def run(self, max_episodes: int | None = None) -> list[dict]:
        """Run evaluation and return per-episode metric dicts."""
        run_dir = self.output_dir / self.task_name / self.run_id
        episodes_dir = run_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load task
        task = self._create_task()
        episodes = task.load_episodes()
        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        # 2. Resolve LLM backend and optionally start server
        backend, base_url = self._resolve_llm_backend()
        server = None
        if backend == "vllm" and base_url is None:
            from easi.llm.server_manager import ServerManager
            from easi.llm.utils import parse_llm_kwargs, split_kwargs

            all_kwargs = parse_llm_kwargs(self.llm_kwargs_raw)
            server_kwargs, _ = split_kwargs(all_kwargs)
            server = ServerManager(
                "vllm", self.model, port=self.port,
                server_kwargs=server_kwargs, log_dir=run_dir,
            )
            base_url = server.start()

        # Save run config
        config = {
            "task_name": self.task_name,
            "agent_type": self.agent_type,
            "agent_seed": self.agent_seed,
            "run_id": self.run_id,
            "max_episodes": max_episodes,
            "total_episodes": len(episodes),
            "backend": backend,
            "model": self.model,
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        # 3. Create agent
        agent = self._create_agent(task.action_space, task._config,
                                   backend=backend, base_url=base_url)

        # 4. Start simulator
        sim, sim_runner = self._create_simulator(task.simulator_key, task=task)

        all_results = []
        try:
            for i, episode in enumerate(episodes):
                episode_id = episode.get("episode_id", f"ep_{i}")
                logger.info(
                    "Episode %d/%d: %s", i + 1, len(episodes), episode_id,
                )

                episode_dir = episodes_dir / f"{i:03d}_{_sanitize_dirname(episode_id)}"
                episode_dir.mkdir(exist_ok=True)

                result = self._run_episode(
                    sim, agent, task, episode, i, episode_dir,
                )
                all_results.append(result)

                (episode_dir / "result.json").write_text(
                    json.dumps(result, indent=2)
                )

        finally:
            sim.close()
            if server:
                server.stop()

        # 5. Aggregate and save summary
        summary = aggregate_metrics(all_results)
        if backend and backend != "legacy":
            summary["llm_usage"] = self._aggregate_llm_usage(all_results)
            summary["model"] = self.model
            summary["backend"] = backend
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Results saved to: %s", run_dir)
        logger.info("Summary: %s", summary)

        return all_results

    def _run_episode(
        self, sim, agent, task, episode: dict, index: int, episode_dir: Path,
    ) -> dict:
        """Run a single episode and return metrics."""
        agent.reset()

        episode_id = episode.get("episode_id", f"ep_{index}")

        # Reset simulator (bridge saves images to episode_dir)
        reset_config = task.format_reset_config(episode)
        observation = sim.reset(
            episode_id,
            reset_config,
            episode_output_dir=str(episode_dir),
        )

        # Propagate dynamic action space from bridge (e.g., EB-Alfred per-episode actions)
        dynamic_actions_json = observation.metadata.get("dynamic_action_space")
        if dynamic_actions_json and hasattr(agent, 'update_action_space'):
            dynamic_actions = json.loads(dynamic_actions_json)
            agent.update_action_space(dynamic_actions)

        # Write reset entry to trajectory
        trajectory_path = episode_dir / "trajectory.jsonl"
        self._write_trajectory_entry(trajectory_path, {
            "step": 0,
            "type": "reset",
            "rgb_path": Path(observation.rgb_path).name,
            "agent_pose": observation.agent_pose,
            "reward": 0.0,
            "done": False,
            "info": {},
        })

        # Agent-simulator loop
        trajectory: list[StepResult] = []
        task_description = task.get_instruction(episode)
        start_time = time.monotonic()

        for step in range(task.max_steps):
            action = agent.act(observation, task_description)
            step_result = sim.step(action)
            trajectory.append(step_result)

            # Write step entry to trajectory
            self._write_trajectory_entry(trajectory_path, {
                "step": step + 1,
                "type": "step",
                "action": action.action_name,
                "rgb_path": Path(step_result.observation.rgb_path).name,
                "agent_pose": step_result.observation.agent_pose,
                "reward": step_result.reward,
                "done": step_result.done,
                "info": step_result.info,
            })

            # Feed action outcome back to agent for ReAct reasoning
            last_success = step_result.info.get("last_action_success", 1.0)
            feedback = step_result.info.get(
                "feedback",
                "success" if last_success else "failed",
            )
            agent.add_feedback(action.action_name, feedback)

            observation = step_result.observation

            if step_result.done:
                break

        elapsed = time.monotonic() - start_time

        # Evaluate
        metrics = task.evaluate_episode(episode, trajectory)
        metrics["episode_id"] = episode_id
        metrics["elapsed_seconds"] = round(elapsed, 2)

        # Snapshot LLM usage for this episode
        if hasattr(agent, 'llm_client') and hasattr(agent.llm_client, 'get_usage'):
            metrics["llm_usage"] = agent.llm_client.get_usage()
            agent.llm_client.reset_usage()

        return metrics

    @staticmethod
    def _write_trajectory_entry(path: Path, entry: dict) -> None:
        """Append a single JSON line to the trajectory file."""
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _aggregate_llm_usage(results: list[dict]) -> dict:
        """Sum up llm_usage from per-episode results."""
        total = {
            "total_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
        for r in results:
            usage = r.get("llm_usage", {})
            total["total_calls"] += usage.get("num_calls", 0)
            total["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
            total["total_completion_tokens"] += usage.get("completion_tokens", 0)
            total["total_cost_usd"] += usage.get("cost_usd", 0.0)
        total["total_tokens"] = total["total_prompt_tokens"] + total["total_completion_tokens"]
        n = len(results) or 1
        total["avg_prompt_tokens_per_episode"] = round(total["total_prompt_tokens"] / n)
        total["avg_cost_per_episode_usd"] = round(total["total_cost_usd"] / n, 6)
        return total

    def _create_task(self):
        from easi.tasks.registry import get_task_entry, load_task_class

        entry = get_task_entry(self.task_name)
        TaskClass = load_task_class(self.task_name)
        return TaskClass(
            split_yaml_path=entry.config_path,
            data_dir=self.data_dir,
        )

    def _create_agent(self, action_space: list[str], task_config: dict,
                      backend: str | None = None, base_url: str | None = None):
        from easi.utils.import_utils import import_class

        if self.agent_type == "dummy":
            from easi.agents.dummy_agent import DummyAgent
            return DummyAgent(action_space=action_space, seed=self.agent_seed)

        elif self.agent_type == "react":
            from easi.agents.react_agent import ReActAgent

            agent_config = task_config.get("agent", {})

            # Create LLM client based on backend
            if backend and backend != "legacy":
                from easi.llm.client import LLMClient
                from easi.llm.utils import (
                    build_litellm_model, parse_llm_kwargs,
                    split_kwargs, validate_backend,
                )

                validate_backend(backend)
                litellm_model = build_litellm_model(backend, self.model)
                all_kwargs = parse_llm_kwargs(self.llm_kwargs_raw)
                _, client_kwargs = split_kwargs(all_kwargs)

                llm = LLMClient(
                    model=litellm_model,
                    base_url=base_url,
                    **client_kwargs,
                )
            else:
                # Legacy path: existing LLMApiClient
                from easi.llm.api_client import LLMApiClient
                llm = LLMApiClient(
                    base_url=base_url or "http://127.0.0.1:8000"
                )

            # Load task-specific prompt builder
            prompt_builder = None
            builder_class_name = agent_config.get("prompt_builder")
            if builder_class_name:
                BuilderClass = import_class(builder_class_name)
                builder_kwargs = agent_config.get("prompt_builder_kwargs", {})
                prompt_builder = BuilderClass(**builder_kwargs)

            return ReActAgent(
                llm_client=llm,
                action_space=action_space,
                prompt_builder=prompt_builder,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def _create_simulator(self, simulator_key: str, task=None):
        import json as _json

        from easi.simulators.registry import (
            load_env_manager_class,
            load_simulator_class,
        )
        from easi.simulators.subprocess_runner import SubprocessRunner

        EnvManagerClass = load_env_manager_class(simulator_key)
        SimClass = load_simulator_class(simulator_key)

        env_manager = EnvManagerClass()
        sim = SimClass()

        # Auto-install simulator env if not ready
        if not env_manager.env_is_ready():
            logger.info("Simulator environment not ready, auto-installing...")
            env_manager.install()

        # Install task-level additional deps
        if task and task.additional_deps:
            env_manager.install_additional_deps(task.additional_deps)

        # Task-specific bridge overrides simulator default
        bridge_path = (
            (task.get_bridge_script_path() if task else None)
            or sim._get_bridge_script_path()
        )

        extra_args = ["--data-dir", str(self.data_dir)]
        if task and task.simulator_kwargs:
            extra_args.extend(["--simulator-kwargs", _json.dumps(task.simulator_kwargs)])

        runner = SubprocessRunner(
            python_executable=env_manager.get_python_executable(),
            bridge_script_path=bridge_path,
            needs_display=env_manager.needs_display,
            xvfb_screen_config=env_manager.xvfb_screen_config,
            extra_args=extra_args,
        )
        runner.launch()
        sim.set_runner(runner)

        return sim, runner
