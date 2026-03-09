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

import inspect
import json
import re
import time
from datetime import datetime
from pathlib import Path

from easi.core.episode import EpisodeRecord, StepResult
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _sanitize_dirname(name: str) -> str:
    """Replace characters unsafe for directory names."""
    return re.sub(r'[^\w\-.]', '_', name)


class EvaluationRunner:
    """Sequential evaluation runner."""

    # Session-specific params excluded from config.json
    _EXCLUDE_FROM_CONFIG = frozenset({"resume_dir", "refresh_data"})

    def __init__(
        self,
        task_name: str,
        agent_type: str = "react",
        output_dir: Path | str = "./logs",
        data_dir: Path | str = "./datasets",
        max_episodes: int | None = None,
        llm_base_url: str | None = None,
        agent_seed: int | None = None,
        backend: str | None = None,
        model: str = "default",
        port: int = 8080,
        llm_kwargs_raw: str | None = None,
        max_retries: int = 3,
        resume_dir: Path | str | None = None,
        refresh_data: bool = False,
        render_platform: str | None = None,
        llm_instances: int | None = None,
        llm_gpus: list[int] | None = None,
        sim_gpus: list[int] | None = None,
    ):
        # Auto-capture all init args for config.json (before any mutation)
        frame = inspect.currentframe()
        self._cli_options = {
            k: v for k, v in inspect.getargvalues(frame).locals.items()
            if k not in ("self", "frame") and k not in self._EXCLUDE_FROM_CONFIG
        }

        self.task_name = task_name
        self.agent_type = agent_type
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.max_episodes = max_episodes
        self.llm_base_url = llm_base_url
        self.agent_seed = agent_seed
        self.backend = backend
        self.model = model
        self.port = port
        self.llm_kwargs_raw = llm_kwargs_raw
        self.max_retries = max_retries
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.refresh_data = refresh_data
        self.render_platform_name = render_platform
        self.llm_instances = llm_instances
        self.llm_gpus = llm_gpus
        self.sim_gpus = sim_gpus
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.model:
            safe_model = self.model.replace("/", "_")
            self.run_id = f"{timestamp}_{safe_model}"
        else:
            self.run_id = timestamp

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

    def _serialize_cli_options(self) -> dict:
        """Serialize _cli_options for JSON output (convert Paths to strings)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self._cli_options.items()
        }

    def run(self) -> list[dict]:
        """Run evaluation and return per-episode metric dicts."""
        if self.resume_dir:
            run_dir = self.resume_dir
        else:
            run_dir = self.output_dir / self.task_name / self.run_id

        episodes_dir = run_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load task (before resume so we know total_episodes)
        task = self._create_task()
        if self.refresh_data:
            task.download_dataset(force=True)
        episodes = task.load_episodes()
        if self.max_episodes is not None:
            episodes = episodes[:self.max_episodes]

        # Handle resume: load completed results and find start point
        if self.resume_dir:
            all_results, start_index = self._load_completed_results(run_dir, len(episodes))
            logger.info(
                "Resuming from %s — %d completed episodes, starting from index %d",
                run_dir, len(all_results), start_index,
            )
        else:
            all_results = []
            start_index = 0

        # 2. Resolve LLM backend and optionally start server
        backend, base_url = self._resolve_llm_backend()
        server = None

        try:
            if backend in ("vllm", "custom") and base_url is None:
                from easi.llm.server_manager import ServerManager
                from easi.llm.utils import parse_llm_kwargs, split_kwargs

                all_kwargs = parse_llm_kwargs(self.llm_kwargs_raw)
                server_kwargs, _ = split_kwargs(all_kwargs)
                server = ServerManager(
                    backend, self.model, port=self.port,
                    server_kwargs=server_kwargs,
                )
                base_url = server.start()

            # Compute resolved generation kwargs (YAML defaults + CLI overrides)
            from easi.llm.utils import parse_llm_kwargs, split_kwargs
            agent_config = task._config.get("agent", {})
            yaml_gen_kwargs = agent_config.get("generation_kwargs", {})
            all_llm_kwargs = parse_llm_kwargs(self.llm_kwargs_raw)
            _, cli_gen_kwargs = split_kwargs(all_llm_kwargs)
            resolved_gen_kwargs = {**yaml_gen_kwargs, **cli_gen_kwargs}

            # Save run config
            config = {
                "run_id": self.run_id,
                "total_episodes": len(episodes),
                "cli_options": self._serialize_cli_options(),
                "resolved_backend": backend,
                "resolved_base_url": base_url,
                "resolved_generation_kwargs": resolved_gen_kwargs,
                "task_config": task._config,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
            logger.trace(
                "Run config:\n%s", json.dumps(config, indent=2, default=str)
            )

            # Skip simulator/agent if all episodes already complete (resume)
            if start_index >= len(episodes):
                logger.info("All %d episodes already complete, re-aggregating summary.", len(episodes))
            else:
                # 3. Create agent
                agent = self._create_agent(task.action_space, task._config,
                                           backend=backend, base_url=base_url)

                # 4. Start simulator
                sim, sim_runner = self._create_simulator(task.simulator_key, task=task)

                # 5. Progress bar
                from easi.utils.progress import ProgressBar

                progress_bar = ProgressBar(
                    total=len(episodes), num_workers=1, start_index=start_index,
                )
                progress_bar.start()

                try:
                    for i, episode in enumerate(episodes):
                        if i < start_index:
                            continue
                        episode_id = episode.get("episode_id", f"ep_{i}")
                        logger.info(
                            "Episode %d/%d: %s", i + 1, len(episodes), episode_id,
                        )

                        episode_dir = episodes_dir / f"{i:03d}_{_sanitize_dirname(episode_id)}"
                        episode_dir.mkdir(exist_ok=True)

                        result = None
                        for attempt in range(1, self.max_retries + 1):
                            try:
                                result = self._run_episode(
                                    sim, agent, task, episode, i, episode_dir,
                                )
                                break
                            except Exception as exc:
                                logger.warning(
                                    "Episode %s attempt %d/%d failed: %s",
                                    episode_id, attempt, self.max_retries, exc,
                                )
                                self._clear_episode_dir(episode_dir)
                                if attempt < self.max_retries:
                                    logger.info("Re-launching simulator for retry...")
                                    try:
                                        sim.close()
                                    except Exception:
                                        pass
                                    sim, sim_runner = self._create_simulator(
                                        task.simulator_key, task=task,
                                    )
                                else:
                                    logger.error(
                                        "Episode %s failed after %d attempts, skipping",
                                        episode_id, self.max_retries,
                                    )
                                    result = {
                                        "episode_id": episode_id,
                                        "instruction": task.get_instruction(episode),
                                        "success": 0.0,
                                        "num_steps": 0,
                                        "elapsed_seconds": 0.0,
                                        "error": str(exc),
                                    }

                        all_results.append(result)

                        # Update progress bar
                        failed_count = sum(1 for r in all_results if "error" in r)
                        progress_bar.update(
                            completed=len(all_results) + start_index,
                            failed=failed_count,
                        )

                        # Save per-episode result (strip internal keys)
                        result_to_save = {
                            k: v for k, v in result.items()
                            if not k.startswith("_")
                        }
                        (episode_dir / "result.json").write_text(
                            json.dumps(result_to_save, indent=2)
                        )

                finally:
                    progress_bar.stop()
                    sim.close()
        finally:
            if server:
                server.stop()

        # 5. Build EpisodeRecords for aggregate_results
        records = []
        for r in all_results:
            trajectory = r.pop("_trajectory", [])
            episode = r.pop("_episode", {})
            records.append(EpisodeRecord(
                episode=episode,
                trajectory=trajectory,
                episode_results=r,
            ))

        # 6. Aggregate and save summary
        metric_results = task.aggregate_results(records)
        summary = {"num_episodes": len(all_results), "metrics": metric_results}
        if backend and backend != "legacy":
            summary["llm_usage"] = self._aggregate_llm_usage(all_results)
            summary["model"] = self.model
            summary["backend"] = backend
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Results saved to: %s", run_dir)
        logger.info("Summary: %s", summary)

        return all_results

    def _load_completed_results(self, run_dir: Path, total_episodes: int) -> tuple[list[dict], int]:
        """Scan episode dirs to find the first incomplete episode.

        Walks episode directories in ascending order (by index prefix).
        An episode is "complete" if its directory has a valid result.json.
        Returns results for all consecutive complete episodes from the start,
        and clears all directories from the first incomplete episode onward.

        Args:
            run_dir: The run directory containing episodes/.
            total_episodes: Total number of episodes in the evaluation.

        Returns:
            (completed_results, start_index) tuple.
        """
        import shutil

        episodes_dir = run_dir / "episodes"
        if not episodes_dir.exists():
            return [], 0

        # Collect all episode dirs, sorted by name (which starts with {i:03d}_)
        episode_dirs = sorted(
            [d for d in episodes_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not episode_dirs:
            return [], 0

        # Walk in order, loading results until we hit an incomplete episode
        completed_results = []
        start_index = 0

        for ep_dir in episode_dirs:
            result_file = ep_dir / "result.json"
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    completed_results.append(result)
                    start_index += 1
                    continue
                except (json.JSONDecodeError, OSError):
                    logger.warning("Corrupt result.json in %s, treating as incomplete", ep_dir)
            # First incomplete episode found — stop here
            break

        # Clear all episode dirs from start_index onward
        dirs_to_clear = episode_dirs[start_index:]
        if dirs_to_clear:
            logger.info(
                "Resume: clearing %d episode dirs from index %d onward",
                len(dirs_to_clear), start_index,
            )
            for d in dirs_to_clear:
                shutil.rmtree(d)

        return completed_results, start_index

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

        # Task-specific post-reset setup (e.g., per-episode action space)
        task.on_episode_reset(observation, agent)

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

            # Handle stop signal (e.g., empty plan from LLM)
            if action.action_name == "<<STOP>>":
                logger.info("Agent signalled stop (empty plan), ending episode")
                break

            step_result = sim.step(action)
            trajectory.append(step_result)

            # Get LLM response from agent memory (None for buffered actions)
            llm_response = None
            if hasattr(agent, 'memory') and agent.memory.steps:
                llm_response = agent.memory.steps[-1].llm_response

            # Write step entry to trajectory
            self._write_trajectory_entry(trajectory_path, {
                "step": step + 1,
                "type": "step",
                "action": action.action_name,
                "llm_response": llm_response,
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
        metrics["instruction"] = task_description
        metrics["elapsed_seconds"] = round(elapsed, 2)

        # Attach trajectory and episode for aggregate_results()
        metrics["_trajectory"] = trajectory
        metrics["_episode"] = episode

        # Snapshot LLM usage for this episode
        if hasattr(agent, 'llm_client') and hasattr(agent.llm_client, 'get_usage'):
            metrics["llm_usage"] = agent.llm_client.get_usage()
            agent.llm_client.reset_usage()

        return metrics

    @staticmethod
    def _clear_episode_dir(episode_dir: Path) -> None:
        """Remove all files in an episode directory for a clean retry."""
        for f in episode_dir.iterdir():
            if f.is_file():
                f.unlink()

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

                # Merge YAML generation_kwargs with CLI kwargs (CLI overrides)
                yaml_gen_kwargs = agent_config.get("generation_kwargs", {})
                merged_kwargs = {**yaml_gen_kwargs, **client_kwargs}

                # Local backends need longer timeout (generation is slower than API)
                if base_url and "timeout" not in merged_kwargs:
                    merged_kwargs["timeout"] = 600.0

                llm = LLMClient(
                    model=litellm_model,
                    base_url=base_url,
                    num_retries=self.max_retries,
                    **merged_kwargs,
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

    def _create_simulator(self, simulator_key: str, task=None, label: str = "bridge"):
        import json as _json

        from easi.simulators.registry import (
            create_env_manager,
            get_simulator_entry,
            load_simulator_class,
        )
        from easi.simulators.subprocess_runner import SubprocessRunner

        entry = get_simulator_entry(simulator_key)
        env_manager = create_env_manager(simulator_key)
        SimClass = load_simulator_class(simulator_key)
        sim = SimClass()

        # Auto-install simulator env if not ready
        if not env_manager.env_is_ready():
            logger.info("Simulator environment not ready, auto-installing...")
            env_manager.install()

        # Task-specific bridge overrides simulator default
        bridge_path = (
            (task.get_bridge_script_path() if task else None)
            or sim._get_bridge_script_path()
        )

        extra_args = ["--data-dir", str(self.data_dir)]
        if task and task.simulator_kwargs:
            extra_args.extend(["--simulator-kwargs", _json.dumps(task.simulator_kwargs)])

        # Extract runner-level timeouts from simulator_configs
        sim_configs = task.simulator_configs if task else {}
        runner_kwargs = {}
        if sim_configs.get("command_timeout"):
            runner_kwargs["command_timeout"] = float(sim_configs["command_timeout"])
        if sim_configs.get("startup_timeout"):
            runner_kwargs["startup_timeout"] = float(sim_configs["startup_timeout"])

        # --- Docker runtime path ---
        if entry.runtime == "docker":
            from easi.core.docker_env_manager import DockerEnvironmentManager
            from easi.core.render_platform import get_render_platform

            assert isinstance(env_manager, DockerEnvironmentManager), (
                f"Simulator {simulator_key} declares runtime=docker but env_manager "
                f"is not a DockerEnvironmentManager"
            )

            runner = SubprocessRunner(
                python_executable=env_manager.container_python_path,
                bridge_script_path=bridge_path,
                render_platform=get_render_platform("headless"),
                extra_args=extra_args,
                label=label,
                **runner_kwargs,
            )
            data_dir_str = str(self.data_dir) if self.data_dir else (
                entry.data_dir.replace("~", str(Path.home())) if entry.data_dir else None
            )
            runner.launch_docker(
                docker_env_manager=env_manager,
                data_dir=data_dir_str,
            )
            sim.set_runner(runner)
            return sim, runner

        # --- Conda runtime path (existing) ---

        # Install task-level additional deps
        if task and task.additional_deps:
            env_manager.install_additional_deps(task.additional_deps)

        # Resolve render platform: CLI > task YAML > env_manager default
        from easi.simulators.registry import resolve_render_platform

        yaml_platform = sim_configs.get("render_platform") if task else None
        platform_name = (
            self.render_platform_name
            or yaml_platform
            or env_manager.default_render_platform
        )

        if platform_name not in env_manager.supported_render_platforms:
            raise ValueError(
                f"Render platform '{platform_name}' is not supported by "
                f"{env_manager.simulator_name}:{env_manager.version}. "
                f"Supported: {env_manager.supported_render_platforms}"
            )

        render_platform = resolve_render_platform(simulator_key, platform_name, env_manager=env_manager)

        # Pass platform name to get_env_vars for conditional logic
        from easi.core.render_platform import EnvVars

        env_vars = env_manager.get_env_vars(render_platform_name=platform_name)

        # Merge task-level env_vars from simulator_configs.env_vars
        if task and task.extra_env_vars:
            env_vars = EnvVars.merge(env_vars, EnvVars(replace=task.extra_env_vars))

        # Apply simulator GPU isolation
        if self.sim_gpus is not None:
            gpu_str = ",".join(str(g) for g in self.sim_gpus)
            env_vars = EnvVars.merge(env_vars, EnvVars(replace={"CUDA_VISIBLE_DEVICES": gpu_str}))

        runner = SubprocessRunner(
            python_executable=env_manager.get_python_executable(),
            bridge_script_path=bridge_path,
            render_platform=render_platform,
            screen_config=env_manager.screen_config,
            extra_args=extra_args,
            extra_env=env_vars if env_vars else None,
            label=label,
            **runner_kwargs,
        )
        runner.launch()
        sim.set_runner(runner)

        return sim, runner
