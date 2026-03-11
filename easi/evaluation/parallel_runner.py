"""Thread-pool based parallel evaluation runner.

Extends EvaluationRunner with concurrent episode execution:
1. Load task once (shared read-only across workers)
2. Fill a queue with (index, episode) tuples
3. Launch N worker threads, each with its own simulator + agent
4. Workers pull episodes from the queue and run them via inherited _run_episode()
5. Collect results thread-safely, aggregate metrics, save summary
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from easi.core.episode import EpisodeRecord
from easi.evaluation.runner import EvaluationRunner, _sanitize_dirname
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _get_gpu_count() -> int | None:
    """Detect the number of GPUs via nvidia-smi.

    Returns the GPU count, or None if detection fails (e.g., no GPUs,
    nvidia-smi not installed).
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


class ParallelRunner(EvaluationRunner):
    """Thread-pool based parallel evaluation runner.

    Each worker thread owns its own simulator and agent instance.
    Episodes are distributed via a shared queue.
    """

    def __init__(self, *, num_parallel: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_parallel = num_parallel
        self._validate_gpu_args()

    def _serialize_cli_options(self) -> dict:
        """Add num_parallel to the serialized config."""
        base = super()._serialize_cli_options()
        base["num_parallel"] = self.num_parallel
        return base

    def _validate_gpu_args(self):
        """Validate GPU allocation arguments."""
        if self.llm_instances and self.llm_instances > 1 and not self.llm_gpus:
            raise ValueError(
                "--llm-gpus is required when --llm-instances > 1. "
                "Specify which GPUs to use for LLM inference."
            )
        if self.llm_gpus and self.sim_gpus:
            overlap = set(self.llm_gpus) & set(self.sim_gpus)
            if overlap:
                raise ValueError(
                    f"--llm-gpus and --sim-gpus must not overlap. "
                    f"Overlapping GPU IDs: {overlap}"
                )
        # Warn if local-server args are set but backend is not a local backend
        if self.backend and self.backend not in ("vllm", "custom"):
            ignored = []
            if self.llm_instances:
                ignored.append("--llm-instances")
            if self.llm_gpus:
                ignored.append("--llm-gpus")
            if ignored:
                logger.warning(
                    "%s will be ignored because --backend is '%s' (not a local LLM backend).",
                    ", ".join(ignored), self.backend,
                )
        # Validate GPU IDs against hardware
        all_gpu_ids = set()
        if self.llm_gpus:
            all_gpu_ids.update(self.llm_gpus)
        if self.sim_gpus:
            all_gpu_ids.update(self.sim_gpus)
        if all_gpu_ids:
            gpu_count = _get_gpu_count()
            if gpu_count is not None:
                invalid = {g for g in all_gpu_ids if g < 0 or g >= gpu_count}
                if invalid:
                    raise ValueError(
                        f"GPU IDs {sorted(invalid)} do not exist. "
                        f"This machine has {gpu_count} GPU(s) "
                        f"(valid IDs: 0-{gpu_count - 1})."
                    )

    def _parse_base_urls(self) -> list[str | None]:
        """Parse base URL(s) into list for round-robin assignment."""
        if self.llm_base_url:
            return [u.strip() for u in self.llm_base_url.split(",") if u.strip()]
        return [None]

    def run(self) -> list[dict]:
        """Run evaluation with thread-pool parallelism."""
        logger.trace(
            "ParallelRunner.run() called: task=%s, num_parallel=%d",
            self.task_name, self.num_parallel,
        )

        # --- Resolve LLM backend and vLLM URLs ---
        backend, base_url = self._resolve_llm_backend()
        server_mgr = None
        self._render_platform = None

        try:
            if backend in ("vllm", "custom") and base_url is None:
                # Auto-manage vLLM instances
                from easi.llm.server_manager import MultiServerManager
                from easi.llm.utils import parse_llm_kwargs, split_kwargs as _split

                all_kw = parse_llm_kwargs(self.llm_kwargs_raw)
                srv_kw, _ = _split(all_kw)

                num_instances = self.llm_instances or 1
                gpu_ids = self.llm_gpus

                server_mgr = MultiServerManager(
                    model=self.model,
                    num_instances=num_instances,
                    gpu_ids=gpu_ids,
                    base_port=self.port,
                    server_kwargs=srv_kw,
                    backend=backend,
                )
                base_urls = server_mgr.start()
            elif base_url:
                base_urls = self._parse_base_urls()
            else:
                base_urls = [None]

            # Setup render platform (starts external services like Xorg if needed)
            self._render_platform = self._setup_render_platform(backend)

            # --- Load task ---
            logger.trace("Loading task")
            task = self._create_task()
            if self.refresh_data:
                task.download_dataset(force=True)
            episodes = task.load_episodes()
            if self.max_episodes is not None:
                episodes = episodes[: self.max_episodes]
            logger.trace(
                "Task loaded. %d episodes, simulator_key=%s",
                len(episodes), task.simulator_key,
            )

            # --- Resolve LLM backend + handle resume ---
            logger.trace("Resolved LLM backend=%s, base_url=%s", backend, base_url)

            # Compute resolved generation kwargs (YAML defaults + CLI overrides)
            from easi.llm.utils import parse_llm_kwargs, split_kwargs

            agent_config = task._config.get("agent", {})
            yaml_gen_kwargs = agent_config.get("generation_kwargs", {})
            all_llm_kwargs = parse_llm_kwargs(self.llm_kwargs_raw)
            _, cli_gen_kwargs = split_kwargs(all_llm_kwargs)
            resolved_gen_kwargs = {**yaml_gen_kwargs, **cli_gen_kwargs}

            # Handle resume
            if self.resume_dir:
                run_dir = self.resume_dir
                completed_results, start_index = self._load_completed_results(
                    run_dir, len(episodes),
                )
                self._reattach_resume_data(completed_results, episodes, run_dir)
                logger.info(
                    "Resuming from %s — %d completed, starting from index %d",
                    run_dir, len(completed_results), start_index,
                )
            else:
                run_dir = self.output_dir / self.task_name / self.run_id
                completed_results = []
                start_index = 0

            # --- Create output directory and save config ---
            logger.trace("Creating output directory and saving config")
            episodes_dir = run_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "run_id": self.run_id,
                "total_episodes": len(episodes),
                "num_parallel": self.num_parallel,
                "cli_options": self._serialize_cli_options(),
                "resolved_backend": backend,
                "resolved_base_url": base_url,
                "resolved_generation_kwargs": resolved_gen_kwargs,
                "task_config": task._config,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
            logger.trace("Run config:\n%s", json.dumps(config, indent=2, default=str))

            # Check if all episodes already complete (resume edge case)
            if start_index >= len(episodes):
                logger.info("All %d episodes already complete, re-aggregating summary.", len(episodes))
                all_results = completed_results
                # Skip to aggregation
                wall_seconds = 0.0
                results_list = [(i, r) for i, r in enumerate(all_results)]
            else:
                # --- Fill episode queue (from start_index) ---
                episode_queue: queue.Queue[tuple[int, dict]] = queue.Queue()
                for i, episode in enumerate(episodes):
                    if i >= start_index:
                        episode_queue.put((i, episode))
                remaining = episode_queue.qsize()
                logger.trace("Queued %d episodes (skipped %d completed)", remaining, start_index)

                # --- Prepare thread-safe collection ---
                results_lock = threading.Lock()
                new_results: list[tuple[int, dict]] = []
                progress = {"completed": 0, "failed": 0}
                progress_lock = threading.Lock()
                total_episodes = len(episodes)

                num_workers = min(self.num_parallel, remaining)

                # --- Progress bar ---
                from easi.utils.progress import ProgressBar

                progress_bar = ProgressBar(
                    total=total_episodes,
                    num_workers=num_workers,
                    start_index=start_index,
                )
                def _worker(worker_id: int) -> None:
                    """Worker thread: owns a simulator + agent, pulls episodes from queue."""
                    logger.trace("[Worker %d] Starting up", worker_id)
                    episodes_done = 0
                    sim = None

                    try:
                        # Create simulator
                        logger.trace(
                            "[Worker %d] Creating simulator (key=%s)",
                            worker_id, task.simulator_key,
                        )
                        sim, sim_runner = self._create_simulator(
                            task.simulator_key, task=task, label=f"bridge-{worker_id}",
                            worker_id=worker_id,
                        )
                        logger.trace(
                            "[Worker %d] Simulator ready (PID=%s)",
                            worker_id,
                            getattr(sim_runner, 'pid', 'unknown'),
                        )

                        # Create agent
                        logger.trace("[Worker %d] Creating agent", worker_id)
                        # Round-robin URL assignment
                        worker_url = base_urls[worker_id % len(base_urls)]
                        agent = self._create_agent(
                            task.action_space, task._config,
                            backend=backend, base_url=worker_url,
                        )
                        logger.trace("[Worker %d] Agent ready", worker_id)
                        while True:
                            # Pull next episode
                            try:
                                idx, episode = episode_queue.get_nowait()
                            except queue.Empty:
                                break

                            logger.trace(
                                "[Worker %d] Queue remaining: ~%d",
                                worker_id, episode_queue.qsize(),
                            )

                            episode_id = episode.get("episode_id", f"ep_{idx}")
                            episode_dir = episodes_dir / f"{idx:03d}_{_sanitize_dirname(episode_id)}"
                            episode_dir.mkdir(exist_ok=True)

                            result = None
                            for attempt in range(1, self.max_retries + 1):
                                logger.trace(
                                    "[Worker %d] Running episode %s (attempt %d/%d)",
                                    worker_id, episode_id, attempt, self.max_retries,
                                )
                                try:
                                    result = self._run_episode(
                                        sim, agent, task, episode, idx, episode_dir,
                                    )
                                    logger.trace(
                                        "[Worker %d] Episode %s completed in %.1fs: %s",
                                        worker_id, episode_id,
                                        result.get("elapsed_seconds", 0),
                                        {k: v for k, v in result.items()
                                         if k in ("success", "num_steps", "elapsed_seconds")},
                                    )
                                    break
                                except Exception as exc:
                                    logger.warning(
                                        "[Worker %d] Episode %s attempt %d/%d failed: %s",
                                        worker_id, episode_id, attempt, self.max_retries, exc,
                                    )
                                    logger.trace(
                                        "[Worker %d] Exception details:",
                                        worker_id, exc_info=True,
                                    )
                                    self._clear_episode_dir(episode_dir)
                                    if attempt < self.max_retries:
                                        logger.info(
                                            "[Worker %d] Re-launching simulator for retry...",
                                            worker_id,
                                        )
                                        try:
                                            sim.close()
                                        except Exception:
                                            pass
                                        try:
                                            sim, sim_runner = self._create_simulator(
                                                task.simulator_key, task=task,
                                                label=f"bridge-{worker_id}",
                                                worker_id=worker_id,
                                            )
                                        except Exception as restart_exc:
                                            logger.error(
                                                "[Worker %d] Simulator restart failed: %s",
                                                worker_id, restart_exc,
                                            )
                                            result = {
                                                "episode_id": episode_id,
                                                "instruction": task.get_instruction(episode),
                                                "success": 0.0,
                                                "num_steps": 0,
                                                "elapsed_seconds": 0.0,
                                                "error": f"simulator restart failed: {restart_exc}",
                                            }
                                            sim = None
                                            break
                                    else:
                                        logger.error(
                                            "[Worker %d] Episode %s failed after %d attempts, skipping",
                                            worker_id, episode_id, self.max_retries,
                                        )
                                        result = {
                                            "episode_id": episode_id,
                                            "instruction": task.get_instruction(episode),
                                            "success": 0.0,
                                            "num_steps": 0,
                                            "elapsed_seconds": 0.0,
                                            "error": str(exc),
                                        }

                            # Save per-episode result (strip internal keys)
                            result_to_save = {
                                k: v for k, v in result.items()
                                if not k.startswith("_")
                            }
                            (episode_dir / "result.json").write_text(
                                json.dumps(result_to_save, indent=2)
                            )

                            # Thread-safe results collection
                            failed = "error" in result
                            with results_lock:
                                new_results.append((idx, result))

                            with progress_lock:
                                progress["completed"] += 1
                                if failed:
                                    progress["failed"] += 1
                                current_completed = progress["completed"] + start_index
                                current_failed = progress["failed"]

                            progress_bar.update(
                                completed=current_completed,
                                failed=current_failed,
                                active_workers=num_workers,
                            )

                            episodes_done += 1

                            # If simulator restart failed, stop this worker
                            if sim is None:
                                logger.warning(
                                    "[Worker %d] No simulator, stopping worker", worker_id,
                                )
                                break

                    finally:
                        logger.trace("[Worker %d] Shutting down simulator", worker_id)
                        if sim is not None:
                            try:
                                sim.close()
                            except Exception:
                                pass
                        logger.trace(
                            "[Worker %d] Shutdown complete (%d episodes done)",
                            worker_id, episodes_done,
                        )

                # --- Launch worker threads ---
                logger.trace("Launching %d worker threads", num_workers)
                wall_start = time.monotonic()

                with progress_bar, ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for wid in range(num_workers):
                        futures.append(executor.submit(_worker, wid))
                    logger.trace("All %d worker threads submitted", num_workers)

                    # Wait for all workers to complete and propagate exceptions
                    for future in futures:
                        future.result()

                wall_seconds = round(time.monotonic() - wall_start, 2)

                # Merge completed results from resume with new results
                new_results.sort(key=lambda x: x[0])
                results_list = [(i, r) for i, r in enumerate(completed_results)]
                results_list.extend(new_results)

            # --- Sort results and aggregate ---
            results_list.sort(key=lambda x: x[0])
            all_results = [r for _, r in results_list]

            num_successful = sum(1 for r in all_results if "error" not in r)
            num_failed = len(all_results) - num_successful
            logger.trace(
                "Results sorted. %d successful, %d failed",
                num_successful, num_failed,
            )

            # Build EpisodeRecords for aggregate_results
            effective = sum(1 for r in all_results if "error" not in r)
            records = []
            for r in all_results:
                trajectory = r.pop("_trajectory", [])
                episode = r.pop("_episode", {})
                episode_results = {k: v for k, v in r.items() if not k.startswith("_")}
                records.append(EpisodeRecord(
                    episode=episode,
                    trajectory=trajectory,
                    episode_results=episode_results,
                ))

            # Aggregate and save summary
            try:
                metric_results = task.aggregate_results(records)
            except Exception as exc:
                logger.error("aggregate_results() failed: %s", exc, exc_info=True)
                metric_results = {"aggregation_error": str(exc)}
            summary = {
                "num_episodes": len(all_results),
                "effective_episodes": effective,
                "metrics": metric_results,
            }
            summary["num_parallel"] = self.num_parallel
            summary["wall_clock_seconds"] = wall_seconds
            if backend and backend != "legacy":
                summary["llm_usage"] = self._aggregate_llm_usage(all_results)
                summary["model"] = self.model
                summary["backend"] = backend
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            logger.info("Results saved to: %s", run_dir)
            logger.info("Summary: %s", summary)

            return all_results
        finally:
            if server_mgr is not None:
                server_mgr.stop()
            if self._render_platform is not None:
                self._render_platform.teardown()
