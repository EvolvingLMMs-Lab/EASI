"""EASI CLI entry point.

Usage:
    easi env list|install|check <simulator>
    easi task list|info|download <task>
    easi sim test <simulator>
    easi start <task> [<task> ...] [--tasks t1,t2]   # Run evaluation
    easi llm-server [--port PORT] [--mode MODE]
"""

import argparse
import sys

from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    # Shared parent so --verbosity works at any position in the command
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--verbosity", type=str, default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )

    parser = argparse.ArgumentParser(
        prog="easi",
        description="EASI - Embodied Reasoning Evaluation for Spatial Intelligence",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- env command group ---
    env_parser = subparsers.add_parser("env", help="Manage simulator environments", parents=[common])
    env_sub = env_parser.add_subparsers(dest="env_action")

    env_sub.add_parser("list", help="List available simulators and versions", parents=[common])

    env_install = env_sub.add_parser("install", help="Install a simulator environment", parents=[common])
    env_install.add_argument("simulator", type=str, help="e.g., 'dummy' or 'ai2thor:v2_1_0'")
    env_install.add_argument("--reinstall", action="store_true",
                             help="Remove existing env and install from scratch")
    env_install.add_argument("--with-task-deps", type=str, default=None, metavar="TASK",
                             help="Also install additional_deps from a task (e.g., 'ebalfred_base')")

    env_check = env_sub.add_parser("check", help="Check if environment is ready", parents=[common])
    env_check.add_argument("simulator", type=str)

    # --- task command group ---
    task_parser = subparsers.add_parser("task", help="Manage tasks (benchmarks)", parents=[common])
    task_sub = task_parser.add_subparsers(dest="task_action")

    task_sub.add_parser("list", help="List available tasks", parents=[common])

    task_info = task_sub.add_parser("info", help="Show task details", parents=[common])
    task_info.add_argument("task", type=str, help="e.g., 'dummy_task'")

    task_download = task_sub.add_parser("download", help="Download task dataset", parents=[common])
    task_download.add_argument("task", type=str)
    task_download.add_argument("--refresh-data", action="store_true", dest="refresh_data",
                               help="Delete cached dataset and re-download from source")

    task_scaffold = task_sub.add_parser("scaffold", help="Generate boilerplate for a new benchmark", parents=[common])
    task_scaffold.add_argument("name", type=str, help="Task name in snake_case (e.g., 'my_benchmark')")
    task_scaffold.add_argument("--simulator", type=str, default="dummy:v1",
                               help="Simulator key (e.g., 'ai2thor:v2_1_0')")
    task_scaffold.add_argument("--max-steps", type=int, default=50)

    # --- sim command group ---
    sim_parser = subparsers.add_parser("sim", help="Control simulators", parents=[common])
    sim_sub = sim_parser.add_subparsers(dest="sim_action")

    sim_test = sim_sub.add_parser("test", help="Run a smoke test (reset + N steps)", parents=[common])
    sim_test.add_argument("simulator", type=str, help="e.g., 'dummy' or 'ai2thor:v5_0_0'")
    sim_test.add_argument("--steps", type=int, default=5, help="Number of steps")
    sim_test.add_argument("--timeout", type=float, default=200.0,
                          help="Bridge startup timeout in seconds (default: 200)")
    sim_test.add_argument(
        "--render-platform", type=str, default=None, dest="render_platform",
        help="Rendering platform override (auto, native, xvfb, egl, headless)")

    # --- start command ---
    start_parser = subparsers.add_parser("start", help="Run a full evaluation", parents=[common])
    # All defaults are None so resume logic can distinguish "user provided" from "default".
    # Real defaults live in EvaluationRunner.__init__.
    start_parser.add_argument("task_names_positional", type=str, nargs="*", default=None, metavar="task",
                              help="Task name(s) (e.g., 'dummy_task', 'ebalfred_base'). "
                                   "Optional when --resume is provided.")
    start_parser.add_argument("--tasks", type=str, default=None, dest="tasks_csv",
                              help="Comma-separated task names (e.g., 'ebalfred_base,ebnavigation_base')")
    start_parser.add_argument("--agent", type=str, default=None, choices=["dummy", "react"],
                              dest="agent_type")
    start_parser.add_argument("--output-dir", type=str, default=None,
                              help="Base output directory (creates <dir>/<task>/<run_id>/)")
    start_parser.add_argument("--data-dir", type=str, default=None,
                              help="Directory for downloading/caching datasets (default: ./datasets)")
    start_parser.add_argument("--max-episodes", type=int, default=None)
    start_parser.add_argument("--llm-url", type=str, default=None, dest="llm_base_url",
                              help="LLM server URL")
    start_parser.add_argument("--seed", type=int, default=None, dest="agent_seed")
    start_parser.add_argument("--backend", type=str, default=None,
                              help="LLM backend: vllm, openai, anthropic, gemini, dummy")
    start_parser.add_argument("--model", type=str, default=None,
                              help="Model name (HF path for vLLM, API name for proprietary)")
    start_parser.add_argument("--port", type=int, default=None,
                              help="Port for local inference server (default: 8080)")
    start_parser.add_argument("--llm-kwargs", type=str, default=None, dest="llm_kwargs_raw",
                              help='JSON string of extra kwargs, e.g. \'{"tensor_parallel_size": 4}\'')
    start_parser.add_argument("--max-retries", type=int, default=None,
                              help="Max LLM retry attempts on transient errors (default: 3)")
    start_parser.add_argument("--num-parallel", type=int, default=None, dest="num_parallel",
                              help="Number of parallel simulator instances (default: 1, sequential). "
                                   "Only supported with proprietary API backends (openai, anthropic, gemini).")
    start_parser.add_argument("--resume", type=str, default=None, dest="resume_dir",
                              help="Path to a previous run directory to resume from")
    start_parser.add_argument("--refresh-data", action="store_true", dest="refresh_data",
                              help="Delete cached dataset and re-download from source")
    start_parser.add_argument(
        "--render-platform", type=str, default=None, dest="render_platform",
        help="Rendering platform: auto, native, xvfb, egl, headless (default: simulator's preference)")

    # --- llm-server command ---
    llm_parser = subparsers.add_parser("llm-server", help="Start dummy LLM server", parents=[common])
    llm_parser.add_argument("--port", type=int, default=8000)
    llm_parser.add_argument("--host", type=str, default="127.0.0.1")
    llm_parser.add_argument("--mode", choices=["fixed", "random"], default="random")
    llm_parser.add_argument(
        "--action-space", type=str, nargs="+",
        default=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
    )

    return parser


# --- Command handlers ---

def cmd_env_list() -> None:
    from easi.simulators.registry import get_simulator_entry, list_simulators

    sims = list_simulators()
    if not sims:
        logger.info("No simulators found.")
        return

    # Deduplicate: show each name:version pair once
    seen = set()
    for key in sims:
        entry = get_simulator_entry(key)
        pair = f"{entry.name}:{entry.version}"
        if pair in seen:
            continue
        seen.add(pair)
        default_marker = " (default)" if key == entry.name else ""
        logger.info("  %s%s  -- %s", pair, default_marker, entry.description)


def cmd_env_install(simulator: str, reinstall: bool = False, with_task_deps: str | None = None) -> None:
    from easi.simulators.registry import create_env_manager

    env_manager = create_env_manager(simulator)

    if reinstall:
        logger.info("Removing existing environment: %s", env_manager.get_env_name())
        env_manager.remove()

    logger.info("Installing environment: %s", env_manager.get_env_name())
    env_manager.install()

    if with_task_deps:
        from easi.tasks.registry import get_task_entry, load_task_class

        entry = get_task_entry(with_task_deps)
        TaskClass = load_task_class(with_task_deps)
        task = TaskClass(split_yaml_path=entry.config_path)
        if task.additional_deps:
            env_manager.install_additional_deps(task.additional_deps)
        else:
            logger.info("Task %s has no additional_deps.", with_task_deps)

    logger.info("Done.")


def cmd_env_check(simulator: str) -> None:
    from easi.simulators.registry import create_env_manager

    env_manager = create_env_manager(simulator)

    missing = env_manager.check_system_deps()
    if missing:
        logger.info("Missing system deps: %s", missing)

    if env_manager.env_is_ready():
        logger.info("Environment %s is ready.", env_manager.get_env_name())
        logger.info("Python: %s", env_manager.get_python_executable())
    else:
        logger.info("Environment %s is NOT ready.", env_manager.get_env_name())
        logger.info("Run: easi env install %s", simulator)


def cmd_task_list() -> None:
    from easi.tasks.registry import get_task_entry, list_tasks

    tasks = list_tasks()
    if not tasks:
        logger.info("No tasks found.")
        return

    for name in tasks:
        entry = get_task_entry(name)
        logger.info("  %s  -- %s (simulator: %s)", name, entry.display_name, entry.simulator_key)


def cmd_task_info(task_name: str) -> None:
    from easi.tasks.registry import get_task_entry

    entry = get_task_entry(task_name)
    logger.info("Task: %s", entry.display_name)
    logger.info("  Name:        %s", entry.name)
    logger.info("  Description: %s", entry.description)
    logger.info("  Simulator:   %s", entry.simulator_key)
    logger.info("  Max steps:   %s", entry.max_steps)


def cmd_task_scaffold(name: str, simulator: str, max_steps: int) -> None:
    from pathlib import Path

    from easi.tasks.scaffold import scaffold_task

    tasks_dir = Path(__file__).parent / "tasks"
    tests_dir = Path(__file__).parent.parent / "tests"
    task_dir = scaffold_task(name, simulator, output_dir=tasks_dir,
                             max_steps=max_steps, tests_dir=tests_dir)
    logger.info("Created task scaffold at: %s", task_dir)
    logger.info("Next steps:")
    logger.info("  1. Edit %s/bridge.py — implement _create_env() and _extract_image()", task_dir.name)
    logger.info("  2. Edit %s/task.py — implement format_reset_config()", task_dir.name)
    logger.info("  3. Edit %s/%s.yaml — configure dataset source", task_dir.name, name)
    logger.info("  4. Run tests: pytest tests/test_%s.py -v", name)


def cmd_task_download(task_name: str, refresh_data: bool = False) -> None:
    from easi.tasks.registry import load_task_class

    TaskClass = load_task_class(task_name)
    task = TaskClass()
    path = task.download_dataset(force=refresh_data)
    if path and str(path):
        logger.info("Dataset ready at: %s", path)
    else:
        logger.info("Task uses built-in episodes (no download needed).")


def cmd_sim_test(simulator: str, steps: int, timeout: float, render_platform_name: str | None = None) -> None:
    from easi.core.episode import Action
    from easi.core.render_platform import get_render_platform
    from easi.simulators.registry import create_env_manager, load_simulator_class
    from easi.simulators.subprocess_runner import SubprocessRunner

    env_manager = create_env_manager(simulator)
    SimClass = load_simulator_class(simulator)
    sim = SimClass()

    # Resolve render platform
    platform_name = render_platform_name or env_manager.default_render_platform
    if platform_name not in env_manager.supported_render_platforms:
        logger.error(
            "Render platform '%s' not supported by %s. Supported: %s",
            platform_name, simulator, env_manager.supported_render_platforms,
        )
        sys.exit(1)
    render_platform = get_render_platform(platform_name)

    logger.info("Testing %s...", simulator)
    logger.info("  Python: %s", env_manager.get_python_executable())
    logger.info("  Render platform: %s", platform_name)

    env_vars = env_manager.get_env_vars(render_platform_name=platform_name)

    runner = SubprocessRunner(
        python_executable=env_manager.get_python_executable(),
        bridge_script_path=sim._get_bridge_script_path(),
        render_platform=render_platform,
        screen_config=env_manager.screen_config,
        startup_timeout=timeout,
        command_timeout=timeout,
        extra_env=env_vars or None,
    )

    try:
        runner.launch()
        sim.set_runner(runner)

        logger.info("  Reset...")
        obs = sim.reset("smoke_test_001")
        logger.info("  Reset OK (rgb: %s)", obs.rgb_path)

        for i in range(steps):
            action = Action(action_name="MoveAhead")
            result = sim.step(action)
            logger.info("  Step %d: done=%s, reward=%s", i + 1, result.done, result.reward)
            if result.done:
                break

        logger.info("  Closing...")
        sim.close()
        logger.info("  Close OK")
        logger.info("Smoke test passed!")

    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down bridge...")
        sim.close()
        logger.info("Bridge process terminated.")
        sys.exit(130)
    except Exception as e:
        logger.error("Smoke test FAILED: %s", e)
        sim.close()
        sys.exit(1)


def _resolve_task_list(args_ns) -> list[str]:
    """Build task list from positional args and/or --tasks flag."""
    tasks: list[str] = []
    if args_ns.tasks_csv:
        tasks = [t.strip() for t in args_ns.tasks_csv.split(",") if t.strip()]
    elif args_ns.task_names_positional:
        tasks = args_ns.task_names_positional
    return tasks


def cmd_start(args):
    import json as _json
    from pathlib import Path

    from easi.evaluation.runner import EvaluationRunner

    task_list = _resolve_task_list(args)

    # Collect explicitly-provided CLI args (argparse defaults are None)
    raw = {k: v for k, v in vars(args).items() if v is not None}
    # Remove argparse internals that aren't runner params
    for key in ("command", "verbosity", "task_names_positional", "tasks_csv"):
        raw.pop(key, None)

    # Extract session-specific params (not saved in config.json)
    resume_dir = raw.pop("resume_dir", None)
    redownload = raw.pop("refresh_data", False)

    if resume_dir:
        if len(task_list) > 1:
            logger.error("--resume cannot be used with multiple tasks.")
            sys.exit(1)
        config_path = Path(resume_dir) / "config.json"
        if not config_path.exists():
            logger.error("Resume directory has no config.json: %s", resume_dir)
            sys.exit(1)
        saved = _json.loads(config_path.read_text()).get("cli_options", {})
        # Saved values fill gaps; explicit CLI args win
        run_kwargs = {**saved, **raw}
        # If no task was given on CLI, pull from saved config
        if not task_list:
            saved_task = saved.get("task_name")
            if saved_task:
                task_list = [saved_task]
    else:
        run_kwargs = raw

    if not task_list:
        logger.error("Task name is required. Provide it as a positional arg, --tasks, or use --resume.")
        sys.exit(1)

    # Remove task_name from run_kwargs; it's passed per-task below
    run_kwargs.pop("task_name", None)
    num_parallel = run_kwargs.pop("num_parallel", None) or 1

    from easi.evaluation.metrics import aggregate_metrics

    all_summaries: list[tuple[str, dict]] = []

    for task_name in task_list:
        logger.info("=== Starting evaluation: %s ===", task_name)

        if num_parallel > 1:
            from easi.evaluation.parallel_runner import ParallelRunner
            runner = ParallelRunner(
                task_name=task_name,
                num_parallel=num_parallel,
                **run_kwargs,
                resume_dir=resume_dir,
                refresh_data=redownload,
            )
        else:
            runner = EvaluationRunner(
                task_name=task_name,
                **run_kwargs,
                resume_dir=resume_dir,
                refresh_data=redownload,
            )

        results = runner.run()
        logger.info("Completed %d episodes for %s.", len(results), task_name)

        summary = aggregate_metrics(results)
        all_summaries.append((task_name, summary))
        for key, value in summary.items():
            logger.info("  %s: %s", key, value)

    # Combined summary when multiple tasks were evaluated
    if len(all_summaries) > 1:
        logger.info("")
        logger.info("=== Combined Summary ===")
        for task_name, summary in all_summaries:
            logger.info("[%s]", task_name)
            for key, value in summary.items():
                logger.info("  %s: %s", key, value)


def cmd_llm_server(host: str, port: int, mode: str, action_space: list[str]) -> None:
    from easi.llm.dummy_server import run_server

    run_server(host=host, port=port, mode=mode, action_space=action_space)


# --- Main ---

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbosity)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch commands
    if args.command == "env":
        if args.env_action == "list":
            cmd_env_list()
        elif args.env_action == "install":
            cmd_env_install(args.simulator, reinstall=args.reinstall, with_task_deps=args.with_task_deps)
        elif args.env_action == "check":
            cmd_env_check(args.simulator)
        else:
            parser.parse_args(["env", "--help"])

    elif args.command == "task":
        if args.task_action == "list":
            cmd_task_list()
        elif args.task_action == "info":
            cmd_task_info(args.task)
        elif args.task_action == "download":
            cmd_task_download(args.task, refresh_data=args.refresh_data)
        elif args.task_action == "scaffold":
            cmd_task_scaffold(args.name, args.simulator, args.max_steps)
        else:
            parser.parse_args(["task", "--help"])

    elif args.command == "sim":
        if args.sim_action == "test":
            cmd_sim_test(args.simulator, args.steps, args.timeout, getattr(args, "render_platform", None))
        else:
            parser.parse_args(["sim", "--help"])

    elif args.command == "start":
        cmd_start(args)

    elif args.command == "llm-server":
        cmd_llm_server(args.host, args.port, args.mode, args.action_space)


if __name__ == "__main__":
    main()
