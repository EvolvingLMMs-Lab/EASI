"""EASI CLI entry point.

Usage:
    easi env list|install|check <simulator>
    easi task list|info|download <task>
    easi sim test <simulator>
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

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run a full evaluation", parents=[common])
    run_parser.add_argument("task", type=str, nargs="?", default=None,
                            help="Task name (e.g., 'dummy_task', 'ebalfred_base'). "
                                 "Optional when --resume is provided.")
    run_parser.add_argument("--agent", type=str, default="react", choices=["dummy", "react"])
    run_parser.add_argument("--output-dir", type=str, default="./logs",
                            help="Base output directory (creates <dir>/<task>/<run_id>/)")
    run_parser.add_argument("--data-dir", type=str, default="./datasets",
                            help="Directory for downloading/caching datasets (default: ./datasets)")
    run_parser.add_argument("--max-episodes", type=int, default=None)
    run_parser.add_argument("--llm-url", type=str, default=None, help="LLM server URL")
    run_parser.add_argument("--seed", type=int, default=None)
    # New LLM backend args
    run_parser.add_argument("--backend", type=str, default=None,
                            help="LLM backend: vllm, openai, anthropic, gemini, dummy")
    run_parser.add_argument("--model", type=str, default="default",
                            help="Model name (HF path for vLLM, API name for proprietary)")
    run_parser.add_argument("--port", type=int, default=8080,
                            help="Port for local inference server (default: 8080)")
    run_parser.add_argument("--llm-kwargs", type=str, default=None,
                            help='JSON string of extra kwargs, e.g. \'{"tensor_parallel_size": 4}\'')
    run_parser.add_argument("--max-retries", type=int, default=3,
                            help="Max LLM retry attempts on transient errors (default: 3)")
    run_parser.add_argument("--resume", type=str, default=None,
                            help="Path to a previous run directory to resume from")

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
    from easi.simulators.registry import load_env_manager_class

    EnvManagerClass = load_env_manager_class(simulator)
    env_manager = EnvManagerClass()

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
    from easi.simulators.registry import load_env_manager_class

    EnvManagerClass = load_env_manager_class(simulator)
    env_manager = EnvManagerClass()

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


def cmd_task_download(task_name: str) -> None:
    from easi.tasks.registry import load_task_class

    TaskClass = load_task_class(task_name)
    task = TaskClass()
    path = task.download_dataset()
    if path and str(path):
        logger.info("Dataset ready at: %s", path)
    else:
        logger.info("Task uses built-in episodes (no download needed).")


def cmd_sim_test(simulator: str, steps: int, timeout: float) -> None:
    from easi.core.episode import Action
    from easi.simulators.registry import load_env_manager_class, load_simulator_class
    from easi.simulators.subprocess_runner import SubprocessRunner

    EnvManagerClass = load_env_manager_class(simulator)
    SimClass = load_simulator_class(simulator)

    env_manager = EnvManagerClass()
    sim = SimClass()

    logger.info("Testing %s...", simulator)
    logger.info("  Python: %s", env_manager.get_python_executable())

    runner = SubprocessRunner(
        python_executable=env_manager.get_python_executable(),
        bridge_script_path=sim._get_bridge_script_path(),
        needs_display=env_manager.needs_display,
        xvfb_screen_config=env_manager.xvfb_screen_config,
        startup_timeout=timeout,
        command_timeout=timeout,
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


def cmd_run(task_name, agent_type, output_dir, data_dir, max_episodes,
            llm_url, seed, backend, model, port, llm_kwargs_raw,
            max_retries, resume):
    import json as _json
    from pathlib import Path

    from easi.evaluation.runner import EvaluationRunner

    # When resuming, load saved config and use as defaults
    if resume:
        config_path = Path(resume) / "config.json"
        if not config_path.exists():
            logger.error("Resume directory has no config.json: %s", resume)
            sys.exit(1)
        saved = _json.loads(config_path.read_text())
        opts = saved.get("cli_options", {})

        # Saved config provides defaults; CLI args override
        task_name = task_name or opts.get("task_name")
        if agent_type == "dummy" and opts.get("agent_type"):
            agent_type = opts["agent_type"]
        if output_dir == "./logs" and opts.get("output_dir"):
            output_dir = opts["output_dir"]
        if data_dir == "./datasets" and opts.get("data_dir"):
            data_dir = opts["data_dir"]
        if llm_url is None:
            llm_url = opts.get("llm_base_url")
        if seed is None:
            seed = opts.get("agent_seed")
        if backend is None:
            backend = opts.get("backend")
        if model == "default" and opts.get("model", "default") != "default":
            model = opts["model"]
        if port == 8080 and opts.get("port", 8080) != 8080:
            port = opts["port"]
        if llm_kwargs_raw is None:
            llm_kwargs_raw = opts.get("llm_kwargs_raw")
        if max_retries == 3 and opts.get("max_retries", 3) != 3:
            max_retries = opts["max_retries"]

    if not task_name:
        logger.error("Task name is required. Provide it as a positional arg or use --resume.")
        sys.exit(1)

    runner = EvaluationRunner(
        task_name=task_name,
        agent_type=agent_type,
        output_dir=output_dir,
        data_dir=data_dir,
        llm_base_url=llm_url,
        agent_seed=seed,
        backend=backend,
        model=model,
        port=port,
        llm_kwargs_raw=llm_kwargs_raw,
        max_retries=max_retries,
        resume_dir=resume,
    )
    results = runner.run(max_episodes=max_episodes)
    logger.info("Completed %d episodes.", len(results))
    from easi.evaluation.metrics import aggregate_metrics

    summary = aggregate_metrics(results)
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
            cmd_task_download(args.task)
        elif args.task_action == "scaffold":
            cmd_task_scaffold(args.name, args.simulator, args.max_steps)
        else:
            parser.parse_args(["task", "--help"])

    elif args.command == "sim":
        if args.sim_action == "test":
            cmd_sim_test(args.simulator, args.steps, args.timeout)
        else:
            parser.parse_args(["sim", "--help"])

    elif args.command == "run":
        cmd_run(args.task, args.agent, args.output_dir, args.data_dir,
                args.max_episodes, args.llm_url, args.seed,
                args.backend, args.model, args.port, args.llm_kwargs,
                args.max_retries, args.resume)

    elif args.command == "llm-server":
        cmd_llm_server(args.host, args.port, args.mode, args.action_space)


if __name__ == "__main__":
    main()
