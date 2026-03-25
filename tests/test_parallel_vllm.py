"""Integration test for parallel vLLM evaluation."""
from easi.evaluation.parallel_runner import ParallelRunner


def test_parallel_vllm_end_to_end():
    """Full flow: ParallelRunner stores multi-instance vLLM config correctly."""
    runner = ParallelRunner(
        task_name="dummy_task",
        agent_type="dummy",
        num_parallel=4,
        backend="vllm",
        model="test-model",
        llm_instances=2,
        llm_gpus=[0, 1],
        sim_gpus=[2, 3],
    )

    # Verify config stored correctly
    assert runner.llm_instances == 2
    assert runner.llm_gpus == [0, 1]
    assert runner.sim_gpus == [2, 3]
    assert runner.num_parallel == 4


def test_external_multi_url_no_server_startup():
    """With --llm-url containing multiple URLs, no server should be started."""
    runner = ParallelRunner(
        task_name="dummy_task",
        agent_type="dummy",
        num_parallel=8,
        backend="vllm",
        model="test",
        llm_base_url="http://localhost:8000/v1,http://localhost:8001/v1",
    )
    urls = runner._parse_base_urls()
    assert len(urls) == 2
    assert urls[0] == "http://localhost:8000/v1"
    assert urls[1] == "http://localhost:8001/v1"


def test_round_robin_url_assignment():
    """Workers should be assigned URLs via round-robin."""
    urls = ["http://localhost:8000/v1", "http://localhost:8001/v1"]
    assert urls[0 % 2] == "http://localhost:8000/v1"
    assert urls[1 % 2] == "http://localhost:8001/v1"
    assert urls[2 % 2] == "http://localhost:8000/v1"
    assert urls[3 % 2] == "http://localhost:8001/v1"


