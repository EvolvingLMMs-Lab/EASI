"""Parse and apply ``--episodes`` filter expressions.

Syntax (comma-separated tokens, freely mixed)::

    :N          index slice — first N episodes
    M:N         index slice — episodes at indices M..N-1
    M:          index slice — from index M onwards
    42          episode ID "42"

Examples::

    --episodes :10            first 10 (replaces old --max-episodes)
    --episodes 30:40          index range 30-39
    --episodes 2,5,7          episode IDs 2, 5, 7
    --episodes 2,10:20,40     episode ID 2 + range 10-19 + episode ID 40

Semantics:
- All selections are unioned and deduplicated.
- Original dataset order is preserved.
- If a requested episode ID is not found, raises ValueError.
"""

from __future__ import annotations


def parse_episodes_flag(value: str) -> tuple[list[tuple[int | None, int | None]], list[str]]:
    """Parse ``--episodes`` value into index slices and episode IDs.

    Returns:
        (slices, ids) where slices is a list of (start, stop) tuples
        and ids is a list of episode ID strings.
    """
    slices: list[tuple[int | None, int | None]] = []
    ids: list[str] = []

    for token in value.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            parts = token.split(":", 1)
            start_s, stop_s = parts[0].strip(), parts[1].strip()
            start = int(start_s) if start_s else None
            stop = int(stop_s) if stop_s else None

            # Validate
            if start is not None and stop is not None and start >= stop:
                raise ValueError(
                    f"Invalid range '{token}': start ({start}) must be less than stop ({stop})"
                )
            if start is not None and start < 0:
                raise ValueError(f"Invalid range '{token}': negative index")
            if stop is not None and stop < 0:
                raise ValueError(f"Invalid range '{token}': negative index")

            slices.append((start, stop))
        else:
            ids.append(token)

    return slices, ids


def filter_episodes(
    episodes: list[dict],
    episodes_flag: str,
) -> list[dict]:
    """Filter an episode list according to an ``--episodes`` expression.

    Args:
        episodes: Full episode list from task.load_episodes().
        episodes_flag: Raw ``--episodes`` CLI value.

    Returns:
        Filtered episode list in original order.

    Raises:
        ValueError: If a requested episode ID is not found, or the
            expression is invalid.
    """
    slices, ids = parse_episodes_flag(episodes_flag)

    if not slices and not ids:
        raise ValueError("Empty --episodes value")

    # Collect selected indices (as a set for deduplication)
    selected_indices: set[int] = set()

    # Apply index slices
    for start, stop in slices:
        s = start if start is not None else 0
        e = stop if stop is not None else len(episodes)
        # Clamp to valid range
        s = max(0, min(s, len(episodes)))
        e = max(0, min(e, len(episodes)))
        for i in range(s, e):
            selected_indices.add(i)

    # Apply episode ID selections
    if ids:
        # Build ID -> index map
        id_to_indices: dict[str, int] = {}
        for i, ep in enumerate(episodes):
            ep_id = str(ep.get("episode_id", ""))
            # First occurrence wins (IDs should be unique, but be safe)
            if ep_id not in id_to_indices:
                id_to_indices[ep_id] = i

        missing = [eid for eid in ids if eid not in id_to_indices]
        if missing:
            raise ValueError(
                f"Episode IDs not found in dataset: {', '.join(missing)}"
            )

        for eid in ids:
            selected_indices.add(id_to_indices[eid])

    # Return in original order
    return [episodes[i] for i in sorted(selected_indices)]
