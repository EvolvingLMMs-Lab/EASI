# EASI Standard Prompt Format Reference

This document defines the standard prompt format for EASI benchmarks that do not provide their own prompt format. EmbodiedBench benchmarks (EB-Alfred, EB-Navigation, EB-Habitat, EB-Manipulation) retain their original published formats for reproducibility.

## Scope

**Applies to:** New benchmarks and benchmarks without a published prompt format (VLN-CE R2R, VLN-CE RxR, ManipulaTHOR, AI2-THOR Rearrangement, HAZARD text-plan variants, all future tasks).

**Does not apply to:** EmbodiedBench benchmarks (retain original format), HAZARD multiple-choice format (fundamentally different paradigm).

---

## System Prompt Structure

The system prompt uses markdown sections in this fixed order. Required sections must always be present. Optional sections are included only when the benchmark needs them.

```
## Role and Environment                    [REQUIRED]
## Observation Description                 [OPTIONAL]
## Available Actions                       [REQUIRED]
## Strategy                                [OPTIONAL]
## Guidelines                              [REQUIRED]
## Response Format                         [REQUIRED]
```

### Role and Environment (Required)

One paragraph establishing who the agent is and what environment it operates in. Keep it concise — 2-3 sentences.

```
You are a robot navigating in a 3D indoor environment. You observe the
environment through a front-facing camera and must follow natural language
instructions to navigate to a goal location.
```

### Observation Description (Optional)

Describes what each piece of environment feedback means. Include this section only when the benchmark provides dynamic feedback (geodesic distances, object states, GPS coordinates, etc.) that the LLM needs context to interpret.

```
## Observation Description
- **Distance to goal**: Geodesic (shortest walkable path) distance in meters
  to the goal location. Decreases as you get closer.
- **Held object**: Name of the object currently being held, or "none".
```

Do NOT describe the image observation here — the LLM can see the image directly. Only describe non-visual feedback that appears as text.

### Available Actions (Required)

List all actions the agent can take, with a brief description of what each does. Include any validity constraints.

```
## Available Actions
- move_forward: Move forward by 0.25 meters
- turn_left: Turn left by 30 degrees
- turn_right: Turn right by 30 degrees
- look_up: Tilt camera up by 30 degrees
- look_down: Tilt camera down by 30 degrees
- stop: Stop and end navigation (use ONLY when you believe you have reached
  the destination)
```

For benchmarks with parameterized actions, include the parameter format:

```
- find <receptacle>: Navigate to the named receptacle
- pick_up <object>: Pick up the named object (must be nearby, hands empty)
```

### Strategy (Optional)

Benchmark-specific tactical advice. Include this when the task has non-obvious strategies that improve performance. Keep it actionable.

```
## Strategy
1. Follow the instruction step by step, matching landmarks mentioned
2. Use move_forward to advance and turn_left/turn_right to change direction
3. Use stop ONLY when confident you have reached the described destination
```

### Guidelines (Required)

Universal rules that apply regardless of benchmark. Always include these core guidelines, adding benchmark-specific ones as needed:

```
## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.
```

### Response Format (Required)

Always use the standard 4-field JSON format:

```
## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation, reflect on
        history and feedback",
    "language_plan": "Describe your next plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}

You may include multiple actions in executable_plan. Actions execute
sequentially.
```

---

## Response Format Specification

### JSON Schema

All EASI prompt builders (within scope) must use this response format:

```json
{
    "visual_state_description": "string",
    "reasoning_and_reflection": "string",
    "language_plan": "string",
    "executable_plan": [
        {"action": "action_name"},
        {"action": "action_name"}
    ]
}
```

### Field Definitions

| Field | Type | Purpose |
|-------|------|---------|
| `visual_state_description` | string | Describe what the agent observes in the current image |
| `reasoning_and_reflection` | string | Reason about current state, reflect on history and feedback, explain why previous actions may have failed |
| `language_plan` | string | Natural language description of the planned actions |
| `executable_plan` | array | Ordered list of actions to execute |

### Action Entry Format

Each action in `executable_plan` is an object with an `action` field:

```json
{"action": "move_forward"}
```

Do NOT use `action_id` — numeric IDs are an internal concept. The LLM should always reference actions by name.

For parameterized actions, include a `params` field:

```json
{"action": "find", "params": {"target": "Cabinet_2"}}
```

### Parsing Rules

1. Apply `fix_json()` before parsing (handles common LLM JSON errors)
2. Accept both `{"action": "name"}` and `{"action_name": "name"}`
3. Validate each action name against `memory.action_space`
4. On first invalid action, stop parsing (don't skip — the plan is ordered)
5. On complete parse failure, return empty action list (agent will re-prompt)

---

## Action History

Action history provides the LLM with context about what happened in previous steps. It is a text section embedded in the user message.

### Format

```
## Action History (last N steps)
Step 0: move_forward -> Distance to goal: 8.2m
Step 1: turn_left -> Distance to goal: 8.2m
Step 2: move_forward -> Distance to goal: 7.9m
```

Each entry: `Step {i}: {action_name} -> {feedback}`

If feedback is disabled (`use_feedback: false`), omit the feedback portion:

```
Step 0: move_forward
Step 1: turn_left
Step 2: move_forward
```

### Configuration

| YAML Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `action_history_len` | int | 20 | Maximum entries to include. 0 = disabled. |
| `use_feedback` | bool | true | Include environment feedback in each entry. |

### Data Source

Action history comes from `memory.action_history`, which returns `list[tuple[str, str]]` — pairs of `(action_name, feedback_string)`.

### Truncation

When history exceeds `action_history_len`, keep only the most recent entries:

```python
history = memory.action_history[-self.action_history_len:]
```

---

## Chat History

Chat history provides the LLM with its own previous responses, enabling it to maintain reasoning continuity across steps. It is a text section embedded in the user message, parallel to action history.

### Format

```
## Chat History (last N responses)
[Step 0 Response]
{"visual_state_description": "I see a hallway...", "reasoning_and_reflection": "I need to...", "language_plan": "Move forward...", "executable_plan": [{"action": "move_forward"}]}

[Step 1 Response]
{"visual_state_description": "I see a door...", "reasoning_and_reflection": "The door matches...", "language_plan": "Turn right...", "executable_plan": [{"action": "turn_right"}]}
```

Each entry is the full LLM JSON response from that step, preceded by a `[Step N Response]` header.

### Configuration

| YAML Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `chat_history` | bool | false | Enable chat history section. |
| `message_window_len` | int | 5 | Maximum responses to include. |

### Data Source

Chat history comes from `memory.steps`, which contains `StepRecord` objects with `llm_response` fields:

```python
if self.chat_history:
    responses = [
        s.llm_response for s in memory.steps
        if s.llm_response is not None
    ][-self.message_window_len:]
```

### Interaction with Action History

When both are enabled, action history and chat history appear as separate sections in the user message. Action history provides a compact summary; chat history provides the full reasoning. They are complementary, not redundant.

Recommended default: `action_history_len: 20, chat_history: false`. Enable chat history only for benchmarks where maintaining reasoning continuity significantly improves performance.

---

## Image Handling

### Encoding

All images are encoded as base64 data URLs:

```python
data:image/png;base64,{base64_encoded_data}
```

### Position in Message

Images appear BEFORE text in the user message content array:

```python
content = []
# Images first
content.append({"type": "image_url", "image_url": {"url": img_url}})
# Text after
content.append({"type": "text", "text": prompt_text})
```

### Multiple Images

When a benchmark provides multiple images (e.g., RGB + depth, current + goal), label them:

```python
# In the text portion:
"(Image 1: Current view, Image 2: Goal state)"
```

### Image Sources

| Source | Field | When to Use |
|--------|-------|-------------|
| Primary RGB | `observation.rgb_path` | Always (every benchmark has this) |
| Depth | `observation.metadata["depth_path"]` | When depth sensing is relevant |
| Goal/reference | `observation.metadata["goal_rgb_path"]` | For rearrangement/comparison tasks |
| Multi-view | `observation.metadata["{view}_rgb_path"]` | For panoramic or multi-camera setups |

---

## Environment Feedback

Environment feedback is benchmark-specific dynamic information from the simulator, delivered via `observation.metadata`. The standard defines where feedback appears and how to toggle it, not what it contains.

### Where Feedback Appears

1. **In action history entries**: `"Step N: action -> {feedback_text}"`
2. **As a dedicated section** (optional): For rich contextual feedback that applies to the current state, not just the last action.

```
## Environment Feedback
Distance to goal: 5.3m
```

### Toggle

Controlled by `use_feedback: true/false` in YAML config. When false, omit feedback from action history entries and omit the Environment Feedback section.

### Common Feedback Patterns

| Pattern | Example | Used By |
|---------|---------|---------|
| Distance to goal | `"Distance to goal: 5.3m"` | VLN-CE R2R/RxR |
| Action success/failure | `"success"` / `"fail: object not reachable"` | EB-Alfred, HAZARD |
| Object states | `"Holding: Apple_1"` | Rearrangement, ManipulaTHOR |
| Spatial info | `"Position: (3.2, 0.1, -1.5), Rotation: 90°"` | Rearrangement, ManipulaTHOR |

---

## YAML Configuration Standard

Every prompt builder should accept these common kwargs. Benchmark-specific kwargs can be added below them.

```yaml
agent:
  prompt_builder: "easi.tasks.<benchmark>.prompts.<BuilderClass>"
  prompt_builder_kwargs:
    # Standard kwargs (all prompt builders should support these)
    use_feedback: true            # Include environment feedback
    action_history_len: 20        # Max action history entries (0 = disabled)
    chat_history: false           # Include previous model responses
    message_window_len: 5         # Max chat history entries (when chat_history: true)

    # Benchmark-specific kwargs (examples)
    # use_geo_distance: true      # VLN-CE: show geodesic distance
    # n_shot: 3                   # Few-shot examples count
    # use_depth: false            # Depth image toggle
  generation_kwargs:
    temperature: 0
    max_tokens: 4096
    top_p: 0.95
```

---

## User Message Assembly Order

The user message is assembled in this fixed order:

```
[Image(s)]                           <- base64 encoded, before text
[Text content, assembled as:]
  ## Task                            <- instruction / task description
  ## Observation Description         <- only if defined (from system prompt context)
  ## Environment Feedback            <- current-step feedback (if use_feedback)
  ## Action History (last N steps)   <- if action_history_len > 0 and has history
  ## Chat History (last N responses) <- if chat_history and has history
  [Response format reminder]         <- brief reminder of JSON format
```

On the first turn, action history and chat history are empty and omitted.

---

## Message Structure

Always exactly 2 messages:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [image_parts..., text_part]},
]
```

The system message contains the static prompt (role, actions, strategy, guidelines, response format). The user message contains the dynamic per-step content (images, task instruction, feedback, history).

This is simpler than multi-turn conversation and works consistently across all LLM backends.

---

## Complete Example: Navigation Benchmark

### System Prompt

```
## Role and Environment
You are a robot navigating in a 3D indoor environment. You observe the
environment through a front-facing camera and must follow natural language
instructions to navigate to a goal location.

## Observation Description
- **Distance to goal**: Geodesic distance in meters to the goal. Decreases
  as you approach the destination.

## Available Actions
- move_forward: Move forward by 0.25 meters
- turn_left: Turn left by 15 degrees
- turn_right: Turn right by 15 degrees
- stop: Stop and end navigation (use ONLY when you believe you have reached
  the destination described in the instruction)

## Strategy
1. Carefully read the navigation instruction
2. Observe your surroundings in the image
3. Follow the instruction step by step, matching landmarks and directions
4. Use stop ONLY when confident you have reached the described destination

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.

## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation and history",
    "language_plan": "Describe your next plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}

You may include multiple actions in executable_plan. Actions execute
sequentially.
```

### User Message (Step 5)

```
[Image: base64 encoded current view]

## Task
Walk down the hallway and turn right into the bedroom.

## Environment Feedback
Distance to goal: 5.3m

## Action History (last 5 steps)
Step 0: move_forward -> Distance to goal: 8.2m
Step 1: move_forward -> Distance to goal: 7.9m
Step 2: move_forward -> Distance to goal: 7.6m
Step 3: turn_right -> Distance to goal: 7.6m
Step 4: move_forward -> Distance to goal: 7.3m

Respond with the JSON format specified above.
```

---

## Complete Example: Object Manipulation Benchmark

### System Prompt

```
## Role and Environment
You are a robotic arm in an indoor environment. You can pick up, place, and
manipulate objects on a table using discrete actions.

## Observation Description
- **Held object**: The object currently in the gripper, or "none".
- **Nearby objects**: Objects within interaction range and their positions.

## Available Actions
- move_to <object>: Move the arm to the named object
- pick_up <object>: Grasp the named object (must be nearby, gripper empty)
- place_on <receptacle>: Place held object on the named receptacle
- open <receptacle>: Open a closed receptacle
- close <receptacle>: Close an open receptacle
- done: Signal task completion

## Strategy
1. Locate the target object before attempting to pick it up
2. Ensure your gripper is empty before picking up a new object
3. Navigate to the destination before placing an object

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.

## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation and history",
    "language_plan": "Describe your next plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}
```

### User Message (Step 3, with chat history enabled)

```
[Image: base64 encoded current view]

## Task
Pick up the apple and place it in the bowl.

## Environment Feedback
Held object: none
Nearby objects: Apple_1 (0.3m), Bowl_2 (1.2m)

## Action History (last 3 steps)
Step 0: move_to Apple_1 -> success
Step 1: pick_up Apple_1 -> fail: object not reachable
Step 2: move_to Apple_1 -> success

## Chat History (last 2 responses)
[Step 1 Response]
{"visual_state_description": "I see a red apple on the counter...", "reasoning_and_reflection": "I moved to the apple successfully...", "language_plan": "Pick up the apple", "executable_plan": [{"action": "pick_up Apple_1"}]}

[Step 2 Response]
{"visual_state_description": "The apple is still on the counter...", "reasoning_and_reflection": "Pickup failed, I need to get closer...", "language_plan": "Move closer then pick up", "executable_plan": [{"action": "move_to Apple_1"}]}

Respond with the JSON format specified above.
```
