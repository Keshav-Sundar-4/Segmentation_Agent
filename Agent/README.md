# BioVision Agent

LangGraph pipeline that turns a metadata YAML description of a microscopy dataset into a
validated, executed Python preprocessing script — with zero manual intervention.

---

## Graph

### Full topology

```
                      ┌─────────────────────────────────────────────────────┐
                      │                  RETRY LOOP (max 3)                 │
                      │                                                      │
  START               ▼           direct           conditional               │
    │        ┌──────────────┐      edge      ┌──────────────────┐  failure  │
    └───────► │   planner    │ ─────────────► │      coder       │ ──────────┘
              └──────────────┘                └──────────────────┘
                                                       │
                                                 direct edge
                                                       │
                                                       ▼
                                            ┌────────────────────┐
                                            │  sandbox_executor  │  ← Docker · sample_dir
                                            └────────────────────┘
                                                       │
                                          ┌────────────┴────────────┐
                                     success                      failure
                                          │                          │
                                          ▼                          │
                                 ┌────────────────┐         retries < MAX_RETRIES?
                                 │ local_executor │              │         │
                                 └────────────────┘            yes        no
                                          │                     │         │
                                    direct edge          back to coder    │
                                          │                          ┌────┘
                                          ▼                          ▼
                                         END ◄──────────── local_executor
                                                          (passes failure state
                                                           through to caller)
```

### Nodes

| Node | Type | Model / Backend | Runs against |
|---|---|---|---|
| `planner` | LLM agent | `claude-3-5-sonnet-latest` | — |
| `coder` | LLM agent | `claude-3-7-sonnet-latest` | — |
| `sandbox_executor` | Tool node | Docker subprocess | `sample_dir` (1–2 images) |
| `local_executor` | Tool node | Native subprocess | `input_dir` (full dataset) |

### Edges

| From | To | Type | Condition |
|---|---|---|---|
| START | `planner` | Entry point | Always |
| `planner` | `coder` | Direct | Always |
| `coder` | `sandbox_executor` | Direct | Always |
| `sandbox_executor` | `local_executor` | Conditional | `execution_success == True` |
| `sandbox_executor` | `coder` | Conditional | `execution_success == False` AND `retries < 3` |
| `sandbox_executor` | `local_executor` | Conditional | `execution_success == False` AND `retries >= 3` (budget exhausted — passes failure state through) |
| `local_executor` | END | Direct | Always |

### Retry mechanism

The retry counter (`state["retries"]`) lives in shared state and is incremented
exclusively by `sandbox_executor_node` on each failure.  `coder_node` reads
`state["error"]` on every invocation: if it is set, the full error text
(pip failure, runtime traceback, etc.) is injected into the prompt so the LLM
can self-correct.  `coder_node` also clears `error` to `None` on each fresh
attempt so stale errors never bleed into subsequent rounds.

### Two-stage execution rationale

| Stage | Why |
|---|---|
| **Sandbox** (Docker · `sample_dir`) | Validates code correctness and package availability cheaply on 1–2 images before touching the full dataset. Failures are safe and fast to retry. |
| **Local** (subprocess · `input_dir`) | Runs the proven script natively for maximum I/O throughput on the full dataset. Uses `validated_dependencies` already confirmed by the sandbox to skip redundant pip work. |

---

## File reference

```
Agent/
├── main.py
├── requirements.txt
├── core/
│   ├── state.py
│   └── schema.py
├── agents/
│   ├── planner.py
│   └── coder.py
├── tools/
│   └── executor.py
└── graph/
    └── builder.py
```

### `main.py`
Public entry point. Exposes `run_pipeline()` (blocking) and `run_pipeline_stream()`
(yields `(node_name, state_delta)` tuples for Napari UI progress updates).

Responsible for three setup tasks before invoking the graph:
1. Resolve the Anthropic API key from the argument or `ANTHROPIC_API_KEY` env-var.
2. **Auto-sample** — scan `input_dir` for image files, copy 1–2 at random into a
   `tempfile.mkdtemp()` directory, and pass its path as `state["sample_dir"]`.
   The temp directory is deleted in a `finally` block regardless of outcome.
3. Build the fully-populated initial `PipelineState` dict.

`output_dir` is optional; defaults to `<input_dir>_biovision_output` alongside
the input folder.

---

### `core/state.py` — `PipelineState`
The single TypedDict that flows through every node. All fields are primitives or
plain collections so LangGraph's `MemorySaver` checkpointer can serialise them
without custom codecs.

| Field group | Fields |
|---|---|
| Inputs | `metadata_yaml`, `input_dir`, `output_dir`, `sample_dir`, `api_key` |
| Planner outputs | `plan_title`, `plan_steps`, `plan_rationale` |
| Coder outputs | `generated_code`, `code_dependencies` |
| Executor outputs | `execution_stdout`, `execution_stderr`, `execution_success`, `validated_dependencies` |
| Control | `error`, `retries` |
| Trace | `messages` (append-only via `add_messages` reducer) |

`validated_dependencies` is populated by `sandbox_executor_node` on success and
reused by `local_executor_node` to avoid reinstalling packages that are already
confirmed to work.

---

### `core/schema.py` — Pydantic output schemas
Defines the structured types that LLM nodes return via `.with_structured_output()`.
Using these schemas guarantees that the LLM response is always a validated Python
object — no fragile text parsing.

| Schema | Produced by | Fields |
|---|---|---|
| `PreprocessingPlan` | `planner_node` | `title: str`, `steps: list[str]`, `rationale: str` |
| `GeneratedCode` | `coder_node` | `code: str`, `dependencies: list[str]` |

These schemas are never stored directly in `PipelineState`; node functions unpack
them into the flat scalar/list fields the state expects.

---

### `agents/planner.py` — `planner_node`
Reads `state["metadata_yaml"]` and calls `claude-3-5-sonnet-latest` with a
bioimage-science system prompt. Returns a `PreprocessingPlan` unpacked into
`plan_title`, `plan_steps`, and `plan_rationale`.

The LLM is constrained to produce only a plan — no code, no file I/O.

---

### `agents/coder.py` — `coder_node`
Reads `plan_title`, `plan_steps`, and `plan_rationale` from state and calls
`claude-3-7-sonnet-latest` to produce a complete, runnable Python script plus
its pip dependencies.

On retry iterations, `state["error"]` is non-`None` and its content is appended
to the prompt as a fenced code block so the LLM sees exactly what went wrong
(runtime traceback or pip failure) and can correct it. `error` is cleared to
`None` in the returned delta so it does not persist into later rounds.

---

### `tools/executor.py` — `exec_sandboxed()`
Low-level execution layer. Writes the generated code to a temp file, optionally
pip-installs dependencies, then runs the script via one of two backends:

**`_ensure_dependencies(packages) -> Optional[str]`**
Attempts `pip install` on the full list as-is — no import-name normalization,
no hardcoded package mappings. Returns `None` on success or an error string on
failure. A failed install short-circuits `exec_sandboxed` before the script is
even executed, returning `{"success": False, "stderr": <pip error>}` for the
graph to route back to the Coder.

**Subprocess backend** (`use_docker=False`)
Spawns a child process with a minimal environment: only `INPUT_DIR`, `OUTPUT_DIR`,
`PATH`, and `PYTHONPATH` are forwarded. No secrets can leak.

**Docker backend** (`use_docker=True`)
Runs inside a locked-down container: `--network none`, `--memory 2g`, `--cpus 2`,
input mounted read-only. Requires a pre-built `biovision-sandbox:latest` image.

Both backends respect a configurable `timeout` (default 300 s) and return a
uniform `{"success": bool, "stdout": str, "stderr": str}` dict.

---

### `graph/builder.py` — `build_graph()`
Assembles and compiles the `StateGraph`. Houses the two executor node functions
(`sandbox_executor_node`, `local_executor_node`) and the conditional routing
function `_route_after_sandbox`.

```python
graph = build_graph()                  # plain, no checkpointing
graph = build_graph(checkpointer=True) # attaches MemorySaver for HITL / pause-resume
```

With `checkpointer=True` the compiled graph supports `graph.get_state()`,
`graph.update_state()`, and `interrupt_before` / `interrupt_after` gates —
enabling human-in-the-loop review between any two nodes without modifying the
node functions themselves.
