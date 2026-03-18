# Coding Agent Roadmap

This roadmap is tailored to the current repository structure and codebase shape.
It is meant to answer two questions:

1. What capabilities should a mature coding agent have?
2. What is the most practical path from the current implementation to that target?

## Product Goal

Build a reliable CLI coding agent that can:

- understand multi-step software tasks
- inspect and edit real codebases safely
- execute tools, tests, and verification loops
- explain its changes and risks clearly
- operate with enough guardrails to be trusted in day-to-day development

## Current State

The repository already has a solid early foundation:

- `coding_agent/cli.py` provides `chat`, `run`, and `config` commands via Typer.
- `coding_agent/agent/core.py` implements a basic ReAct-style loop with tool execution.
- `coding_agent/llm/client.py` provides a unified LiteLLM-backed client with provider-aware tool formatting and debug logging.
- `coding_agent/tools/registry.py` provides tool registration plus OpenAI and Anthropic schema conversion.
- `coding_agent/tools/file_tools.py` includes workspace-bounded file inspection and editing helpers.
- `coding_agent/tools/shell_tools.py` includes a command allowlist and basic git helpers.
- `coding_agent/tools/code_tools.py` includes lightweight structure and symbol discovery with tree-sitter fallbacks.
- `tests/` covers CLI behavior, registry behavior, file tools, basic agent flow, and LLM logging.

That foundation is enough for a useful prototype, but it is not yet a mature coding agent. The biggest gaps today are:

- task planning and state management are minimal
- code editing is still coarse-grained and lacks a first-class patch workflow
- verification loops are manual and shallow
- safety and policy enforcement are basic rather than defense-in-depth
- code intelligence is search-oriented rather than symbol- and dependency-aware
- observability exists for LLM traffic but not yet for end-to-end task traces
- evaluation and regression measurement are missing

## Maturity Model

A mature coding agent should have the following capability layers:

1. Interaction
   Clear CLI flows, streaming output, resumable sessions, task status, user approvals, concise summaries.
2. Reasoning and Task Management
   Planning, subtask tracking, iteration budgets, failure recovery, memory of task decisions.
3. Codebase Understanding
   Fast search, symbol lookup, call graph awareness, dependency discovery, relevant-context selection.
4. Safe Execution
   Workspace boundaries, command policy, destructive-action controls, audit logs, secret protection.
5. Reliable Editing
   Patch-based edits, cross-file refactors, conflict handling, change previews, rollback support.
6. Verification
   Automated tests, lint, type checks, targeted reruns, self-repair loops, confidence scoring.
7. Extensibility and Operations
   MCP integrations, provider routing, metrics, benchmarks, CI automation, release discipline.

## Guiding Principles

- Keep the CLI simple even as the internals grow more capable.
- Preserve separation between CLI, agent orchestration, LLM integration, and tool implementations.
- Prefer async-first IO and avoid spreading `asyncio.run(...)` deeper into the architecture.
- Make safety explicit rather than implicit.
- Optimize for reviewable changes and incremental wins.
- Add tests and benchmarks alongside each capability upgrade.

## Phase 1: Foundation Hardening

Goal: make the existing prototype dependable for small real-world tasks.

### Scope

- Refactor `coding_agent/agent/core.py` to shrink the main loop into focused helpers.
- Introduce explicit message, tool-call, and tool-result typed models instead of loose dictionaries.
- Align project metadata and tooling with the repository's Python 3.14 baseline.
- Improve config ergonomics in `coding_agent/config.py` for logging, safety, and future feature flags.
- Expand unit tests around tool formatting, message history, and failure paths.

### Deliverables

- `AgentSession` or equivalent internal state object
- typed message conversion helpers for OpenAI-compatible and Anthropic-compatible flows
- clearer separation of `build messages`, `call model`, `execute tools`, and `render response`
- structured error types for LLM, tool, and validation failures
- better debug output around iterations and tool execution

### Success Criteria

- the core loop is easier to test without full CLI setup
- each tool invocation path has deterministic tests
- mypy coverage expands on agent and LLM modules
- no behavior regressions for `chat` and `run`

## Phase 2: Safe Editing and Verification Loop

Goal: move from "can answer and run tools" to "can complete coding tasks with verification."

### Scope

- Add a first-class patch editing tool instead of relying only on full-file writes.
- Add file diff and change preview tooling.
- Add a verification subsystem that can run targeted checks such as `pytest`, `ruff`, and `mypy`.
- Teach the agent loop to interpret failures and retry within a bounded repair budget.
- Improve shell safety with clearer command classes and a stricter policy model.

### Deliverables

- `apply_patch`-style file editing tool with workspace enforcement
- `view_diff` or `git diff` helper tool for change inspection
- verification runner abstraction with structured results
- automatic "edit -> verify -> repair" loop for `run`
- clearer user-facing summaries of what changed and what was verified

### Success Criteria

- the agent can implement a small bug fix and verify it automatically
- destructive edits are reduced in favor of minimal patches
- failed verification results are preserved in history for model reasoning
- shell command approval logic is easier to audit

## Phase 3: Better Code Intelligence

Goal: give the agent enough code understanding to handle multi-file work without bloating context.

### Scope

- Upgrade `coding_agent/tools/code_tools.py` from simple summaries to symbol-aware navigation.
- Add richer project search primitives: references, imports, call sites, and module relationships.
- Add context selection utilities that gather only relevant files and snippets.
- Cache file summaries and indexes when safe to reduce repeated work.

### Deliverables

- project index for files, symbols, and imports
- symbol reference search tool
- import/dependency summary tool
- related-files recommender for a prompt or symbol
- token-budget aware context pack builder in `coding_agent/llm/`

### Success Criteria

- the agent can identify which files matter for a feature or bug
- prompts contain higher-signal context and fewer entire-file dumps
- large-repository tasks are faster and cheaper
- code navigation quality is noticeably better than regex-only search

## Phase 4: Planning, Memory, and UX

Goal: make the agent feel collaborative rather than stateless.

### Scope

- Add an explicit planning layer in `coding_agent/agent/`.
- Track subtask progress, decisions, and verification outcomes across turns.
- Add resumable session state beyond shell history.
- Improve CLI rendering with clearer task progress, current step, and summarized tool outputs.

### Deliverables

- optional planning mode for complex tasks
- structured session state persisted under the config directory
- concise task summary after each major iteration
- user controls for max iterations, repair budget, and verbosity
- cancellation and resume support for long-running operations

### Success Criteria

- the agent can explain what it is doing and why
- long tasks can survive interruption and continue later
- repeated user clarifications do not require reconstructing all context
- CLI output remains readable under heavy tool usage

## Phase 5: Multi-Provider Reliability and Tool Ecosystem

Goal: make the platform robust across providers and ready for expansion.

### Scope

- Expand provider normalization in `coding_agent/llm/client.py`.
- Handle tool-calling differences, streaming differences, and error payload differences explicitly.
- Introduce a clean integration path for MCP servers and future tool packs.
- Add provider-specific retry, timeout, and fallback behavior.

### Deliverables

- provider adapter layer instead of provider checks scattered through the code
- normalized response model for text, tool calls, and finish reasons
- MCP discovery and registration flow
- request retry policy with backoff and error classification
- cost and latency instrumentation by provider and model

### Success Criteria

- model switches do not require behavior changes in agent core
- provider-specific failures are visible and recoverable
- new tools can be added without changing orchestration logic
- debugging a failed model call is straightforward from logs

## Phase 6: Production Readiness

Goal: support daily use by a team rather than occasional local experiments.

### Scope

- Add structured observability for tasks, tools, model calls, and verification runs.
- Build an evaluation suite with representative coding-agent tasks.
- Add regression tracking for success rate, latency, cost, and repair rate.
- Strengthen safety around secrets, ignored files, generated files, and repository policies.

### Deliverables

- structured task trace format
- benchmark task suite under `tests/` or a dedicated `evals/` directory
- golden-path scenarios for bug fix, refactor, explanation, and test generation
- policy checks for sensitive paths and file classes
- documentation for operating the agent in CI or shared environments

### Success Criteria

- every release can be evaluated against a fixed task set
- regressions in reliability or cost are caught early
- maintainers can reason about failures from logs instead of anecdotes
- the agent is safe enough for wider internal adoption

## Suggested Workstreams By Module

### `coding_agent/cli.py`

- add flags for verification behavior, verbosity, and resume
- improve task summaries and failure messaging
- keep command surface small and predictable

### `coding_agent/agent/`

- split orchestration from presentation
- introduce planning, state, and repair abstractions
- replace large monolithic methods with typed helpers

### `coding_agent/llm/`

- add provider adapters and normalized response models
- improve token budgeting and context packing
- track latency, retries, and finish reasons consistently

### `coding_agent/tools/`

- add patch/diff/verification tools
- harden shell policy and file safety
- expand symbol, reference, and dependency analysis

### `tests/`

- broaden unit coverage for provider formatting and failure paths
- add integration tests for multi-step tasks
- build benchmark-style eval cases over time

## Recommended Milestones

### Milestone A: Trusted Prototype

- complete Phase 1
- complete the patch tool from Phase 2
- add targeted verification hooks

Outcome:
The agent can safely make small code changes and explain what happened.

### Milestone B: Useful Daily Assistant

- complete the rest of Phase 2
- complete Phase 3
- land the planning pieces from Phase 4

Outcome:
The agent can solve routine bug fixes and small feature tasks in multi-file repositories.

### Milestone C: Team-Ready Agent

- complete Phase 4 through Phase 6
- establish benchmarks and release gates

Outcome:
The project is maintainable, measurable, and suitable for regular team usage.

## Near-Term Priority Queue

If only the next few iterations matter, prioritize work in this order:

1. Refactor `coding_agent/agent/core.py` into smaller typed units.
2. Add a real patch editing tool and diff inspection tool.
3. Add automated verification loops for tests, lint, and typing.
4. Improve shell and file safety policies.
5. Upgrade code intelligence beyond regex searches.
6. Add structured session state and planning.
7. Add evals, metrics, and release discipline.

## Definition of Done for a "Mature" v1

The project can be considered mature when it can reliably:

- inspect an unfamiliar repository and find relevant files
- propose and apply minimal code changes
- run targeted verification automatically
- recover from common failures without looping indefinitely
- explain changes, risks, and remaining gaps clearly
- operate with auditable safety controls and measurable quality
