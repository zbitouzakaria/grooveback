# Working agreements

Portable across projects. Nothing here is specific to one codebase — project knowledge lives in that project's ADRs
and its `operate-*` skill.

**This file is the versioned copy.** The active one is `~/.claude/CLAUDE.md`, which applies to every project. When
this workspace becomes a monorepo, this copy moves to the root and takes over. Keep the two in sync: a change here
is a `docs/update-claude-md` PR, and the merged version is copied back to `~/.claude/`.

## Where knowledge lives

- **`docs/decisions/`** — ADRs: findings, and the evidence and reasoning behind them. The single source of truth for
  *what we learned*. Read them before proposing a change of method; they record what was already tried and why it was
  rejected.
- **`.claude/skills/operate-<project>/`** — how to actually run a project: setup, entry points, boundaries. Written
  so a different project could pick this one up. Skill-formatted so it is invocable by name, and imported by that
  project's root `CLAUDE.md` with `@path` so it is also always in context.
- **`.claude/skills/<tool>/`** — cross-project procedures (renting GPUs, deploys).
- **`docs/memory/`** — mirror of the assistant's `~/.claude` memory. See *Memory mirror* below.

Do not restate an ADR's conclusion anywhere else. Two copies of a finding means one is stale, and the stale one is
the one that gets read.

## Commits

Subject line only. Add a body only when the change is genuinely not self-explanatory from the diff and the title —
a surprising constraint, a rejected alternative, a workaround whose reason is invisible in the code. Never write a
body that restates the diff. Reasoning belongs in ADRs.

One reviewable unit of work per commit. Not every five-line change, and not a batch of unrelated changes at the end.
A whole document rewrite is one commit. Commit as you go rather than batching. Keep pure moves separate from content
edits. Stage explicit paths, never `git add -A`, when other work is in flight in the same tree.

## Pull requests

Written for someone with no session context.

- Illustrate a bug with a concrete example — the input, the wrong output, the right one — not a paragraph of prose.
- Describe the end state, not the path. Refactorings, false starts, and problems found and solved along the way do
  not belong in an external-facing description.

## Long-running and remote commands

**Create a virtual environment even on remote machines.** System Python is often externally managed, and `pip` will
refuse silently — in quiet mode with no error at all. Use `--system-site-packages` to reuse a preinstalled torch
rather than downloading gigabytes of it again.

Anything that can hang on an outside effect — provisioning, network installs, external APIs — gets audited live:

1. Launch it in the background with output redirected to a local log file, so both of us can read it.
2. Launch a parallel timer. When it fires, read the log.
3. If progress matches prediction, re-arm. If not, stop and report.

Three rules make that work:

- **Write the progress criterion before launching**, as something mechanical: a file count that grows, a log line
  that appears, bytes that increase. "Looks sensible" is not checkable and invites reading silence as work.
- **Set a total budget, not just an interval.** A timer that re-arms forever is itself a hang.
- **Stage the intervals.** Check early (~2 min) to catch immediate failure, then widen. A fixed interval is usually
  wrong in both directions.

Never `tail -f` a finite job: it never reaches EOF, so no completion notification can ever fire. Watch the process,
not the log. Prefer a plain background `sleep` over spawning an agent as the timer — an agent is a second thing that
can die silently.

**Avoid loops in scripts.** Scripting tasks almost never need one, and a loop that spins on an unexpected input costs
hours before the real work starts. When a loop appears, look for the version without it.

## Memory mirror

`docs/memory/` is a copy of the assistant's `~/.claude` memory for this workspace, so it is versioned, reviewable and
survives the machine.

- The live `~/.claude` memory is authoritative **at runtime**.
- The repo copy is authoritative **on review**: when they diverge, the reviewed version wins and gets copied back.
- A memory change is mirrored into `docs/memory/` and raised as its own PR (`docs/update-claude-md`), never folded
  into an unrelated change.
- **Never commit secrets.** API keys and tokens are read from the environment; the mirror names the variable and
  where to get its value.
