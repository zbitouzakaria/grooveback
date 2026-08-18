---
name: audit-long-running-commands
description: "Long or remote commands get a re-arming timer, a written progress criterion, and a total budget"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a40108c-c6aa-4b86-b4a1-81c77b27918c
  modified: 2026-08-18T11:54:35.085Z
---

Remote and long-running commands must be auditable live, by both of us. Run them so their output lands in a local
file, and pair anything that could hang on an outside effect with a timer that calls you back.

The loop (Zakaria, 2026-08-18):

1. Launch the task in the background, output redirected to a log file.
2. Launch a parallel timer in the background (`sleep N`), which re-invokes you when it fires.
3. On wake, read the log. If progress matches what you predicted, re-arm the timer. If not, stop and report.

**Why:** two hangs in one session. `runpodctl pod create --wait` blocked 10 minutes on a pod whose SSH port was never
published — it could never succeed. A `uv sync` over SSH wrote its output nowhere, so neither of us could see it was
stuck behind PEP 668. Silence is indistinguishable from progress unless you decide in advance what progress looks
like.

**How to apply:**

- **Write the progress criterion before launching**, as something mechanical: a file count that should grow, a log
  line that should appear, bytes that should increase. "Looks sensible" is not checkable and invites rationalising
  silence as work. This is the part that actually prevents the failure.
- **Set a total budget, not just an interval.** A timer that re-arms forever is itself a hang. When the budget is
  spent, kill the job and reconsider — a 15-minute cap on that `uv sync` would have saved 35 minutes.
- **Stage the intervals.** Fixed 5 minutes is usually wrong in both directions. Check at ~2 minutes first (catches
  immediate failure, wrong flag, silent refusal), then widen to 10–15. Tune to what is actually being waited on.
- **Prefer a plain background `sleep` over spawning an agent** for the timer. An agent is a second thing that can die
  silently; a backgrounded shell command already re-invokes you when it exits.
- **Only for jobs that can hang on something outside the process** — remote calls, provisioning, network installs,
  external APIs. A local deterministic script does not need it.
- **Avoid loops in scripts.** A six-hour run this session was a directory walk that hit `/` and spun forever, before
  the real work started. Scripting tasks almost never need a loop; when one appears, look for the version without it.
- For remote work specifically: capture with `ssh host 'cmd 2>&1' | tee local.log` so the output exists on this side,
  and prefer detaching the remote job and polling over holding an SSH connection open.

Related: [[never-use-tail]] is the same lesson for watchers that can never reach EOF.
