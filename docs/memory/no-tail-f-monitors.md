---
name: never-use-tail
description: tail -f banned for finite jobs (no EOF, no notification); bounded tail -N fine; avoid launch-time pipeline filtering
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-17T13:04:01.212Z
---

**The rule (amended by Zakaria, 2026-08-17): bounded `tail -N` is fine; `tail -f` is banned for watching finite
jobs** — it never reaches EOF, never exits, and no completion notification can fire. If `tail -f` is ever needed, a
second process must watch for the first one ending.

**Why:** Three failure modes hit in one weekend: (1) shell pipelines return the LAST command's status, so
`cmd | grep | tail` hides cmd's failure unless `set -o pipefail` is on (tail itself exits >0 on its own errors, per
the man page — the masking is pipeline semantics, not a tail defect); (2) `tail` in a live pipeline emits
nothing until EOF, so tasks show "no output" for entire multi-hour runs; (3) `tail -f` never exits: no EOF, no process exit, **no completion notification ever fires** — the agent waits
forever on an event that structurally cannot arrive, while a phantom "running" task sits in the Claude Desktop GUI
reading as a heavy job competing on a 16 GB machine.

**How to apply:** Launch background commands with raw output to a log file, nothing after the command. Read results at
read time from the file: the Read tool, `grep`, `sed -n '$p'`, or `awk 'END{...}'`. To watch a finite job, watch the
process: `while true; do kill -0 <pid> || { <print result>; break; }; sleep 60; done` — the watcher ends when the job
ends.
