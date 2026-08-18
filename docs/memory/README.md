# Memory mirror

A copy of the assistant's `~/.claude` memory for this workspace, so it is versioned, reviewable, and survives the
machine. The live memory is authoritative at runtime; this copy is authoritative on review. Changes arrive as their
own PR — see *Memory mirror* in `CLAUDE.md`.

Secrets are never mirrored: files name the environment variable and where to get its value.

`docs/CLAUDE.md` is mirrored the same way, from `~/.claude/CLAUDE.md`.

- [Minimal commit messages](minimal-commit-messages.md) — subject line only unless a body is genuinely needed; reasoning lives in ADRs
- [Atomic commits per unit of work](atomic-commits-per-unit-of-work.md) — one reviewable chunk per commit, committed as you go
- [PR descriptions for external readers](pr-descriptions-for-external-readers.md) — show bugs with concrete examples, drop in-session iteration
- [Apollo beats A2SB](apollo-beats-a2sb.md) — listening verdict, and the A2SB gotchas needed to run it at all
- [Apollo crackle on AN-2](apollo-crackle-on-an2.md) — RESOLVED: discard-mode chunking fixed it, confirmed by ear
- [No tail -f on finite jobs](no-tail-f-monitors.md) — tail -N fine; tail -f never notifies; filter at read time, pipefail if filtering at launch
- [RunPod CLI](runpod-cli.md) — how to run GPU experiments on RunPod: auth, pods, the $1/mo volume, house rules
- [SAME decoder invents HF](same-decoder-invents-hf.md) — ~30 dB hallucinated top end on rips; round-trip is the honest reference
- [Transport vector floor](transport-vector-floor.md) — cheap brightener, not restoration; the vinyl reference corrected the earlier read
- [Audit long-running commands](audit-long-running-commands.md) — re-arming timer, written progress criterion, total budget
- [RunPod PEP 668 venv](runpod-pep668-venv.md) — pip silently refuses on pod images; --system-site-packages venv instead
