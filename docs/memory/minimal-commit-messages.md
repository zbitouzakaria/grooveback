---
name: minimal-commit-messages
description: "Zakaria wants commit messages kept to the subject line, with a body only when genuinely necessary"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-15T14:22:26.117Z
---

Keep commit messages minimal: a conventional-commit subject line and nothing else. Add a body only when the change is
genuinely not self-explanatory from the diff and the title.

**Why:** Long explanatory bodies under every commit are noise to read back through. The reasoning that matters belongs
in ADRs, which this project keeps in `docs/decisions/` — duplicating it in git history makes both harder to maintain.

**How to apply:** Default to subject line only. Reserve the body for non-obvious *why* that has no other home — a
surprising constraint, a rejected alternative, a workaround whose reason is invisible in the code. Never write a body
that restates what the diff already shows. Still applies to atomic-commit discipline: one reviewable unit of work per
commit (see [[atomic-commits-per-unit-of-work]]).
