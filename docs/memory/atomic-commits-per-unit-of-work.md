---
name: atomic-commits-per-unit-of-work
description: "Zakaria wants each commit to be one reviewable unit of work, not incremental dribbles"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-15T14:22:32.885Z
---

Commit whenever a piece of work is finished and forms its own reviewable chunk. Not every five-line change, and not a
batch of unrelated changes at the end either. A whole README rewrite is one commit.

**Why:** Commits are review units. Splitting one logical change across many makes it unreviewable; merging several
unrelated ones does the same.

**How to apply:** Commit as you go rather than batching at the end. Keep pure moves separate from content edits. Keep a
status change and the document that causes it together when they are one logical change. Stage explicit paths, never
`git add -A`, when other work is in flight in the same tree. Pair with [[minimal-commit-messages]].
