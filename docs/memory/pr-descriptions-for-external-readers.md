---
name: pr-descriptions-for-external-readers
description: "PR bodies are for outside readers — illustrate bugs concretely, omit in-session iteration"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d5c93125-9700-44bb-967a-f3423d0ebd90
  modified: 2026-08-15T18:04:25.632Z
---

Two rules for PR descriptions:

1. **Illustrate a bug with a concrete example**, not just prose. Show the array, the values, the before/after —
   something a reader grasps in two seconds rather than parsing a paragraph.
2. **Leave out work that only happened during the session.** Refactorings, false starts, and problems that were found
   and solved along the way do not belong in an external-facing description. If it was solved, the reader does not
   need it.

**Why:** The PR is read by someone with no session context. Anything that only makes sense as narrative of how the
work unfolded is noise to them, and a bug explained in prose is slower to grasp than the same bug shown.

**How to apply:** Describe the end state, not the path. For each bug worth mentioning, add a minimal literal example
(input array → wrong output → right output). Keep it in the same register as [[minimal-commit-messages]] — the diff and
the ADRs carry the detail.
