# Decide before building

Every new piece of work starts with an ADR (Any Decision Record) in `docs/decisions/`, in a PR of its own: the
problem, what we looked at, what we want to build, and why. A new ADR is proposed by default. Write no code for it
until Zakaria accepts it; the implementation then follows in its own PR.

An ADR's tests count as passed only when two things hold: the code passes them, and smartly mutating the code makes
them fail. For each test, put in the bug its name guards against and watch it fail.
