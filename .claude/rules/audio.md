# Audio comparisons

These decide whether a result is real. Breaking any of them produces a confident wrong conclusion rather than an
error.

- **Level-match to −14 LUFS before any comparison**, with one common headroom gain across the set. Loudness
  dominates informal comparison; without this you are measuring gain.
- **Energy is not perception.** Band deltas catch regressions. Blinded listening on monitoring hardware decides, and
  it has already overturned an energy-based reading once.

## Correctness gates

Audio breaks silently: no exception, nothing looks wrong, the output is merely worse. Tests target that specifically.

- an identity model through the full inference path must return the input — catches chunking and overlap-add bugs
- every evaluation run also scores the untouched input and the clean reference; if the reference does not win, the
  harness is broken rather than the model
- loudness is asserted before any comparison is scored
- property tests on invariants: sample rate and channel count preserved, no NaN, no clipping introduced

An identity model is coherent by construction, so none of these catch a model that invents content *differently on
each call* — the case where two windows produce the same band at different phase and blending them loses power.
Crossfaded chunking is safe only for models whose output is consistent given their input — measured true of Apollo,
whose chunking and its tests live in the fork. No test covers an inconsistent model; add one before chunking
anything stochastic.

## Findings live in ADRs

`docs/decisions/` holds what we have learned and the evidence for it. Read before proposing a change of method —
they record what was already tried and why it was rejected.

Avoid duplicating an ADR's conclusion where it is not doing work — a copy that drifts is worse than a
pointer. Restating one in a rule or a memory file is fine when it is operationally useful; cite the ADR so
the source of truth stays obvious.

| ADR | |
|---|---|
| 0001 | Record architecture decisions — why this directory exists |
| 0002 | Plan code architecture — module layout, testing, and what is deliberately not built yet |
| 0003 | Run Apollo on target distribution — superseded by 0005 |
| 0004 | **Restoration with a generative prior** — the thesis: the prior is the artifact, the damage is handled at inference, output is plausible rather than faithful |
| 0005 | Establish baselines and prior viability — what had to be measured before any method work, and the stopping point |
| 0006 | **Baseline findings: Apollo and A2SB** — Apollo restores across the whole band and wins by ear; A2SB only fills above a cutoff and is heavy |
| 0007 | Benchmark codec restoration — MP3 twins of clean chunks, five metrics against the master |
| 0008 | SA3 generation probe — the codec as instrument; no checkpoint is master-grade on both metric families |
