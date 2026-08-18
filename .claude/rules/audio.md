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
`test_invented_content_survives_the_seams` covers it with a fixed-amplitude tone at random phase per call, asserting
band power stays flat across seams. Extend that pattern before chunking anything else generative.

## Findings live in ADRs

`docs/decisions/` holds what we have learned and the evidence for it. Read before proposing a change of method —
they record what was already tried and why it was rejected. Do not restate their conclusions anywhere else; two
copies means one goes stale, and the stale one is the one that gets read.

| ADR | |
|---|---|
| 0001 | Record architecture decisions — why this directory exists |
| 0002 | Plan code architecture — module layout, testing, and what is deliberately not built yet |
| 0003 | Run Apollo on target distribution — superseded by 0005 |
| 0004 | **Restoration with a generative prior** — the thesis: the prior is the artifact, the damage is handled at inference, output is plausible rather than faithful |
| 0005 | Establish baselines and prior viability — what had to be measured before any method work, and the stopping point |
| 0006 | **Baseline findings: Apollo and A2SB** — Apollo restores across the whole band and wins by ear; A2SB only fills above a cutoff and is heavy |
