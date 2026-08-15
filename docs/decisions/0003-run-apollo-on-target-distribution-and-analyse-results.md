# 3. Run Apollo on target distribution and analyse results

Date: 2026-04-22

## Status

Superseded by [5. Establish baselines and prior viability on real library material](0005-baselines-and-prior-viability-on-real-library-material.md)

The instinct here — measure before deciding — is kept. What changed is scope: ADR-0004 replaced the supervised design
with a prior-first one, so Apollo is now a standing baseline rather than a reference point on the way to training a
similar model, a second free baseline (A2SB) has since appeared, and the autoencoder and pretrained prior also need
smoke-testing before any budget is committed.

## Context
 
Apollo (Li & Luo, ICASSP 2025) is a published open-source model with a pretrained checkpoint targeting MP3-compressed music restoration — the closest existing work to grooveback's problem.
 
Apollo is trained on MUSDB18-HQ and MoisesDB: clean professional multitrack recordings, MP3-compressed at various bitrates. grooveback's target distribution is different: vinyl rips re-encoded by YouTube, on a narrow genre. Whether Apollo generalizes usefully across that gap cannot be predicted from the paper alone.
 
Without running Apollo on grooveback's actual distribution, any architectural decision rests on assumption rather than evidence.
 
## Decision
 
Before writing or training any grooveback model, run Apollo's pretrained checkpoint on a small curated set of grooveback-distribution tracks and document the findings. The results inform a subsequent ADR on model architecture.
 
## Consequences
 
- Architectural decisions become empirical rather than speculative.
- Minor delay before model work begins.

## Revisit triggers
 
- Apollo proves impractical or hard  to run
 
