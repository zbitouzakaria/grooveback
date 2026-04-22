# 3. Run Apollo on target distribution and analyse results

Date: 2026-04-22

## Status

Accepted

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
 
