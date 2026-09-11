# 12. AudioSR as a bandwidth-extension baseline

Date: 2026-09-10

## Status

Proposed — rendered and scored; the listening verdict is pending

## Context

AudioSR ([Liu et al., arXiv:2309.07314](https://arxiv.org/abs/2309.07314))
performs diffusion super-resolution in the mel domain with a vocoder, taking
any input bandwidth between 2 and 16 kHz to a 48 kHz output. It is the last
open-weights restoration method in the README survey without a "Tested here"
mark, and the second bandwidth-extension model on the board after A2SB
(ADR-0006), which fills the removed band conservatively. Whether a mel-domain
diffusion model invents a more convincing top end than a Schrödinger bridge
is a question the benchmark can answer directly.

Two properties of the paper's inference shape the expectation. First, the
paper prescribes no sampling values and ablates none — and the release
ships two defaults: the CLI passes 50 ddim steps, while the API function
defaults to 200. The benchmark pins the CLI's 50, the configuration
upstream's own tool gives its users, after measuring the two against each
other (the fidelity check below) and preferring the tamer fill by ear. Second, inference replaces the spectrogram below the detected
0.99 roll-off frequency with the input's own content, in both the latent
output and the vocoder output — the kept band should therefore track the
degraded input, and the columns that can differ are the fill band and the
listening verdict. ADR-0007's decomposition bounds what any fill is worth on
the waveform metrics (at most ~0.5 dB); the phase-blind fill columns and the
ear are where this method can distinguish itself.

## Decision

### Environment

The pip release `audiosr==0.0.7` pins torch 2.0.1 and transformers 4.30.2,
which conflict with the project-wide torch 2.7.1 pin (ADR-0007). It
therefore runs in its own virtual environment behind a subprocess — the
A2SB arrangement — but without a fork: the package needs no code changes,
so `scripts/audiosr_setup.sh` builds `third_party/audiosr/.venv` from pip
with the install-time pins that keep a 2023 release installable (`numpy<2`,
`huggingface_hub<0.26`). Should an upstream code change ever become
unavoidable, the recorded escape hatch is a fork whose git URL replaces the
pip source in the setup script.

### The driver

`scripts/audiosr_restore.py` is ours and lives in this repository, so its
tests run in CI. It accepts any sample rate and returns the input's rate,
length and channel count; the model's native format (48 kHz mono) is private
to the driver, which resamples once with soxr VHQ before chunking, restores
each channel separately, and resamples back. The release has no long-audio
path, so the driver chunks: 10.24 s chunks (two of the model's 5.12 s
grains) cut on input boundaries, each render trimmed to its chunk before the
model's padding can accumulate, consecutive chunks joined by a 0.1 s linear
crossfade. The fade is equal-gain rather than equal-power so that identical
content passes through exactly — the identity gate of ADR-0002 — and its
cost on independently invented content is measured below rather than
assumed, per the chunking rule in `.claude/rules/audio.md`.

Sampling knobs left unset are not passed to the model at all, so the pinned
package's own defaults apply. The one knob the benchmark sets is
`ddim_steps=50` — upstream's CLI default, preferred by ear over the API's
200 (Measurements below) — in `configs/model/audiosr.yaml` behind the
Hydra entry point (`model=audiosr`), which also gains `model=apollo` so
every waveform baseline is launchable the same way.

One driver step follows from the package rather than from us: 0.0.7
peak-normalizes both its input and its output to 0.5, discarding level in
each direction per call, so raw chunk renders would join at unrelated
gains. Each render is therefore scaled back to its chunk's level by a
least-squares gain fit against the chunk — exact up to the invented band,
because the render's kept band is the input's own content — before the
crossfade. The fit is exactly 1 on identical content, so the identity gate
is unaffected; a dedicated test pins the normalization case.

### The benchmark

`audiosr` joins `apollo` and `a2sb` as an anchor in `scripts/run_xp.py`,
with no wall at the codec edge: A2SB is walled because its own cutoff
detection provably misreads sharp synthetic edges (ADR-0007); AudioSR runs
as released, and what its roll-off detection does with a LAME edge is a
finding of this experiment. Concurrently the matrix narrows to the ADR-0011
kept set — sources codec and aerofunk at 32/64/128 kbps, εar-VAE arms and
the three anchors — and the losing ADR-0011 renders are deleted from
`artifacts/xp`; they remain reproducible from that ADR's configuration, and
its tables remain the record. The local comparison page `demo/xp` (formerly
`demo/roundtrip`) carries the kept set plus AudioSR in the published
solo-key order.

## Measurements before trust

**Recorded from the installed 0.0.7 source (2026-09-10):**

- The API defaults are `seed=42, ddim_steps=200, guidance_scale=3.5`
  (`super_resolution`'s signature), while the shipped CLI passes
  `ddim_steps=50` explicitly. The first benchmark pass ran the API's 200;
  after the fidelity check below measured that arm inventing 5–14 dB more
  top-band energy than the CLI's 50, and monitors preferred the tamer
  fill, the benchmark pins `ddim_steps=50` and leaves the other knobs
  unset.
- The paper's low-band replacement is present, twice: the generated mel's
  low band is replaced with the input's (`mel_replace_ops`), and the
  vocoder output is post-processed against the lowpassed input waveform
  (`postprocessing` in `generate_batch`).
- Nothing clips: the loader peak-normalizes input to 0.5 (also dropping all
  channels but the first — moot here, the driver feeds mono chunks) and
  `generate_batch` peak-normalizes its output to 0.5. The anticipated
  pre-gain contingency is unnecessary; the per-chunk gain fit above is the
  consequence instead.

**Recorded from the renders (L4/A100 runs, 2026-09-10/11):**

- **Seam consistency** (codec 64 kbps twin, whole versus `--chunk-seconds
  2.56`, same seed, at the standing 50 steps): between 11 and 16 kHz the
  two renders agree within 1.1–6.3 dB per 1 kHz band; above 16 kHz the
  chunked render carries +26 to +43 dB more energy, in content the whole
  window leaves near silence (−68 dBFS and below). The chunked arm is the
  louder one everywhere, which rules out crossfade power loss (a fade can
  only lose power): the divergence is that this model's fill depends
  strongly on window length, so chunk length is quietly a content knob —
  the benchmark holds it fixed at 10.24 s everywhere. The same gate at 200
  steps showed the same one-sided pattern (+8 to +14 dB above 16 kHz).
  The fade is kept.
- **Alignment**: `best_lag == 0` for both renders — no vocoder frame shift.
- **Level**: the gain fit anchors the kept band exactly — the render's band
  energies below 8 kHz match the twin's within 0.3 dB. Integrated loudness
  still reads ~3 LU above the twin on the codec source because the invented
  top band adds K-weighted energy; that is the method's content, recorded
  rather than corrected (correcting whole-file loudness would deflate the
  kept band).
- **The crossover, measured**: on the codec 64 kbps twin (true edge 11 kHz)
  the render exceeds the twin by +4.5 dB in the 8–11 kHz band and matches it
  below — AudioSR's roll-off detection placed the replacement crossover
  near 8 kHz and re-synthesized the twin's real 8–11 kHz content. The sharp
  LAME edge misleads its detector, the same failure A2SB's detector shows
  (ADR-0007); here it is recorded as the shipped method's behavior.
- **Driver fidelity, verified against upstream directly** (2026-09-11,
  after the listening session raised the question): the codec 64 kbps twin
  rendered through upstream's own one-call path — its torchaudio resample,
  channel 0 of the stereo file, output saved untouched at its peak-0.5
  convention — differs from the driver's left channel by 1.2 dB LSD, with
  band energies agreeing within 0.2 dB in every band and identical scores
  against the master (LSD 24.9 both). Up to one gain scalar, the driver's
  output *is* upstream's. The same call at the CLI's 50 steps lands 9.9 dB
  LSD from the 200-step arm: the API default invents 5–14 dB more energy
  above 11 kHz than the CLI default — sampling steps change how much this
  model fills, not only how well.
- **Cost**: ~4 min per 10.24 s chunk-call at 200 steps on an L4
  (a 6 s stereo pack ≈ 8 min); ~30 s per call on an A100 SXM (a 180 s
  stereo pack, 36 calls ≈ 18 min). The full grid plus smokes and seam
  renders came to ≈ $1.80 of rented GPU.

## Results (2026-09-11 run, `ddim_steps=50`)

Each table is one source at one bitrate; columns are the five ADR-0007
metrics against the master, LSD lower-is-better. The fill-band decomposition
lives in `results.json`. The 200-step first pass is in this file's history;
against it, 50 steps gains up to +3 dB BSS-SDR (codec at 64 kbps), sits
within ~1 dB elsewhere, and its audible difference is a far tamer invented
top — which is why it stands.

**codec @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 12.1 | 10.2 | 9.8 | 11.3 | 41.5 |
| earvae | 7.9 | 7.1 | 6.3 | 9.4 | 40.8 |
| apollo-earvae | 4.7 | 4.8 | 3.7 | 9.9 | 9.4 |
| earvae-sub | 7.1 | 6.7 | 5.7 | 9.9 | 16.5 |
| apollo | 5.8 | 6.0 | 5.0 | 12.3 | 8.6 |
| a2sb | 11.5 | 9.9 | 9.4 | 11.8 | 35.0 |
| audiosr | 3.2 | 3.2 | 2.3 | 4.5 | 35.6 |

**codec @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 14.2 | 12.6 | 12.5 | 14.0 | 33.1 |
| earvae | 7.9 | 7.2 | 6.4 | 10.3 | 32.6 |
| apollo-earvae | 5.0 | 5.0 | 4.0 | 10.1 | 9.4 |
| earvae-sub | 7.7 | 7.0 | 6.1 | 10.7 | 16.2 |
| apollo | 7.1 | 7.2 | 6.5 | 13.7 | 8.2 |
| a2sb | 13.9 | 12.4 | 12.2 | 14.0 | 29.9 |
| audiosr | 7.3 | 7.2 | 6.6 | 8.5 | 25.0 |

**codec @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 17.0 | 16.3 | 16.5 | 18.6 | 20.5 |
| earvae | 7.8 | 7.1 | 6.3 | 11.2 | 21.2 |
| apollo-earvae | 5.2 | 5.2 | 4.2 | 10.2 | 9.0 |
| earvae-sub | 7.8 | 7.0 | 6.3 | 11.3 | 9.7 |
| apollo | 8.6 | 8.9 | 8.4 | 15.3 | 6.8 |
| a2sb | 16.5 | 15.9 | 16.1 | 18.6 | 14.6 |
| audiosr | 13.2 | 12.6 | 12.3 | 15.1 | 15.9 |

**aerofunk @ 32k** (edge 5.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 19.0 | 14.8 | 14.9 | 15.4 | 23.6 |
| earvae | 16.8 | 13.6 | 13.6 | 14.8 | 23.2 |
| apollo-earvae | 15.7 | 13.2 | 12.9 | 15.8 | 11.2 |
| earvae-sub | 15.3 | 12.8 | 12.7 | 15.2 | 13.1 |
| apollo | 16.8 | 14.0 | 13.9 | 16.8 | 10.4 |
| a2sb | 18.3 | 14.5 | 14.6 | 15.7 | 18.6 |
| audiosr | 7.7 | 7.6 | 7.1 | 8.6 | 18.6 |

**aerofunk @ 64k** (edge 11 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 18.6 | 17.0 | 17.3 | 18.6 | 16.7 |
| earvae | 15.3 | 13.5 | 13.3 | 16.4 | 16.6 |
| apollo-earvae | 14.7 | 13.2 | 12.9 | 17.2 | 10.0 |
| earvae-sub | 14.7 | 13.1 | 12.9 | 16.8 | 9.2 |
| apollo | 16.9 | 15.9 | 15.8 | 20.3 | 8.9 |
| a2sb | 18.1 | 16.6 | 16.9 | 18.4 | 14.0 |
| audiosr | 10.7 | 10.7 | 10.3 | 11.6 | 14.8 |

**aerofunk @ 128k** (edge 16.5 kHz)

| | BSS-SDR | SDR | SI-SNR | Spectral SNR | LSD ↓ |
|---|---|---|---|---|---|
| degraded input | 23.1 | 21.4 | 23.0 | 23.2 | 7.3 |
| earvae | 15.2 | 13.7 | 13.5 | 17.7 | 9.6 |
| apollo-earvae | 14.8 | 13.4 | 13.2 | 17.6 | 9.1 |
| earvae-sub | 15.2 | 13.7 | 13.5 | 17.7 | 8.2 |
| apollo | 20.2 | 19.9 | 20.0 | 24.3 | 6.7 |
| a2sb | 22.5 | 20.9 | 22.4 | 22.4 | 7.0 |
| audiosr | 20.5 | 19.5 | 20.2 | 21.4 | 9.0 |

What the metrics say, ahead of listening:

- The ADR-0007 pattern holds: audiosr never approaches the untouched input
  on the waveform metrics, sitting at or near the bottom of the board on
  the heavier-damage packs (BSS-SDR 3.2 on codec at 32 kbps against the
  input's 12.1) and closing only at 128 kbps.
- The two fill views split, each punishing the opposite sin: audiosr's fill
  LSD beats the input's silence in every pack and a2sb's in four of six —
  it fills more than the Schrödinger bridge — while its fill spectral SNR
  is negative in every pack (−8 to −17 dB): the filled band carries the
  wrong energy sample for sample. Apollo dominates both fill views
  everywhere.
- The kept band pays for the misplaced crossover at 32 and 64 kbps
  (spectral SNR 4.5 and 8.5 against the input's 11.3 and 14.0 on the codec
  source).

## Listening verdict

Pending: the monitor session against apollo, a2sb and the εar-VAE arms, on
the `demo/xp` page.

## Consequences

- Every open-weights method in the survey that this benchmark can run is now
  either scored on it or excluded on record; HP-codecX (fixed 16→48 kHz,
  applicable only at 32 kbps) is the remaining open release, deferred.
- The benchmark's standing matrix is the kept set; the full autoencoder
  comparison lives in ADR-0011.

## Revisit triggers

- An upstream code change becomes unavoidable → fork, and point the venv
  install at the fork.
- Sampling settings become a question → sweep through the Hydra entry point
  (`cli.run -m model=audiosr model.ddim_steps=…`), the ADR-0009 pattern.
- AudioSR releases a stereo or 44.1 kHz-native checkpoint.
