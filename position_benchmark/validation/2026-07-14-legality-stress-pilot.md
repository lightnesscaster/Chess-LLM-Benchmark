# Legality-stress discriminator pilot — 2026-07-14

This pilot reran the frozen six-position legality-stress candidate on three
GPT-5.6 configurations. The panel was selected without GPT-5.6 evidence, and the
pilot used the exact production illegal-move retry. It has no production rating
effect.

## Results

| Configuration | Initial illegals | Retry recoveries | Prior illegals on the same six | Live-game first-attempt illegal rate |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.6 Luna medium | 1/6 | 1/1 | 1/6 | 14/207 (6.76%) |
| GPT-5.6 Terra low | 1/6 | 1/1 | 1/6 | 13/246 (5.28%) |
| GPT-5.6 Sol high | 0/6 | n/a | 0/6 | 3/311 (0.96%) |
| **Total** | **2/18 (11.11%)** | **2/2** | **2/18 (11.11%)** | **30/764 (3.93%)** |

The pilot exactly reproduced the earlier aggregate count and per-model split on
these six positions. Its 11.11% first-attempt illegal rate was 2.83 times the
three models' aggregate live-game rate and 4.17 times their 4/150 full-core rate.
The sample is intentionally small, so these ratios describe this pilot rather
than an estimated universal multiplier.

## Repeatability

The aggregate repeat did not come from deterministic trap rows:

- only 7 of 18 first moves exactly matched the earlier core answer;
- Luna medium's new illegal was on `core-equal-044`, while its earlier illegal
  on this subset was `core-equal-024`;
- Terra low's new illegal was on `core-equal-036`, while its earlier illegal was
  `core-equal-019`; and
- Sol high was legal on all six in both runs.

Neither of the two earlier illegal model-position pairs failed again. The panel
therefore appears useful as a compact model-level stress sample, not as a list of
positions on which a particular model will reliably fail. Repeated observations
should be aggregated at model/configuration level; a single row should not be
treated as a stable model characteristic.

## Retry interpretation

Both new initial illegals recovered under `production-game-retry-v1`. This adds
two recovery observations but no failed retries. It is consistent with the
refreshed live-game audit, where 64 of 66 eligible retries recovered, but is far too little
evidence to estimate a configuration-specific conditional retry-failure rate.
The main measurable problem remains first-attempt illegality and later second
strikes, not same-turn retry failure.

## Decision

Keep `candidates/legality_stress_6.json` frozen as a research panel. It efficiently
enriches first-attempt illegal events and can exercise the real retry path, but it
must not replace the 50-position core or contribute CPL/rating evidence. Because
the two retry opportunities both recovered and the live games already provide 56
retry outcomes, expanding this panel solely to estimate retry failure has low
expected value. The next high-value validation evidence is more independent game
outcomes, followed by cross-family evaluation when model-call budget permits.

That independent-game follow-up was subsequently completed for Luna medium and
Terra low; see `validation/2026-07-14-gpt56-targeted-games.md`.
