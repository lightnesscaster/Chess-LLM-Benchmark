# Eubos anchor calibration (2026-09-30)

## Summary

The `eubos` anchor carries lichess @eubos's classical rating, but until
2026-09-30 it ran the local build unscaled at 15+10 on a much faster machine
than the bot. That setup plays **+200 Elo** above Eubos under the bot's
conditions (95% +48..+410).

- **From 2026-09-30T21:44:45Z:** `eubos` reproduces the bot's classical
  conditions and uses its current lichess classical rating, **2357**.
- **Earlier games:** each era keeps its lichess rating at the time, plus the
  measured gap:

  | Era | Before | After |
  |---|---|---|
  | before 2026-09-08T13:02:05Z | 2211 | **2411** |
  | 2026-09-08 → 2026-09-30T21:44:45Z | 2346 | **2546** |

Config: `config/benchmark.yaml` (`eubos`, `rating_history`). Commit `d32dca7`.

## Reference conditions (lichess bios and exports, 2026-09-30)

| Bot | Classical rating | Engine and stated setup |
|---|---|---|
| @eubos | 2357 ± 45 (3510 games) | Eubos v4.3, Raspberry Pi 5, 512 MB hash, ~500k nodes/s |
| @baby_eubos | 2075 ± 46 (1590 games) | Eubos v4.3, Pi Zero, 1 thread, 64 MB hash, ≤25k nodes/s |
| @Cheng-4 | 2510 ± 81 (2898 games) | Cheng 4.39, i7-8700, 1 thread, 1 GB hash, ~1.5M nodes/s |

`data/lichess_classical_time_controls.json` holds each bot's last 510 rated
classical games: time controls, thinking time per move and opponent mix. Each
emulated game draws its clock from that mix.

- **Opponent pools:** both Eubos bots play almost only bots. Cheng-4 plays bots
  and some humans. Maia's lichess ratings come mostly from humans.
- **Speed:** each engine's local nodes/s is probed at startup and its clock
  scaled by `reference_nps / local_nps`. On the M1 Pro this is about 0.28 for
  eubos, 0.013 for baby_eubos and 0.51 for Cheng.

## Runs (`data/*.jsonl`)

| Run | Match | Games | Score (A) | Implied rating of A |
|---|---|---|---|---|
| A | local eubos 15+10 vs eubos-lichess | 40 | 0.863 | 2676 (same-engine, inflated) |
| B | baby-lichess vs eubos-lichess | 30 | 0.067 | 1898 (expected 2075) |
| C | local eubos 15+10 vs cheng-lichess | 50 | 0.240 | 2310 |
| D | eubos-lichess vs cheng-lichess | 44 | 0.091 | 2110 (expected 2357) |
| ver_4.2 | Eubos 4.2 vs 4.5, both 60+0.6 | 50 | 0.160 | −288 vs 4.5 |
| ver_4.3 | Eubos 4.3 vs 4.5, both 60+0.6 | 50 | 0.260 | −182 vs 4.5 |

### Findings

1. **The local setup vs the bot.** C − D measures the gap through a different
   engine, so it has no same-engine distortion: **+200** (bootstrap 95%
   +48..+410). Run A, Eubos against Eubos, gives +319 raw. Scaled by run B's
   inflation (baby_eubos vs eubos: 458 head-to-head vs 281 by lichess rating),
   it comes to about +195, which agrees.
2. **Lichess bot ratings don't stay consistent between engines.** eubos scores
   0.09 against the Cheng emulation, when the rating gap predicts 0.29. On
   lichess itself, eubos is 0/105 against Cheng-4 across all time controls. So
   the absolute level depends on which bot anchors the scale. The anchor stays
   defined by @eubos's own rating, and Cheng is used only to measure the
   *relative* gap.
3. **Eubos really got stronger between eras.** v4.2 and v4.3 both lose clearly
   to v4.5 (above), which matches @eubos's lichess rating rising from about
   2211 to 2350 in 2026. So the eras keep separate base ratings, rather than
   one number for all past games.
4. **maia-1900 isn't a usable engine reference.** baby-lichess scored 99.5/100
   against it (prototype run, not saved), whereas on lichess baby_eubos scores
   0.84 against maia9 across all time controls.

### Caveats

- **Bios may be out of date.** The @eubos bio says v4.3, but its rating rose
  when v4.5 was released, so the bot may run v4.5. The emulation follows the
  bio.
- **Era 1 is inferred, not measured.** The era-1 build
  (`/Volumes/MainStorage/Programming/EubosChess`) isn't available. Its +200
  assumes the same speed advantage over the bot as era 2.
- **The Cheng anchor is uncertain.** Cheng-4's RD is 81 because it has been
  inactive since July 2026. Its rating was stable at about 2490–2520 for 18
  months.

## Reproducing

1. **Engines:** download Eubos `v4.2`, `v4.3` and `v4.5` releases into
   `data/engines_cache/eubos-<v>/Eubos.jar`, each with an `eubos.sh` launcher.
2. **Cheng 4.39:** the original `kmar/cheng4` repository is gone; use
   `github.com/phenri/cheng4` (commit `7bc5ddc`). For Apple Silicon, apply
   `cheng4-aarch64.patch`, which replaces x86 popcount/bitscan assembly with
   compiler builtins and doesn't change search behaviour (perft 5 = 4,865,609).
   Then build:

   ```bash
   clang++ -O3 -DNDEBUG -std=c++11 -fno-rtti -fno-exceptions -o data/engines_cache/cheng-4.39/cheng4 cheng4/allinone.cpp -lpthread
   ```

3. **Run matches** with `scripts/calibrate_engine_anchor.py`. Run at most one
   game per performance core, because spilling onto efficiency cores breaks the
   speed scaling.

   ```bash
   python scripts/calibrate_engine_anchor.py eubos-local-15+10 cheng-lichess 10 41 runC.jsonl
   python scripts/calibrate_engine_anchor.py eubos-lichess cheng-lichess 15 51 runD.jsonl
   python scripts/calibrate_engine_anchor.py --summary runC.jsonl 2510
   ```

4. **Refresh the time controls:** they come from
   `GET https://lichess.org/api/games/user/<bot>?perfType=classical&max=500&clocks=true`,
   which now requires a lichess token. The calibration runs used fixed speed
   scales measured beforehand (0.277 eubos, 0.0134 baby_eubos, 0.512 Cheng).
   The script and production probe speed at startup instead, which gave
   0.279–0.283 for eubos.
