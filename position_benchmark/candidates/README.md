# Candidate Pools

These files are source pools used during panel selection. They are not production
benchmark inputs:

- `equal_100_depth16.json`: older 100-position equal candidate pool.
- `blunder_1000_depth16.json`: extracted 1,000-position blunder candidate pool.
- `failure_transfer_screen_v1/`: frozen leave-one-base-model-out screen containing
  12 saved GPT-5.6 failure states, 12 matched legal controls, and three target
  files. Source and target base-model lines never overlap.
- `failure_transfer_positive_3.json`: three GPT-5.6 transfer-positive failure
  states plus matched controls. This shortlist was selected using GPT-5.6 and
  requires validation on unrelated model families.

Selected production and optional panels live in `../panels/`.
