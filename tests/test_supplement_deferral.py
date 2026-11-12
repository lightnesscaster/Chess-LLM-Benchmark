"""Regression checks for screening a core with independently measured sibling panels."""

from dataclasses import replace
import unittest

from position_benchmark import acquisition
from rating.rating_store import benchmark_predictions_from_inputs, load_position_benchmark_inputs


class SupplementDeferralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all = load_position_benchmark_inputs(False)
        cls.policy = acquisition.load_acquisition_policy()

    def without_supplements(self, target, *sources):
        players = (*sources, target)
        return replace(
            self.all,
            core_results={p: {k: v for k, v in self.all.core_results[p].items() if k != "supplements"}
                          for p in players},
            game_like_results={p: self.all.game_like_results[p] for p in sources},
            stability_results={p: self.all.stability_results[p] for p in sources},
        )

    def test_core_only_is_separate_from_measured_and_keeps_own_seed(self):
        target, source = "gpt-6-astra (high)", "gpt-6-astra (medium)"
        inputs = self.without_supplements(target, source)
        state = acquisition.load_acquisition_state(self.policy, inputs=inputs)
        self.assertEqual(state[target], {"core"})
        self.assertEqual(state.deferred_sources[target], source)
        self.assertEqual(acquisition.missing_panels(target, self.policy, state), [])
        self.assertEqual(len(acquisition.missing_panels(target, self.policy, state, full_audit=True)), 2)
        self.assertNotEqual(benchmark_predictions_from_inputs(inputs)[target],
                            benchmark_predictions_from_inputs(inputs)[source])

    def test_nearest_measured_sibling_cannot_be_bypassed_if_it_was_capped(self):
        target = "gemini-4-argon (high)"
        inputs = self.without_supplements(target, "gemini-4-argon (low)", "gemini-4-argon (medium)")
        self.assertIsNone(acquisition.deferred_supplement_source(target, inputs))

    def test_higher_core_requires_new_evidence_if_lower_panels_would_cap_it(self):
        target = "gpt-5.6-sol (medium)"
        inputs = self.without_supplements(target, "gpt-5.6-sol (low)")
        self.assertIsNone(acquisition.deferred_supplement_source(target, inputs))

    def test_remote_only_sibling_is_measured_but_unmeasured_sibling_is_ineligible(self):
        target, source = "gpt-6-astra (high)", "gpt-6-astra (medium)"
        inputs = self.without_supplements(target, source)
        lower = dict(inputs.core_results[source], supplements={
            "game_like": inputs.game_like_results[source], "stability": inputs.stability_results[source],
        })
        extra = "gpt-6-astra (xhigh)"
        inputs = replace(inputs, core_results={**inputs.core_results, source:lower,
                         extra: dict(inputs.core_results[target], summary=dict(inputs.core_results[target]["summary"], player_id=extra))}, game_like_results={}, stability_results={})
        state = acquisition.load_acquisition_state(self.policy, inputs=inputs)
        self.assertEqual(state.deferred_sources[extra], source)
        self.assertEqual(state[source], set(acquisition.SUPPORTED_AUTOMATIC_PANELS))

    def test_partial_target_or_incomplete_source_requires_acquisition(self):
        target, source = "gpt-6-astra (high)", "gpt-6-astra (medium)"
        inputs = self.without_supplements(target, source)
        partial = replace(inputs, game_like_results={**inputs.game_like_results, target: {"results": []}})
        self.assertIsNone(acquisition.deferred_supplement_source(target, partial))
        stale = replace(inputs, stability_results={source: {"summary": {}, "results": []}})
        self.assertIsNone(acquisition.deferred_supplement_source(target, stale))


def test_overlapping_static_acquisitions_preserve_other_players(tmp_path, monkeypatch):
    import asyncio
    import json
    import sys
    from unittest.mock import Mock
    from position_benchmark import run_benchmark as runner

    output = tmp_path / "core.json"
    target = "gemini-4-carbon (medium)"
    sample = load_position_benchmark_inputs(False).core_results[target]

    async def acquire(*args):
        # A second model finishes while the first is waiting on its provider.
        output.write_text(json.dumps({"other model": sample}))
        return dict(sample, results=sample["results"][:1], summary=sample["summary"].copy())

    monkeypatch.setattr(sys, "argv", ["acquire", "--players", target, "--limit", "1", "--output", str(output)])
    monkeypatch.setattr(runner, "run_benchmark", acquire)
    monkeypatch.setattr(runner.chess.engine.SimpleEngine, "popen_uci", Mock())
    asyncio.run(runner.main())
    assert set(json.loads(output.read_text())) == {target, "other model"}


def test_continuation_summary_without_all_measured_trajectories_cannot_defer():
    target, source = "gpt-6-astra (high)", "gpt-6-astra (medium)"
    inputs = load_position_benchmark_inputs(False)
    stability = inputs.stability_results[source]
    inputs = replace(inputs, core_results={p: inputs.core_results[p] for p in (target, source)},
                     game_like_results={source: inputs.game_like_results[source]},
                     stability_results={source: dict(stability, results=stability["results"][:-1])})
    assert acquisition.deferred_supplement_source(target, inputs) is None


def test_published_screen_ignores_unpublished_local_siblings_but_keeps_partial_remote_evidence():
    player = "gpt-6-astra (high)"
    bundled = load_position_benchmark_inputs(False)
    partial = {"supplements": {"game_like": {"results": []}}}
    inputs = load_position_benchmark_inputs(False, published_only=True, remote_results={
        player: bundled.core_results[player], "partial target": partial,
    })
    assert "gpt-6-astra (medium)" not in inputs.core_results
    assert acquisition.deferred_supplement_source(player, inputs) is None
    assert acquisition.player_panel_records(inputs, "partial target")[1] == {"results": []}


def test_shorter_probe_protocol_does_not_count_as_measured():
    source = "gpt-6-astra (medium)"
    inputs = load_position_benchmark_inputs(False)
    stability = inputs.stability_results[source]
    shortened = dict(stability, summary=dict(stability["summary"], probe_plies=4))
    policy = acquisition.load_acquisition_policy()
    inputs = replace(inputs, stability_results={source: shortened})
    assert "continuation_stability" not in acquisition.load_acquisition_state(policy, inputs=inputs)[source]


def test_production_state_can_publish_saved_target_without_using_unpublished_sibling(monkeypatch):
    target, source = "gpt-6-astra (high)", "gpt-6-astra (medium)"
    inputs = load_position_benchmark_inputs(False)
    remote = replace(inputs, core_results={target: inputs.core_results[target], source: inputs.core_results[source]},
                     game_like_results={source: inputs.game_like_results[source]},
                     stability_results={source: inputs.stability_results[source]})
    monkeypatch.setattr(acquisition, "load_position_benchmark_inputs", lambda *a, **kw: remote)
    policy = acquisition.load_acquisition_policy()
    state = acquisition.load_acquisition_state(policy, use_firestore=True)
    assert state[target] == {"core"}
    assert target not in state.deferred_sources
    assert state.local_ready_panels[target] == set(acquisition.SUPPORTED_AUTOMATIC_PANELS)
