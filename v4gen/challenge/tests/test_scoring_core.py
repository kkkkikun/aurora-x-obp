from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from challenge.scoring_core import ChallengeScorer, Decision
from challenge.weather_simulator import weather_quality


def complete_start(scorer: ChallengeScorer, tile_id: str, lower=None, upper=None):
    tile = scorer.tiles[tile_id]
    for index, slot in enumerate(scorer.slots[:-1]):
        start = slot.timestamp_utc
        if (lower is not None and start < lower) or (upper is not None and start >= upper):
            continue
        remaining = tile.nominal_exptime_seconds
        cursor = index
        okay = True
        while remaining > 0:
            current = scorer.slots[cursor]
            if current.night_id != slot.night_id or not scorer._tile_legal(tile, current.timestamp_utc):
                okay = False
                break
            weather = scorer.weather.get_effective_conditions(current.slot_id, tile_id)
            if not weather["is_observable"]:
                okay = False
                break
            remaining -= min(remaining, current.duration_seconds)
            cursor += 1
        if okay:
            return index
    raise AssertionError(f"no complete opportunity found for {tile_id}")


class ScoringCoreTests(unittest.TestCase):
    def test_efficiency_never_changes_the_band(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        slot = next(item for item in scorer.weather.weather if item.is_observable)
        tile = next(iter(scorer.tiles.values()))
        conditions = scorer.weather.get_effective_conditions(slot.slot_id, tile.tile_id)
        airmass = float(scorer.geometry.get_tile_geometry(tile.tile_id, slot.timestamp_utc)["airmass"])
        bands = set()
        for efficiency in (0.10, 0.55, 0.95, 1.00):
            degraded = {**conditions, "instrument_efficiency": efficiency}
            bands.add(scorer._quality_band(weather_quality(degraded, airmass, scorer.weather.config, include_efficiency=False)))
        self.assertEqual(len(bands), 1)

    def test_efficiency_still_scales_the_score(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        slot = next(item for item in scorer.weather.weather if item.is_observable)
        tile = next(iter(scorer.tiles.values()))
        conditions = scorer.weather.get_effective_conditions(slot.slot_id, tile.tile_id)
        airmass = float(scorer.geometry.get_tile_geometry(tile.tile_id, slot.timestamp_utc)["airmass"])
        full = weather_quality(conditions, airmass, scorer.weather.config)
        degraded = weather_quality({**conditions, "instrument_efficiency": conditions["instrument_efficiency"] * 0.5}, airmass, scorer.weather.config)
        self.assertLess(degraded, full)
        self.assertEqual(
            scorer._quality_band(weather_quality(conditions, airmass, scorer.weather.config, include_efficiency=False)),
            scorer._quality_band(weather_quality({**conditions, "instrument_efficiency": conditions["instrument_efficiency"] * 0.5}, airmass, scorer.weather.config, include_efficiency=False)),
        )

    def test_lunar_quality_scales_science_and_program_band(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        selected = None
        for tile in scorer.tiles.values():
            if tile.nominal_exptime_seconds != 450:
                continue
            for index, slot in enumerate(scorer.slots):
                if not scorer._tile_legal(tile, slot.timestamp_utc):
                    continue
                conditions = scorer.weather.get_effective_conditions(
                    slot.slot_id, tile.tile_id
                )
                if not conditions["is_observable"]:
                    continue
                midpoint = slot.timestamp_utc + timedelta(seconds=225)
                geometry = scorer.geometry.get_tile_geometry(tile.tile_id, midpoint)
                atmospheric = weather_quality(
                    conditions, float(geometry["airmass"]), scorer.weather.config
                )
                if 0.65 <= atmospheric < 1.6:
                    selected = tile, index, atmospheric
                    break
            if selected:
                break
        self.assertIsNotNone(selected)
        tile, index, atmospheric = selected
        original_geometry = scorer.geometry.get_tile_geometry

        def lunar_stressed_geometry(tile_id, moment):
            result = original_geometry(tile_id, moment)
            result["lunar_quality_factor"] = 0.25
            return result

        scorer.geometry.get_tile_geometry = lunar_stressed_geometry
        scorer.slot_index = index
        action = scorer.apply_decision(
            Decision(
                "LUNAR",
                scorer.slots[index].slot_id,
                "observe",
                tile.tile_id,
                "BACKUP",
                "",
                "combined lunar quality test",
            )
        )
        self.assertEqual(action["outcome"], "completed")
        self.assertEqual(len(action["segments"]), 1)
        segment = action["segments"][0]
        self.assertAlmostEqual(segment["atmospheric_quality"], atmospheric, places=6)
        self.assertEqual(segment["lunar_quality_factor"], 0.25)
        self.assertAlmostEqual(
            segment["combined_quality"], atmospheric * 0.25, places=6
        )
        # Bands are determined on the efficiency-free quality now.
        band_quality = weather_quality(conditions, float(scorer.geometry.get_tile_geometry(tile.tile_id, slot.timestamp_utc + timedelta(seconds=225))["airmass"]), scorer.weather.config, include_efficiency=False) * 0.25
        thresholds = scorer.config["quality_thresholds"]
        expected_band = "DARK" if band_quality >= float(thresholds["dark"]) else "BRIGHT" if band_quality >= float(thresholds["bright"]) else "BACKUP"
        self.assertEqual(segment["quality_band"], expected_band)
        self.assertEqual(segment["program_matched"], expected_band == "BACKUP")
        self.assertGreater(action["base_science_score"], 0.0)

    def test_cross_slot_exposure_and_second_action_in_same_slot(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        tile = next(item for item in scorer.tiles.values() if item.nominal_exptime_seconds > 900)
        index = complete_start(scorer, tile.tile_id)
        scorer.slot_index = index
        first = scorer.apply_decision(Decision("D1", scorer.slots[index].slot_id, "observe", tile.tile_id, "DARK", "", "cross-slot test"))
        self.assertEqual(first["outcome"], "completed")
        self.assertGreaterEqual(len(first["segments"]), 2)
        current = scorer.current_slot()
        self.assertIsNotNone(current)
        self.assertGreater(scorer.offset_seconds, 0)
        second = scorer.apply_decision(Decision("D2", current.slot_id, "wait", "", "", "", "same-slot second action"))
        self.assertEqual(second["outcome"], "wait")

    def test_site_closed_observe_is_unsafe_and_consumes_slot(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        closed = next(index for index, item in enumerate(scorer.weather.weather) if not item.is_observable)
        scorer.slot_index = closed
        slot = scorer.current_slot()
        tile = next(iter(scorer.tiles.values()))
        action = scorer.apply_decision(Decision("D1", slot.slot_id, "observe", tile.tile_id, "BACKUP", "", "unsafe test"))
        self.assertEqual(action["outcome"], "unsafe_observation")
        self.assertEqual(action["penalty"], scorer.config["penalties"]["unsafe_observation"])
        self.assertEqual(scorer.slot_index, closed + 1)

    def test_request_tagged_revisit_scores_and_counts_a_visit(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        request = next(iter(scorer.requests.values()))
        tile_id = next(iter(scorer.request_tiles[request.request_id]))
        index = complete_start(scorer, tile_id, request.available_from_utc, request.deadline_utc)
        scorer.slot_index = index
        scorer.completed_tiles.add(tile_id)
        action = scorer.apply_decision(Decision("D1", scorer.slots[index].slot_id, "observe", tile_id, "BRIGHT", request.request_id, "request revisit"))
        self.assertEqual(action["outcome"], "completed")
        self.assertGreater(action["base_science_score"], 0.0)
        best = scorer.tile_best_scores[tile_id]
        self.assertAlmostEqual(best[0] + best[1], action["base_science_score"] + action["program_bonus_score"], places=5)
        self.assertEqual(scorer.request_visits[(request.request_id, tile_id)], 1)

    def test_global_cutoff_does_not_synthesize_future_wait(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        first = scorer.apply_decision(Decision("D1", scorer.slots[0].slot_id, "wait", "", "", "", "one committed action"))
        self.assertEqual(first["outcome"], "wait")
        report = scorer.finalize("global_wallclock_expired")
        self.assertEqual(report["termination_reason"], "global_wallclock_expired")
        self.assertEqual(report["final_cursor"]["slot_index"], 1)
        self.assertEqual(scorer.wait_seconds["explicit"], scorer.slots[0].duration_seconds)
        self.assertNotIn("future_unprocessed", scorer.wait_seconds)

    def test_expired_request_without_any_legal_slot_is_excused(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        source = next(iter(scorer.requests.values()))
        daytime_start = scorer.slots[0].end_utc + timedelta(hours=1)
        impossible = replace(
            source,
            request_id="IMPOSSIBLE",
            issued_at_utc=scorer.slots[0].timestamp_utc,
            available_from_utc=daytime_start,
            deadline_utc=daytime_start + timedelta(hours=1),
            required_tile_count=1,
        )
        tile_id = next(iter(scorer.tiles))
        scorer.requests = {impossible.request_id: impossible}
        scorer.request_tiles = {impossible.request_id: {tile_id: 1}}
        scorer.slot_index = next(
            index
            for index, slot in enumerate(scorer.slots)
            if slot.timestamp_utc > impossible.deadline_utc
        )
        report = scorer.finalize()
        self.assertEqual(report["requests"][0]["status"], "excused_unobservable")
        self.assertEqual(report["requests"][0]["penalty"], 0.0)

    def _observe(self, scorer: ChallengeScorer, decision_id: str, tile_id: str, index: int, program: str = "DARK", request_id: str = "") -> dict:
        scorer.slot_index = index
        scorer.offset_seconds = 0
        return scorer.apply_decision(Decision(decision_id, scorer.slots[index].slot_id, "observe", tile_id, program, request_id, "test observation"))

    def test_repeat_observation_banks_the_maximum(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        tile = next(item for item in scorer.tiles.values() if item.nominal_exptime_seconds <= 900)
        original_conditions = scorer.weather.get_effective_conditions

        def degraded(slot_id, tile_id=None, **kwargs):
            payload = original_conditions(slot_id, tile_id, **kwargs)
            if payload["is_observable"]:
                payload = {**payload, "instrument_efficiency": payload["instrument_efficiency"] * 0.5}
            return payload

        # First observation under degraded conditions banks a low best.
        first_index = complete_start(scorer, tile.tile_id)
        scorer.weather.get_effective_conditions = degraded
        first = self._observe(scorer, "D1", tile.tile_id, first_index)
        scorer.weather.get_effective_conditions = original_conditions
        self.assertEqual(first["outcome"], "completed")
        first_score = first["base_science_score"] + first["program_bonus_score"]
        self.assertGreater(first_score, 0.0)

        # A repeat under better conditions raises the banked total to its own score.
        second_index = complete_start(scorer, tile.tile_id, lower=scorer.slots[first_index + 1].timestamp_utc)
        second = self._observe(scorer, "D2", tile.tile_id, second_index)
        self.assertEqual(second["outcome"], "completed")
        second_score = second["base_science_score"] + second["program_bonus_score"]
        self.assertGreater(second_score, first_score)
        self.assertAlmostEqual(scorer.base_science_score + scorer.program_bonus_score, second_score, places=5)

        # A repeat under degraded conditions scores lower and leaves the best untouched.
        third_index = complete_start(scorer, tile.tile_id, lower=scorer.slots[second_index + 1].timestamp_utc)
        scorer.weather.get_effective_conditions = degraded
        third = self._observe(scorer, "D3", tile.tile_id, third_index)
        scorer.weather.get_effective_conditions = original_conditions
        self.assertEqual(third["outcome"], "completed")
        self.assertLess(third["base_science_score"] + third["program_bonus_score"], second_score)
        self.assertAlmostEqual(scorer.base_science_score + scorer.program_bonus_score, second_score, places=5)

    def test_completion_still_banks_on_first_legal_observation(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        tile = next(iter(scorer.tiles.values()))
        index = complete_start(scorer, tile.tile_id)
        first = self._observe(scorer, "D1", tile.tile_id, index)
        self.assertEqual(first["outcome"], "completed")
        self.assertIn(tile.tile_id, scorer.completed_tiles)
        second_index = complete_start(scorer, tile.tile_id, lower=scorer.slots[index + 1].timestamp_utc)
        second = self._observe(scorer, "D2", tile.tile_id, second_index)
        self.assertEqual(second["outcome"], "completed")
        report = scorer.finalize()
        self.assertEqual(report["completion"]["completed_tiles"].count(tile.tile_id), 1)

    def test_untagged_repeat_is_legal(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        tile = next(iter(scorer.tiles.values()))
        index = complete_start(scorer, tile.tile_id)
        first = self._observe(scorer, "D1", tile.tile_id, index)
        self.assertEqual(first["outcome"], "completed")
        second_index = complete_start(scorer, tile.tile_id, lower=scorer.slots[index + 1].timestamp_utc)
        second = self._observe(scorer, "D2", tile.tile_id, second_index)
        self.assertEqual(second["outcome"], "completed")
        self.assertEqual(second["penalty"], 0.0)

    def test_request_on_observed_tile_requires_post_issue_observation(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        request = next(item for item in scorer.requests.values() if item.available_from_utc > scorer.slots[0].timestamp_utc)
        tile_id = next(iter(scorer.request_tiles[request.request_id]))
        # An observation before the request's window never counted as a visit;
        # the new post-issue request-tagged observation does, and scores normally.
        index = complete_start(scorer, tile_id, request.available_from_utc, request.deadline_utc)
        scorer.completed_tiles.add(tile_id)  # tile observed before the request existed
        action = self._observe(scorer, "D1", tile_id, index, request_id=request.request_id)
        self.assertEqual(action["outcome"], "completed")
        self.assertGreater(action["base_science_score"], 0.0)
        self.assertEqual(scorer.request_visits[(request.request_id, tile_id)], 1)
        # A tag before the request's window is still invalid.
        scorer2 = ChallengeScorer.from_files(ROOT / "reference")
        rejected = self._observe(scorer2, "D1", tile_id, 0, request_id=request.request_id)
        self.assertEqual(rejected["outcome"], "invalid_request_tag")
        self.assertEqual(scorer2.request_visits[(request.request_id, tile_id)], 0)

    def test_request_observation_of_unobserved_tile_is_ordinary_completion(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        request = next(iter(scorer.requests.values()))
        tile_id = next(iter(scorer.request_tiles[request.request_id]))
        index = complete_start(scorer, tile_id, request.available_from_utc, request.deadline_utc)
        action = self._observe(scorer, "D1", tile_id, index, request_id=request.request_id)
        self.assertEqual(action["outcome"], "completed")
        self.assertIn(tile_id, scorer.completed_tiles)
        self.assertIn(tile_id, scorer.tile_best_scores)
        self.assertEqual(scorer.request_visits[(request.request_id, tile_id)], 1)
        report = scorer.finalize()
        row = next(item for item in report["requests"] if item["request_id"] == request.request_id)
        self.assertGreaterEqual(row["satisfied_tile_count"], 1)

    def test_wait_is_avoidable_when_a_repeat_could_improve(self) -> None:
        # Only completed tiles can be observed (all tiles completed, nothing banked
        # yet): a repeat can always improve on a zero best, so waiting is penalised.
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        scorer.completed_tiles = set(scorer.tiles)
        tile = next(item for item in scorer.tiles.values() if item.nominal_exptime_seconds <= 900)
        index = complete_start(scorer, tile.tile_id)
        scorer.slot_index = index
        scorer.offset_seconds = 0
        self.assertGreater(scorer._repeat_score_potential(tile, index, 0), 0.0)
        scorer._consume_wait(scorer.current_slot().duration_seconds, "explicit")
        self.assertGreater(scorer.wait_seconds["avoidable"], 0)
        self.assertGreater(scorer.penalties["avoidable_wait"], 0.0)

    def test_wait_is_unavoidable_when_no_repeat_could_improve(self) -> None:
        # All tiles completed with banked bests above any achievable score:
        # no repeat can improve, so waiting stays unavoidable.
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        scorer.completed_tiles = set(scorer.tiles)
        scorer.tile_best_scores = {tile_id: (1e9, 0.0) for tile_id in scorer.tiles}
        tile = next(item for item in scorer.tiles.values() if item.nominal_exptime_seconds <= 900)
        index = complete_start(scorer, tile.tile_id)
        scorer.slot_index = index
        scorer.offset_seconds = 0
        scorer._consume_wait(scorer.current_slot().duration_seconds, "explicit")
        self.assertEqual(scorer.penalties["avoidable_wait"], 0.0)
        self.assertGreater(scorer.wait_seconds["unavailable"], 0)

    def test_wait_is_unavoidable_when_site_is_closed(self) -> None:
        scorer = ChallengeScorer.from_files(ROOT / "reference")
        closed = next(index for index, item in enumerate(scorer.weather.weather) if not item.is_observable)
        scorer.slot_index = closed
        scorer._consume_wait(scorer.current_slot().duration_seconds, "explicit")
        self.assertEqual(scorer.penalties["avoidable_wait"], 0.0)
        self.assertGreater(scorer.wait_seconds["unavailable"], 0)


if __name__ == "__main__":
    unittest.main()
