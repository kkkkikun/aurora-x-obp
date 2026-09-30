from __future__ import annotations

import copy
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

from challenge.challenge_workflow import ChallengeWorkflow
from decision_graph import MinimalDecisionAgent
from model_factory import ModelSettings, build_chat_model
from challenge.scoring_preview import preview_actions


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeResponse(json.dumps(self.payload))


class MinimalAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workflow = ChallengeWorkflow(ROOT / "reference")
        cls.initial = workflow.initial_publication()
        cls.snapshot = workflow.decision_snapshot(1)

    def test_preview_is_sorted_and_uses_matching_program(self) -> None:
        previews = preview_actions(
            self.snapshot, self.initial["scoring_contract"]
        )
        self.assertTrue(previews)
        rates = [item.estimated_gain_per_second for item in previews]
        self.assertEqual(rates, sorted(rates, reverse=True))
        for item in previews:
            self.assertEqual(item.program, item.quality_band)
            self.assertGreater(item.estimated_science_score, 0)
            self.assertIn("future weather", item.estimate_semantics)

    def test_preview_baseline_excludes_instrument_efficiency(self) -> None:
        previews = preview_actions(self.snapshot, self.initial["scoring_contract"])
        row = previews[0]
        candidate = next(c for c in self.snapshot["candidate_tiles"] if c["tile_id"] == row.tile_id)
        weather = candidate["effective_weather"]
        self.assertNotIn("instrument_efficiency", weather)
        interface = self.initial["scoring_contract"]["weather_score_interface"]
        expected = min(
            weather["transparency"] * weather["sky_quality"]
            / (weather["seeing_arcsec"] * candidate["geometry"]["airmass"] ** interface["airmass_exponent"]),
            interface["maximum_weather_quality"],
        )
        self.assertAlmostEqual(row.atmospheric_quality, expected, places=6)

    def test_closed_candidates_produce_wait(self) -> None:
        snapshot = copy.deepcopy(self.snapshot)
        for candidate in snapshot["candidate_tiles"]:
            candidate["effective_weather"]["is_observable"] = False
        agent = MinimalDecisionAgent(self.initial, model=None, top_k=12)
        decision = agent.decide(snapshot)
        self.assertEqual(decision["action"], "wait")
        self.assertEqual(decision["decision_source"], "deterministic")

    def test_completed_candidates_show_zero_marginal_gain_without_best_scores(self) -> None:
        snapshot = copy.deepcopy(self.snapshot)
        for candidate in snapshot["candidate_tiles"]:
            candidate["already_completed"] = True
        previews = preview_actions(snapshot, self.initial["scoring_contract"])
        self.assertTrue(previews)
        self.assertTrue(all(item.estimated_science_score == 0 for item in previews))

    def test_completed_candidates_show_marginal_gain_over_tracked_best(self) -> None:
        snapshot = copy.deepcopy(self.snapshot)
        for candidate in snapshot["candidate_tiles"]:
            candidate["already_completed"] = True
        previews = preview_actions(snapshot, self.initial["scoring_contract"])
        tile_id = previews[0].tile_id
        potential = previews[0].tile_science_value * previews[0].combined_quality * (
            1.0 + self.initial["scoring_contract"]["score_config"]["program_bonus"][previews[0].quality_band]
        )
        tracked = preview_actions(snapshot, self.initial["scoring_contract"], tile_best_scores={tile_id: potential - 1.0})
        row = next(item for item in tracked if item.tile_id == tile_id)
        self.assertAlmostEqual(row.estimated_science_score, 1.0, places=4)
        saturated = preview_actions(snapshot, self.initial["scoring_contract"], tile_best_scores={tile_id: potential + 1.0})
        row = next(item for item in saturated if item.tile_id == tile_id)
        self.assertEqual(row.estimated_science_score, 0.0)

    def test_no_model_uses_highest_public_estimate(self) -> None:
        best = preview_actions(
            self.snapshot, self.initial["scoring_contract"]
        )[0]
        decision = MinimalDecisionAgent(
            self.initial, model=None, top_k=12
        ).decide(self.snapshot)
        self.assertEqual(
            (decision["tile_id"], decision["program"], decision["request_id"]),
            (best.tile_id, best.program, best.request_id),
        )
        self.assertEqual(decision["decision_source"], "deterministic")

    def test_valid_model_choice_is_accepted(self) -> None:
        chosen = preview_actions(
            self.snapshot, self.initial["scoring_contract"]
        )[1]
        model = FakeModel(
            {
                "action": "observe",
                "tile_id": chosen.tile_id,
                "program": chosen.program,
                "request_id": chosen.request_id,
                "reason": "valid alternate from visible candidates",
            }
        )
        decision = MinimalDecisionAgent(
            self.initial, model=model, top_k=12
        ).decide(self.snapshot)
        self.assertEqual(decision["tile_id"], chosen.tile_id)
        self.assertEqual(decision["decision_source"], "model")
        self.assertEqual(len(model.calls), 1)

    def test_hallucinated_model_choice_falls_back(self) -> None:
        model = FakeModel(
            {
                "action": "observe",
                "tile_id": "NOT_A_TILE",
                "program": "DARK",
                "request_id": "",
                "reason": "hallucinated",
            }
        )
        best = preview_actions(
            self.snapshot, self.initial["scoring_contract"]
        )[0]
        decision = MinimalDecisionAgent(
            self.initial, model=model, top_k=12
        ).decide(self.snapshot)
        self.assertEqual(decision["tile_id"], best.tile_id)
        self.assertEqual(decision["decision_source"], "deterministic")

    def test_environment_defaults_to_secret_free_deterministic_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertTrue(settings.deterministic)
        self.assertEqual(settings.api_key, "")
        self.assertNotIn("api_key", repr(settings))

    def test_provider_alias_and_key_indirection(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MODEL_PROVIDER": "grok",
                "MODEL_NAME": "test-model",
                "MODEL_BASE_URL": "https://example.invalid/v1",
                "XAI_API_KEY": "test-secret",
            },
            clear=True,
        ):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertEqual(settings.provider, "xai")
        self.assertEqual(settings.api_key, "test-secret")
        self.assertNotIn("test-secret", repr(settings))

    def test_openai_and_compatible_adapters_use_distinct_api_modes(self) -> None:
        class DummyChatOpenAI:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        module = types.SimpleNamespace(ChatOpenAI=DummyChatOpenAI)
        with patch.dict(sys.modules, {"langchain_openai": module}):
            native = build_chat_model(
                ModelSettings(
                    provider="openai",
                    model="test-openai",
                    base_url="",
                    api_key="secret",
                    api_mode="responses",
                )
            )
            compatible = build_chat_model(
                ModelSettings(
                    provider="deepseek",
                    model="test-compatible",
                    base_url="https://example.invalid/v1",
                    api_key="secret",
                )
            )
        self.assertTrue(native.kwargs["use_responses_api"])
        self.assertFalse(compatible.kwargs["use_responses_api"])
        self.assertEqual(
            compatible.kwargs["base_url"], "https://example.invalid/v1"
        )

    def test_anthropic_uses_native_adapter(self) -> None:
        class DummyChatAnthropic:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        module = types.SimpleNamespace(ChatAnthropic=DummyChatAnthropic)
        with patch.dict(sys.modules, {"langchain_anthropic": module}):
            model = build_chat_model(
                ModelSettings(
                    provider="anthropic",
                    model="test-anthropic",
                    base_url="",
                    api_key="secret",
                )
            )
        self.assertEqual(model.kwargs["model"], "test-anthropic")

    def test_named_compatible_profiles_resolve_their_documented_keys(self) -> None:
        profiles = {
            "grok": ("xai", "XAI_API_KEY"),
            "glm": ("zai", "ZAI_API_KEY"),
            "deepseek": ("deepseek", "DEEPSEEK_API_KEY"),
            "kimi": ("moonshot", "MOONSHOT_API_KEY"),
            "qwen": ("dashscope", "DASHSCOPE_API_KEY"),
            "minimax": ("minimax", "MINIMAX_API_KEY"),
        }
        for entered, (normalized, key_name) in profiles.items():
            with self.subTest(provider=entered), patch.dict(
                os.environ,
                {
                    "MODEL_PROVIDER": entered,
                    "MODEL_NAME": "test-model",
                    "MODEL_BASE_URL": "https://example.invalid/v1",
                    key_name: "test-secret",
                },
                clear=True,
            ):
                settings = ModelSettings.from_environment(Path("/does/not/exist"))
                self.assertEqual(settings.provider, normalized)
                self.assertEqual(settings.api_key, "test-secret")

    def test_platform_openai_variables_take_precedence(self) -> None:
        platform = {
            "OPENAI_BASE_URL": "https://platform.invalid/functions/v1/observer-model/v1",
            "OPENAI_API_KEY": "run-credential",
        }
        local = {
            "MODEL_PROVIDER": "glm",
            "MODEL_NAME": "local-model",
            "MODEL_BASE_URL": "https://example.invalid/v1",
            "ZAI_API_KEY": "local-secret",
        }
        with patch.dict(os.environ, {**local, **platform}, clear=True):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertEqual(settings.base_url, platform["OPENAI_BASE_URL"])
        self.assertEqual(settings.api_key, "run-credential")
        self.assertEqual(settings.model, "local-model")
        self.assertEqual(settings.api_mode, "chat")
        # Local runs without the platform variables keep the provider settings.
        with patch.dict(os.environ, local, clear=True):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertEqual(settings.base_url, "https://example.invalid/v1")
        self.assertEqual(settings.api_key, "local-secret")

    def test_platform_proxy_needs_no_model_name_and_uses_chat_completions(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MODEL_PROVIDER": "openai",
                "OPENAI_BASE_URL": "https://platform.invalid/v1",
                "OPENAI_API_KEY": "run-credential",
            },
            clear=True,
        ):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertEqual(settings.model, "team-model")
        self.assertEqual(settings.api_mode, "chat")
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "https://platform.invalid/v1", "OPENAI_API_KEY": "run-credential"},
            clear=True,
        ):
            self.assertTrue(ModelSettings.from_environment(Path("/does/not/exist")).deterministic)
        with patch.dict(
            os.environ,
            {"MODEL_PROVIDER": "openai", "OPENAI_MODEL": "chosen", "OPENAI_API_KEY": "k"},
            clear=True,
        ):
            settings = ModelSettings.from_environment(Path("/does/not/exist"))
        self.assertEqual((settings.model, settings.api_mode), ("chosen", "responses"))


if __name__ == "__main__":
    unittest.main()
