"""Regenerate the shipped ``decision_replay.html`` DATA payload and compare it with the sample.

The fixture directory vendors the organizer's live-week validation run (``offline_score.json`` =
score-report-v3, ``decisions.csv`` and the shipped ``decision_replay.html``). The decisions come from
that run; the report and replay were re-scored and re-rendered against the current bundled reference
scenario at ``challenge/reference`` (checksums are asserted), so every night carries the real published
windows.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from challenge import replay
from challenge.contracts import sha256_file
from challenge.project_paths import EXAMPLE3_ROOT


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live_week_validation"
SAMPLE_HTML = FIXTURES / "decision_replay.html"
SAMPLE_REPORT = FIXTURES / "offline_score.json"
SAMPLE_DECISIONS = FIXTURES / "decisions.csv"
SCREENSHOT = ROOT / "artifacts" / "replay" / "replay-check.png"
SAMPLE_TITLE = "ASTRA // example3 weekly decision replay"
SAMPLE_AGENT_LABEL = "DeepSeek minimal-agent · seven-night replay"
FLOAT_TOLERANCE = 1e-6
REPORT_INPUT_FILES = {
    "tiles": "outputs/reference/tiles.csv",
    "targets": "outputs/reference/targets.csv",
    "slots": "outputs/reference/slots.csv",
    "weather": "outputs/reference/weather.csv",
    "forecasts": "outputs/reference/weather_forecasts.csv",
    "events": "outputs/reference/weather_events.csv",
    "requests": "outputs/reference/observation_requests.csv",
    "request_tiles": "outputs/reference/observation_request_tiles.csv",
    "anomalies": "outputs/reference/tile_anomalies.csv",
    "calendar_config": "config/calendar_config.json",
    "tile_config": "config/tile_config.json",
    "weather_config": "config/weather_config.json",
    "request_config": "config/request_config.json",
    "score_config": "config/score_config.json",
}


def extract_data(html_text: str) -> dict:
    match = re.search(r"^const DATA = (.*);$", html_text, re.M)
    assert match, "no DATA literal in HTML"
    return json.loads(match.group(1))


def assert_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(left, right, f"{path}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        assert math.isclose(float(actual), float(expected), abs_tol=FLOAT_TOLERANCE, rel_tol=0.0), f"{path}: {actual!r} != {expected!r}"
    else:  # pragma: no cover - the sample only contains JSON scalars
        raise AssertionError(f"{path}: unexpected type {type(expected)}")


@pytest.fixture(scope="module")
def sample() -> dict:
    return extract_data(SAMPLE_HTML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(SAMPLE_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated(report) -> dict:
    data = replay.build_replay_data(EXAMPLE3_ROOT, report, title=SAMPLE_TITLE, agent_label=SAMPLE_AGENT_LABEL, decisions_path=SAMPLE_DECISIONS)
    return json.loads(replay.serialize_replay_data(data))


def test_sample_report_was_scored_against_bundled_reference_scenario(report):
    manifest = json.loads((EXAMPLE3_ROOT / "outputs" / "reference" / "scenario_manifest.json").read_text(encoding="utf-8"))["files"]
    for key, relative in REPORT_INPUT_FILES.items():
        assert report["input_sha256"][key] == sha256_file(EXAMPLE3_ROOT / relative), key
        if relative in manifest:
            assert report["input_sha256"][key] == manifest[relative]["sha256"], key
    assert report["input_sha256"]["decisions"] == sha256_file(SAMPLE_DECISIONS)


def test_template_is_sample_with_data_placeholder():
    template = replay.TEMPLATE_PATH.read_bytes()
    sample_bytes = SAMPLE_HTML.read_bytes()
    assert template.count(replay.DATA_PLACEHOLDER.encode()) == 1
    match = re.search(rb"^const DATA = (.*);$", sample_bytes, re.M)
    assert template.replace(replay.DATA_PLACEHOLDER.encode(), match.group(1), 1) == sample_bytes


def test_data_matches_sample(generated, sample):
    assert set(generated) == set(sample) == {"meta", "site", "tiles", "nights", "events", "rounds", "score_summary"}
    assert generated["meta"] == sample["meta"]
    assert_close(generated["site"], sample["site"], "site")
    assert_close(generated["tiles"], sample["tiles"], "tiles")
    assert list(generated["events"]) == list(sample["events"]), "event ids must follow weather_events.csv order"
    assert_close(generated["events"], sample["events"], "events")
    assert set(generated["nights"]) == set(sample["nights"])
    assert list(generated["nights"]) == list(sample["nights"])
    assert_close(generated["rounds"], sample["rounds"], "rounds")
    assert_close(generated["score_summary"], sample["score_summary"], "score_summary")


def test_events_are_anonymised(generated):
    for event in generated["events"].values():
        assert set(event) == {"condition", "regions", "severity"}
    literal = replay.serialize_replay_data(generated)
    for forbidden in ("actual_start_utc", "actual_end_utc", "spatial_scope_payload", "force_close", "_multiplier"):
        assert forbidden not in literal


def test_nights_reproduce_sample_where_possible(generated, sample, report):
    for night_id, night in sample["nights"].items():
        ours = generated["nights"][night_id]
        assert set(ours) == set(night) == {"tile_status", "forecast_snapshot", "forecasts"}
        assert ours["forecast_snapshot"] == night["forecast_snapshot"]
        assert_close(ours["forecasts"], night["forecasts"], f"nights.{night_id}.forecasts")
        assert ours["tile_status"] == night["tile_status"], night_id
    # Every completed exposure in the report started inside a window we publish for that night.
    windows = {
        (night_id, status["tile_id"]): status["windows"]
        for night_id, night in generated["nights"].items() for status in night["tile_status"]
    }
    completed = [action for action in report["actions"] if action["outcome"] == "completed"]
    assert completed
    for action in completed:
        night_id = action["slot_id"].split("-")[0]
        assert any(w["window_start_utc"] <= action["start_utc"] < w["window_end_utc"] for w in windows[(night_id, action["tile_id"])]), action["decision_id"]
    assert {action["slot_id"].split("-")[0] for action in completed} - set(list(sample["nights"])[:3]), "sample nights 4-7 do contain completed exposures"


def test_write_replay_html_is_self_contained(tmp_path, report):
    out = replay.write_replay_html(EXAMPLE3_ROOT, report, tmp_path / "replay.html", title=SAMPLE_TITLE, agent_label=SAMPLE_AGENT_LABEL, decisions_path=SAMPLE_DECISIONS)
    page = out.read_text(encoding="utf-8")
    assert "http://" not in page and "https://" not in page and "<link" not in page
    script = page[page.index("<script>"):page.index("</script>")]
    assert "fetch(" not in script and "import " not in script
    assert page.count("<script") == 1
    # Everything outside the DATA literal is byte-identical to the shipped sample.
    strip = lambda text: re.sub(r"^const DATA = .*;$", "const DATA = X;", text, flags=re.M)
    assert strip(page) == strip(SAMPLE_HTML.read_text(encoding="utf-8"))


def test_cli_renders_report(tmp_path):
    out = tmp_path / "cli" / "replay.html"
    result = subprocess.run(
        [sys.executable, "-m", "challenge.replay", "--root", str(EXAMPLE3_ROOT), "--report", str(SAMPLE_REPORT), "--output", str(out),
         "--title", "CLI replay", "--agent-label", "cli & agent"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == str(out)
    page = out.read_text(encoding="utf-8")
    assert "<title>CLI replay</title>" in page and '<div class="subtitle">cli &amp; agent</div>' in page
    data = extract_data(page)
    assert data["meta"]["title"] == "CLI replay" and data["meta"]["round_count"] == 265
    # decisions.csv beside the report is picked up automatically for reason text.
    assert data["rounds"][0]["decision"]["reason"] == "Highest estimated gain per second among listed candidates."


def test_replay_renders_in_chromium(tmp_path, report):
    sync_api = pytest.importorskip("playwright.sync_api")
    out = replay.write_replay_html(EXAMPLE3_ROOT, report, tmp_path / "replay.html", title=SAMPLE_TITLE, agent_label=SAMPLE_AGENT_LABEL, decisions_path=SAMPLE_DECISIONS)
    problems: list[str] = []
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except sync_api.Error as exc:  # pragma: no cover - depends on the local browser install
            pytest.skip(f"chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda message: problems.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: problems.append(str(error)))
        page.goto(out.resolve().as_uri())
        page.wait_for_timeout(750)
        canvas = page.evaluate(
            """() => {
                const canvas = document.getElementById('sky');
                const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
                const colors = new Set();
                let painted = 0;
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i + 3] !== 0 && (data[i] | data[i + 1] | data[i + 2]) !== 0) painted++;
                    if (colors.size < 256) colors.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
                }
                return {width: canvas.width, height: canvas.height, painted, colors: colors.size};
            }"""
        )
        round_label = page.text_content("#round-label")
        night_label = page.text_content("#night-label")
        SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SCREENSHOT))
        browser.close()
    assert problems == []
    assert canvas["width"] > 0 and canvas["height"] > 0
    assert canvas["painted"] > 0.5 * canvas["width"] * canvas["height"], canvas
    assert canvas["colors"] >= 3, canvas
    assert round_label == "ROUND 001 / 265" and night_label == "NIGHT 01 / 7"
    assert SCREENSHOT.stat().st_size > 0
