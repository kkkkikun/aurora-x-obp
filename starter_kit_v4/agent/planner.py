"""Baseline planner for the v4 survey: pick a pointing, fill the 16 fibres, choose exposure and program.

The idea, step by step (every number below is a tunable constant at the top of the file):

1. At night: find the targets that are above the altitude limit now and stay up for a while.
2. Rank them. Required targets that are not done yet get a large bonus (a missing required target
   costs a lot at the end). Targets that will set soon, or have few nights left, rank higher.
3. For the best few "anchor" targets, try 16 pointings: the anchor centred on each fibre. For each
   pointing, give every fibre the most valuable target that lands on its glass. Keep the best pointing.
4. Choose the exposure length that gives the most expected score per second, and the program
   (DARK / BRIGHT / BACKUP) that most assigned targets will match.
5. Learn from results: each hit reports its score, so the planner learns how good the sky is right
   now and which targets are already done.

The planner never sees hidden weather. It only uses the public catalogue, the public score formula,
bulletins/forecasts and its own results. Standard library only.
"""
from __future__ import annotations

import bisect
import math
import os
from collections import deque
from datetime import datetime, timedelta

from skymath import (
    SIDEREAL_DEG_PER_SECOND,
    FiberGrid,
    Moon,
    altaz_to_radec,
    local_sidereal_deg,
    max_hour_angle_deg,
    normalized_airmass,
    parse_utc,
    radec_to_altaz,
    shift_altaz,
    tangent_offsets,
    wrap180,
)

ALT_MARGIN_DEG = 0.6            # keep targets this far above the altitude limit for the whole exposure
REQUIRED_BONUS = 60.0           # planning value of one required target (the penalty for missing one is 50)
REQUIRED_SAFE_FACTOR = 0.62     # treat a required target as done once its estimated factor reaches this
DONE_FACTOR = 0.95              # other targets are done at this factor
PLAN_FACTOR_SAFETY = 0.9        # plan exposures as if the sky were 10% worse than estimated
EDGE_MARGIN_DEG = 0.08          # prefer targets at least this far inside the fibre glass
DURATIONS = (300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600)
RECENT_SAMPLES = 24             # fault check: recent clean samples, spanning at least two nights
EARLIER_SAMPLES = 24            # ... compared with at least this many earlier ones
UNIF_BOOST = 0.8                # priority boost per unit RA-band completion deficit
SKY_MEMORY_HOURS = 2.0         # forget sky-quality samples older than this (in survey time)
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_DEG = 2.1
CALIB_MIN_SAMPLES = 60         # pointing-offset calibration: samples before first estimate
CALIB_REESTIMATE = 40          # re-estimate after this many new samples
CALIB_STEP_DEG = 0.01          # grid step of the offset search (local tangent plane)
CALIB_MAX_DEG = 0.18           # search radius (Hard-mode offsets are ~0.1 deg)
CALIB_MIN_MAG = 0.03           # minimum offset magnitude worth compensating
CALIB_MIN_GAIN = 0.10          # required relative consistency gain over zero-offset
ANCHORS = 4                   # widened for long seasons (short cards keep 3, see __init__)
ANCHOR_POOL = 220             # top-ranked candidates checked for what they can still gain tonight
ANCHORS_MIN_NIGHTS = 14       # a 4th anchor delays required work on short seasons: measured -112
CLOSED_KINDS = {"rain", "storm"}
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


class Planner:
    def __init__(self, init: dict, log=lambda text: None):
        self.log = log
        site = init["site"]
        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site["minimum_altitude_deg"])
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"])) for n in init["survey"]["nights"]]
        self.survey_end = parse_utc(init["survey"]["end_utc"])
        self.slot_seconds = int(init["survey"]["slot_seconds"])
        instrument = init["instrument"]
        self.grid = FiberGrid(instrument)
        self.min_exposure = int(instrument["exposure"]["min_duration_seconds"])
        self.max_exposure = int(instrument["exposure"]["max_duration_seconds"])
        score = init["scoring"]
        self.f0t0 = float(score["flux_zero_point"]) * float(score["exposure_zero_point_seconds"])
        self.q0 = float(score["q0"])
        self.airmass_exponent = float(score["airmass_exponent"])
        self.bands = score["program"]["bands"]
        self.multipliers = score["program"]["multipliers"]
        self.mismatch = float(score["program"]["mismatch_multiplier"])
        self.lunar_model = score["lunar_model"]

        columns = init["targets"]["columns"]
        col = {name: columns.index(name) for name in columns}
        rows = init["targets"]["rows"]
        self.ids = [row[col["target_id"]] for row in rows]
        self.index_of = {target_id: i for i, target_id in enumerate(self.ids)}
        self.ra = [float(row[col["ra_deg"]]) for row in rows]
        self.dec = [float(row[col["dec_deg"]]) for row in rows]
        self.flux = [float(row[col["feature_flux"]]) for row in rows]
        self.weight = [float(row[col["science_weight"]]) for row in rows]
        self.required = [bool(row[col["required"]]) for row in rows]
        self.anchor_count = ANCHORS if len(self.nights) >= ANCHORS_MIN_NIGHTS else 3
        self.hmax = [max_hour_angle_deg(d, self.lat, self.min_alt + ALT_MARGIN_DEG) for d in self.dec]
        self.factor = [0.0] * len(rows)          # best estimated exposure factor so far
        self.misses = [0] * len(rows)            # assigned but not hit (e.g. too close to a fibre edge)
        self.attempts = [0] * len(rows)          # required hits that still ended below factor 0.5
        self.active = [i for i in range(len(rows)) if self.hmax[i] > 0.0]
        self._build_index()
        self._build_windows()

        self.scale = 1.0                         # learned sky quality relative to the clear-sky model
        self.era_ratchet = None                  # best sustained clean-sky level seen so far (fault reference)
        self.pending_quake_hours = -1e9          # a quake notice that hit HEALTHY sky starts the fault-evidence cooldown
        self.prior_scale = 1.0                   # long-run median, used when recent samples are missing
        self.samples: deque = deque(maxlen=24)   # (hours, ratio) of recent unsaturated hits
        self.all_ratios: deque = deque(maxlen=400)
        self.clean_history: list[tuple[float, int, float]] = []   # (hours since start, night, quality ratio)
        self.pending_night = -1
        self.band_checks: deque = deque(maxlen=60)    # (program declared, matched?, model) from saturated hits
        self.force_program = None                # set by the agent to run a diagnostic exposure
        self.pending: dict[str, dict] = {}       # target_id -> prediction for the observe in flight
        self.pending_program = "BACKUP"
        self.pending_duration = 0
        self.blocked: list[tuple[float, float]] = []          # (az, alt) where a hit scored zero
        self.notices: set[tuple[str, str]] = set()
        self.terrain: set[str] = set()
        self.extra_avoid: set[str] = set()       # directions an advisor asked to avoid tonight
        self._calib_samples: list[tuple[float, float, int, bool]] = []   # (dn, de, fiber, hit)
        self._calib_since = 0
        self.pointing_corr: tuple[float, float] | None = None   # estimated (d_north, d_east) of actual-vs-commanded
        # env 旋钮是实验入口（P1-1 时长-覆盖扫描）；LLM/规则每夜重置到这个 base
        try:
            self.base_duration_scale = max(0.3, min(2.0, float(os.environ.get("AURORA_DURATION_SCALE", "1.0") or 1.0)))
        except ValueError:
            self.base_duration_scale = 1.0
        self.duration_scale = self.base_duration_scale
        self.fast_level = 0
        self.band_width = float(score["uniformity"]["ra_band_width_deg"])
        self.band_totals: dict[int, int] = {}
        for r in self.ra:
            b = int(r // self.band_width)
            self.band_totals[b] = self.band_totals.get(b, 0) + 1
        self.band_half = {b: 0 for b in self.band_totals}

    # --- precomputation --------------------------------------------------------------------------

    def _build_index(self) -> None:
        self.cells: dict[int, list[tuple[float, int]]] = {}
        for i in self.active:
            self.cells.setdefault(int(math.floor(self.dec[i])), []).append((self.ra[i], i))
        for band in self.cells.values():
            band.sort()
        self.cell_ras = {key: [ra for ra, _ in band] for key, band in self.cells.items()}

    def neighbours(self, ra: float, dec: float, radius: float) -> list[int]:
        found = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = radius / cos_dec
        for key in range(int(math.floor(dec - radius)), int(math.floor(dec + radius)) + 1):
            band = self.cells.get(key)
            if not band:
                continue
            ras = self.cell_ras[key]
            spans = [(ra - width, ra + width)]
            if spans[0][0] < 0:
                spans = [(0.0, spans[0][1]), (spans[0][0] + 360.0, 360.0)]
            elif spans[0][1] >= 360:
                spans = [(spans[0][0], 360.0), (0.0, spans[0][1] - 360.0)]
            for low, high in spans:
                for k in range(bisect.bisect_left(ras, low), bisect.bisect_right(ras, high)):
                    found.append(band[k][1])
        return found

    def _build_windows(self) -> None:
        """First and last night on which each target has at least 20 minutes above the limit."""
        need = 20 * 60 * SIDEREAL_DEG_PER_SECOND
        spans = [(local_sidereal_deg(start, self.lon), (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND)
                 for start, end in self.nights]
        self.first_night = [len(self.nights)] * len(self.ra)
        self.last_night = [-1] * len(self.ra)
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = (self.ra[i] - h - l0) % 360.0
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if self.first_night[i] > k:
                        self.first_night[i] = k
                    self.last_night[i] = k

    # --- messages and results --------------------------------------------------------------------

    def on_messages(self, messages: list, latest_bulletin) -> None:
        for message in messages:
            kind = message.get("record_type")
            if kind == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        self.terrain.add(notice.get("direction", ""))
            elif kind == "state_resync":
                self._resync(message)
        bulletin = latest_bulletin or {}
        # earthquake notices do NOT dirty samples: quake dips decay and the fault detector's
        # ratchet/slope logic interprets them; discarding those nights only punches holes in
        # the evidence window exactly when a fault needs to be seen.
        self.notices = {(n.get("event_kind", ""), n.get("direction", "")) for n in bulletin.get("notices", [])
                        if n.get("event_kind") not in ("terrain_obstruction", "earthquake")}

    def _resync(self, message: dict) -> None:
        """Part of the recent data was lost: restart the factor estimates from the engine's best scores."""
        best = {row["target_id"]: float(row["best_score"]) for row in message.get("best_scores", [])}
        top = max(self.multipliers.values())
        for i, target_id in enumerate(self.ids):
            score = best.get(target_id, 0.0)
            self.factor[i] = min(1.0, score / (self.weight[i] * top)) if score > 0 else 0.0
        self.active = [i for i in range(len(self.ids)) if self.hmax[i] > 0.0]
        self.band_half = {}
        for i, f in enumerate(self.factor):
            if f >= 0.5:
                b = int(self.ra[i] // self.band_width)
                self.band_half[b] = self.band_half.get(b, 0) + 1
        self.pending = {}
        self.log(f"state_resync: {len(best)} targets keep a score; plan rebuilt")

    def site_closed(self) -> bool:
        return any(kind in CLOSED_KINDS and direction == "ALL" for kind, direction in self.notices)

    def all_sky_notice(self) -> bool:
        return any(direction == "ALL" for _, direction in self.notices)

    def on_result(self, result, now: datetime, hours: float) -> None:
        """Update factor estimates and the sky-quality estimate from the previous observe."""
        if not result or result.get("action") != "observe" or not self.pending:
            self.pending = {}
            return
        hits = {hit["target_id"]: float(hit["score"]) for hit in result.get("hits", [])}
        any_positive = any(score > 0 for score in hits.values())
        declared = self.multipliers[self.pending_program]
        for target_id, prediction in self.pending.items():
            i = self.index_of[target_id]
            score = hits.get(target_id)
            if score is None:
                self.misses[i] += 1  # a miss: the target did not land on its fibre glass
                pred = self.pending.get(target_id) or {}
                if "fiber" in pred:
                    self._calib_samples.append((pred["dn"], pred["de"], pred["fiber"], False))
                    self._calib_since += 1
                continue
            if score <= 0.0:
                if any_positive:
                    self.blocked.append((prediction["az"], prediction["alt"]))
                continue
            # score = weight * factor * multiplier; the multiplier is `declared` if the program matched.
            # A saturated hit (factor = 1) shows the multiplier exactly, so it tells whether the sky's
            # program band matched the declared program. Instrument efficiency does not enter the band.
            multiplier_seen = score / self.weight[i]
            if prediction["clean"]:
                if abs(multiplier_seen - declared) < 2e-4:
                    self.band_checks.append((self.pending_program, True, prediction["model"]))
                elif abs(multiplier_seen - self.mismatch) < 2e-4:
                    self.band_checks.append((self.pending_program, False, prediction["model"]))
            factor_if_match = score / (self.weight[i] * declared)
            factor_if_miss = score / (self.weight[i] * self.mismatch)
            ratio_match = factor_if_match * self.f0t0 / (self.flux[i] * self.pending_duration * prediction["model"])
            matched = self._band(ratio_match * prediction["band_model"]) == self.pending_program
            factor = factor_if_match if matched else factor_if_miss
            before = self.factor[i]
            self.factor[i] = max(self.factor[i], min(1.0, factor))
            pred = self.pending.get(target_id) or {}
            if "fiber" in pred:
                self._calib_samples.append((pred["dn"], pred["de"], pred["fiber"], True))
                self._calib_since += 1
            if before < 0.5 <= self.factor[i]:
                b = int(self.ra[i] // self.band_width)
                self.band_half[b] = self.band_half.get(b, 0) + 1
            if self.required[i] and self.factor[i] < 0.5:
                self.attempts[i] += 1  # not enough yet: lower its priority a little for next time
            if factor < 0.97:
                ratio = factor * self.f0t0 / (self.flux[i] * self.pending_duration * prediction["model"])
                self.samples.append((hours, ratio))
                self.all_ratios.append(ratio)
                if prediction["clean"]:
                    self.clean_history.append((hours, self.pending_night, ratio))
        self.pending = {}
        self.update_scale(hours)
    def update_scale(self, hours: float) -> None:
        """Sky quality now = median of recent samples; fall back to the long-run median when stale."""
        if len(self.all_ratios) >= 8:
            ordered = sorted(self.all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self.samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if len(recent) >= 4 else self.prior_scale

    def _band(self, q_band: float) -> str:
        if q_band >= float(self.bands["DARK"]):
            return "DARK"
        if q_band >= float(self.bands["BRIGHT"]):
            return "BRIGHT"
        return "BACKUP"

    # --- anomaly check ---------------------------------------------------------------------------

    def fault_evidence(self, hours: float) -> dict | None:
        """A fault is a step down in instrument efficiency; weather dips and earthquakes decay or
        recover. So compare the recent clean-sky median against the LONG-RUN median (all_ratios):
        a sustained step shows up as a stable drop, while decaying noise pulls the reference down
        with it. Requires the recent window to span at least two nights and 1.5 hours."""
        history = self.clean_history
        if len(history) < RECENT_SAMPLES + EARLIER_SAMPLES or len(self.all_ratios) < 32:
            return None
        # Announced earthquakes depress efficiency and decay over nights. A notice that hit
        # HEALTHY sky marks a fresh impact: evidence may only use samples after its cooldown.
        # Notices arriving while the sky is already depressed are afterglow and never re-arm.
        cutoff = self.pending_quake_hours + 48.0
        usable = [rec for rec in history if rec[0] >= cutoff]
        if len(usable) < RECENT_SAMPLES + EARLIER_SAMPLES:
            return None
        recent = usable[-RECENT_SAMPLES:]
        span = recent[-1][0] - recent[0][0]
        nights = len({night for _, night, _ in recent})
        if span < 1.5 or nights < 2:
            return None
        recent_median = sorted(r for _, _, r in recent)[len(recent) // 2]
        # recovering sky (quake residual / weather healing) rises night over night; a fault step
        # stays flat. A clear upward slope within the recent window vetoes the fault hypothesis.
        half = max(4, len(recent) // 2)
        first_half = sorted(r for _, _, r in recent[:-half])[len(recent[:-half]) // 2]
        second_half = sorted(r for _, _, r in recent[-half:])[len(recent[-half:]) // 2]
        if second_half > first_half * 1.08:
            return None
        # ratchet reference: the best sustained 48-sample window median ever seen. Weather drifts a
        # few percent; a fault is a -40..-50% step, so the ratchet separates them cleanly and
        # survives quake epochs (forget_quality_history must not reset it). Empirically beats a
        # trailing-week p75, which lets quake tails refill the reference and silence the detector.
        window = usable[-(RECENT_SAMPLES + EARLIER_SAMPLES):]
        era = sorted(r for _, _, r in window)[len(window) // 2]
        if self.era_ratchet is None or era > self.era_ratchet:
            self.era_ratchet = era
        reference = self.era_ratchet
        dark_line = float(self.bands["DARK"]) * 1.3
        # reference-independent filter: "predicted clearly DARK under decent sky". Scaling by the
        # current level empties the check during faults (the band excludes efficiency) and would
        # loop the DARK diagnostic forever.
        dark = [matched for program, matched, model in self.band_checks
                if program == "DARK" and model >= dark_line * 0.95][-16:]
        return {"recent_median": round(recent_median, 3), "earlier_median": round(reference, 3),
                "drop": round(recent_median / max(1e-9, reference), 3), "recent_samples": len(recent),
                "recent_nights": nights, "earlier_samples": len(window),
                "dark_checks": len(dark), "dark_matched": sum(dark)}

    def sky_depressed(self, hours: float, margin: float = 0.92) -> bool:
        """Recent clean-sky level meaningfully below the era ratchet (quake impact active)."""
        if not self.era_ratchet:
            return False
        fresh = [r for when, _, r in self.clean_history[-8:] if when >= hours - 24.0]
        if len(fresh) < 3:
            return False
        return sorted(fresh)[len(fresh) // 2] < self.era_ratchet * margin

    def _calibrate_pointing(self) -> None:
        """Estimate a fixed commanded-vs-actual pointing offset from fibre hit/miss geometry.

        For candidate offset (dn, de): a target's predicted local position is its commanded
        position minus the offset; hits must land inside the assigned glass (score +1),
        misses must fall outside it (score +1 when consistent). Grid search, then activate
        only on strong, sizeable evidence (a real Hard-mode offset lifts consistency a lot)."""
        if self.pointing_corr is not None or len(self._calib_samples) < CALIB_MIN_SAMPLES:
            return
        samples = self._calib_samples[-400:]
        best_off, best_score = (0.0, 0.0), None
        steps = int(CALIB_MAX_DEG / CALIB_STEP_DEG)
        n = steps * 2 + 1
        for a in range(n):
            dn = (a - steps) * CALIB_STEP_DEG
            for b in range(n):
                de = (b - steps) * CALIB_STEP_DEG
                score = 0.0
                for dn0, de0, fiber, hit in samples:
                    row, col = divmod(fiber, 4)
                    half = 2.0 * 0.632456 / 2.0
                    lo_n = (row - 2) * 0.632456
                    lo_e = (col - 2) * 0.632456
                    inside = (lo_n <= dn0 - dn < lo_n + 0.632456) and (lo_e <= de0 - de < lo_e + 0.632456)
                    score += 1.0 if inside == hit else -1.0
                if best_score is None or score > best_score:
                    best_score, best_off = score, (dn, de)
        zero_score = sum(1.0 if (((int(f / 4) - 2) * 0.632456 <= dn0 < (int(f / 4) - 1) * 0.632456)
                                 and ((f % 4 - 2) * 0.632456 <= de0 < (f % 4 - 1) * 0.632456)) == hit else -1.0
                         for dn0, de0, f, hit in samples)
        mag = math.hypot(*best_off)
        gain = (best_score - zero_score) / max(1, len(samples))
        if mag >= CALIB_MIN_MAG and gain >= CALIB_MIN_GAIN:
            self.pointing_corr = best_off
            self.log(f"pointing offset estimated ({best_off[0]:+.3f}, {best_off[1]:+.3f}) deg, "
                     f"consistency gain {gain:.2f} over {len(samples)} samples")

    def forget_quality_history(self) -> None:
        """After a report, start the quality estimates afresh (the level may change)."""
        self.clean_history = []
        self.band_checks.clear()
        self.samples.clear()
        self.all_ratios.clear()
        self.prior_scale = 1.0

    # --- planning --------------------------------------------------------------------------------

    def current_night(self, now: datetime):
        for k, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return k, start, end
        return None

    def next_night_start(self, now: datetime):
        for start, _ in self.nights:
            if start > now:
                return start
        return None

    def _direction_factor(self, alt: float, az: float) -> float:
        factor = 1.0
        for direction in self.terrain:
            if direction in DIRECTION_AZ and alt < 50.0 and _az_distance(az, DIRECTION_AZ[direction]) <= 60.0:
                return 0.0
        for kind, direction in self.notices:
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, 0.35)
        for direction in self.extra_avoid:
            if direction in DIRECTION_AZ and _az_distance(az, DIRECTION_AZ[direction]) <= 67.5 and alt < 70.0:
                factor = min(factor, 0.35)
        for blocked_az, blocked_alt in self.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12.0 and alt <= blocked_alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def value(self, i: int) -> float:
        f = self.factor[i]
        damp = 0.6 ** self.misses[i]
        if self.required[i]:
            if f >= REQUIRED_SAFE_FACTOR:
                return self.weight[i] * max(0.0, 1.0 - f) * damp
            return (self.weight[i] * (1.0 - f) + REQUIRED_BONUS * (1.0 if f < 0.5 else 0.35)) * damp
        return 0.0 if f >= DONE_FACTOR else self.weight[i] * (1.0 - f) * damp

    def plan(self, now: datetime, night_end: datetime, night_index: int, hours: float):
        """Return an observe action dict, or None when nothing useful is up."""
        self.update_scale(hours)
        self.night_index = night_index
        if self._calib_since >= CALIB_REESTIMATE or (self.pointing_corr is None
                and len(self._calib_samples) >= CALIB_MIN_SAMPLES and self._calib_since == 0):
            self._calibrate_pointing()
            self._calib_since = 0
        lst = local_sidereal_deg(now, self.lon)
        horizon = min(night_end, self.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < self.min_exposure:
            return None
        min_visible = min(MIN_VISIBLE_SECONDS, seconds_left) * SIDEREAL_DEG_PER_SECOND
        # 1. visible, not-done targets (hour-angle test: cheap, no trigonometry)
        boosts = {}
        if self.band_totals:
            ratios = {b: self.band_half.get(b, 0) / n for b, n in self.band_totals.items()}
            mean_r = sum(ratios.values()) / len(ratios)
            boosts = {b: 1.0 + UNIF_BOOST * max(0.0, mean_r - r) for b, r in ratios.items()}
        still_active = []
        candidates = []
        for i in self.active:
            v = self.value(i)
            if v <= 0.0:
                continue
            still_active.append(i)
            ha = wrap180(lst - self.ra[i])
            h = self.hmax[i]
            if -h <= ha and ha + min_visible <= h:
                nights_left = max(1, self.last_night[i] - night_index + 1)
                setting = 1.0 + 0.5 * max(0.0, ha / h) if h < 180 else 1.0
                candidates.append((v * boosts.get(int(self.ra[i] // self.band_width), 1.0)
                                   * (1.0 + 2.0 / nights_left) * setting, i))
        self.active = still_active
        if not candidates:
            return None
        candidates.sort(reverse=True)
        moon = Moon(now + timedelta(seconds=450), lst, self.lat, self.lunar_model)
        altaz_cache: dict[int, tuple[float, float]] = {}

        def altaz(i: int) -> tuple[float, float]:
            if i not in altaz_cache:
                altaz_cache[i] = radec_to_altaz(self.ra[i], self.dec[i], lst, self.lat)
            return altaz_cache[i]

        visible = {i for _, i in candidates}
        achievable_cache: dict[int, float] = {}

        def achievable(i: int) -> float:
            """Value this target can still gain tonight with the longest exposure it allows.
            A required target whose factor cannot reach 0.5 now brings no bonus now; try it later."""
            if i not in achievable_cache:
                alt, az = altaz(i)
                model = moon.lunar_factor(self.ra[i], self.dec[i]) / (
                    self.q0 * normalized_airmass(max(alt, 1.0)) ** self.airmass_exponent)
                k = self.flux[i] * model * self.scale * PLAN_FACTOR_SAFETY / self.f0t0
                up = (self.hmax[i] - wrap180(lst - self.ra[i])) / SIDEREAL_DEG_PER_SECOND if self.hmax[i] < 180 else 1e9
                reach = min(1.0, k * min(self.max_exposure, up, seconds_left))
                f = self.factor[i]
                gain = self.weight[i] * max(0.0, min(1.0, reach) - f)
                if self.required[i] and f < 0.5 and reach >= 0.5:
                    gain += REQUIRED_BONUS
                damp = 0.6 ** self.misses[i] * 0.7 ** self.attempts[i]
                achievable_cache[i] = gain * damp * self._direction_factor(alt, az)
            return achievable_cache[i]

        anchors = []
        for checked, (priority, i) in enumerate(candidates):
            if checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                break
            weighted = achievable(i) * priority / max(1e-9, self.value(i))  # keep the urgency terms
            if weighted > 0:
                anchors.append((weighted, i))
        if not anchors:
            return None
        anchors.sort(reverse=True)
        n_anchors = 1 if self.fast_level >= 1 else self.anchor_count
        fibers = range(self.grid.n) if self.fast_level < 2 else (5, 6, 9, 10)
        best = None
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and best is not None:
                break
            if tried >= n_anchors + 8:
                break  # nothing places well right now
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in self.neighbours(self.ra[anchor], self.dec[anchor], NEIGHBOUR_RADIUS_DEG) if j in visible]
            near_values = {j: achievable(j) for j in near}
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not self.min_alt + 1.5 <= c_alt <= 89.0:
                    continue
                c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}
                for j, v in near_values.items():
                    if v <= 0.0:
                        continue
                    alt, az = altaz(j)
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    # after a miss, only trust placements well inside the glass
                    score = v * (1.0 if margin >= EDGE_MARGIN_DEG * (1 + 1.5 * self.misses[j]) else 0.4)
                    if fib not in chosen or score > chosen[fib][0]:
                        chosen[fib] = (score, j, margin)
                if not chosen:
                    continue
                total = sum(item[0] for item in chosen.values())
                if best is None or total > best[0]:
                    best = (total, c_alt, c_az, chosen)
        if best is None:
            return None
        _, c_alt, c_az, chosen = best
        if self.pointing_corr is not None:
            c_alt, c_az = shift_altaz(c_alt, c_az, -self.pointing_corr[0], -self.pointing_corr[1])
            c_alt, c_az = round(c_alt, 4), round(c_az, 4) % 360.0
        return self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours)

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours):
        """Pick the exposure length and program for the chosen pointing."""
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, self.lat)
        c_hmax = max_hour_angle_deg(c_dec, self.lat, self.min_alt + 0.3)
        c_ha = wrap180(lst - c_ra)
        info = {}
        for fiber, (_, j, _) in chosen.items():
            alt, az = altaz(j)
            lunar = moon.lunar_factor(self.ra[j], self.dec[j])
            model = lunar / (self.q0 * normalized_airmass(max(alt, 1.0)) ** self.airmass_exponent)
            # seconds this target stays above the limit
            up = (self.hmax[j] - wrap180(lst - self.ra[j])) / SIDEREAL_DEG_PER_SECOND
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up,
                           "k": self.flux[j] * model * self.scale * PLAN_FACTOR_SAFETY / self.f0t0}
        center_up = (c_hmax - c_ha) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9
        best = None
        for base in DURATIONS:
            duration = int(round(base * self.duration_scale / 30.0) * 30)
            duration = max(self.min_exposure, min(self.max_exposure, duration))
            if duration > seconds_left or duration > center_up:
                continue
            gain = 0.0
            for item in info.values():
                if item["up"] < duration:
                    continue
                i = item["i"]
                reached = min(1.0, item["k"] * duration)
                f = self.factor[i]
                gain += self.weight[i] * max(0.0, min(1.0, reached) - f)
                if self.required[i] and f < 0.5 and reached >= 0.5:
                    gain += REQUIRED_BONUS
            rate = gain / duration
            if best is None or rate > best[0]:
                best = (rate, duration)
        if best is None:
            return None
        duration = best[1]
        if best[0] <= 0.0:
            if any(when >= hours - SKY_MEMORY_HOURS for when, _ in self.samples):
                return None  # the estimate is fresh and says nothing improves here
            # The sky estimate is stale: take one normal exposure to measure it again.
            duration = next((d for d in (900, 600, 300) if d <= seconds_left and d <= center_up), None)
            if duration is None:
                return None
        assignments = {str(fiber): self.ids[item["i"]] for fiber, item in info.items() if item["up"] >= duration}
        # program: the band most of the expected score falls into
        band_scale = self.scale / 0.95
        votes = {"DARK": 0.0, "BRIGHT": 0.0, "BACKUP": 0.0}
        for fiber, item in info.items():
            if str(fiber) not in assignments:
                continue
            band = self._band(item["model"] * band_scale)
            i = item["i"]
            votes[band] += self.weight[i] * min(1.0, item["k"] * duration) + (REQUIRED_BONUS * 0.02 if self.required[i] else 0.0)
        program = max(votes, key=lambda name: (votes[name] * self.multipliers[name]
                                               + sum(v for k, v in votes.items() if k != name) * self.mismatch, name))
        if self.force_program:
            program = self.force_program
        clean = not self.all_sky_notice()
        self.pending = {}
        for fiber, item in info.items():
            if str(fiber) in assignments:
                off = tangent_offsets(item["alt"], item["az"], c_alt, c_az) or (0.0, 0.0)
                self.pending[self.ids[item["i"]]] = {
                    "model": item["model"], "band_model": item["model"] / 0.95, "alt": item["alt"], "az": item["az"],
                    "dn": off[0], "de": off[1], "fiber": int(fiber),
                    "clean": clean and self._direction_factor(item["alt"], item["az"]) >= 1.0}
        self.pending_program = program
        self.pending_duration = duration
        self.pending_night = self.night_index
        return {"action": "observe", "pointing": {"alt_deg": c_alt, "az_deg": c_az}, "assignments": assignments,
                "duration_seconds": duration, "program": program}
