"""Public-source T-10 observation, prospective only with proven F3 model and history.

Preflight is NOT a race lock. Live lock must occur at 15:30..15:31 JST
on November 1, 2026 with actual source-observed live winning odds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from keiba_place_lab.live_public import parse_roster, parse_win_snapshot
from keiba_place_lab.market import (
    harville_top_k_probabilities,
    normalized_win_probabilities,
)
from keiba_place_lab.tennosho_forward import (
    locked_bundle,
    predict_from_frozen,
    sha256,
)

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_CONFIG = Path("config/tennosho_preoff_20261101.json")
PREOFF_SECONDS_START = 600
PREOFF_SECONDS_END = 540
FRESH_SECONDS = 300


def ensure_lock_clock(off: datetime, now: datetime) -> None:
    if off.tzinfo is None or now.tzinfo is None:
        raise ValueError("Aware actual clock required; cannot invent pre-off time")
    seconds = (off - now).total_seconds()
    if not (PREOFF_SECONDS_END < seconds <= PREOFF_SECONDS_START):
        raise ValueError(
            "Genuine T-minus-10 lock requires current clock in 15:30..15:31"
        )


def ensure_provider_clock(
    provider: datetime, acquired: datetime, lock_at: datetime, off: datetime,
) -> None:
    for ts in (provider, acquired, lock_at, off):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("Market observation timestamps must be timezone-aware")
    if not provider <= acquired <= lock_at < off:
        raise ValueError("Provider/capture/lock chronological order invalid")
    if not 0 <= (lock_at - provider).total_seconds() <= FRESH_SECONDS:
        raise ValueError("Actual provider update is older than five minutes")
    if not 0 <= (lock_at - acquired).total_seconds() <= 60:
        raise ValueError("Capture was not made inside the actual T-10 minute")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_html(url: str) -> tuple[str, dict]:
    acquired_at = datetime.now(UTC)
    response = requests.get(
        url, timeout=(10, 12),
        headers={
            "User-Agent": "keiba-place-probability-lab/0.1 (+noncommercial reproducible research)",
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )
    response.raise_for_status()
    captured = datetime.now(UTC)
    html = response.text
    if len(html) < 1000:
        raise ValueError("Public roster/odds body is absent or blocked")
    meta = {
        "url": url,
        "request_started_utc": acquired_at.isoformat(),
        "captured_at_utc": captured.isoformat(),
        "html_sha256": sha_bytes(response.content),
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "cache_age_header": response.headers.get("Age"),
        "cache_control_header": response.headers.get("Cache-Control"),
        "provider_observation_not_http_capture": True,
    }
    return html, meta


def strict_roster(
    roster: pd.DataFrame, expected: dict, race_id: str,
) -> set[str]:
    horses = expected["horse_no_to_name"]
    if (len(roster) != expected["expected_starters"]
            or roster.horse_id.duplicated().any()
            or roster.horse_no.duplicated().any()
            or set(roster.race_id.astype(str)) != {race_id}):
        raise ValueError("Whole-field roster not exact; scratches need new frozen plan")
    actual = {
        str(r.horse_no): str(r.horse_name)
        for r in roster.itertuples(index=False)
    }
    if actual != horses:
        raise ValueError("Official pre-frozen horse-number/name roster changed")
    return set(roster.horse_id.astype(str))


def strict_same_market(
    roster: pd.DataFrame, odds: pd.DataFrame,
) -> pd.DataFrame:
    expected = roster[
        ["race_id", "horse_id", "horse_no", "horse_name"]
    ].copy()
    got = odds[
        ["race_id", "horse_id", "horse_no", "horse_name", "win_odds"]
    ].copy()
    for x in (expected, got):
        x["race_id"] = x.race_id.astype(str)
        x["horse_id"] = x.horse_id.astype(str)
        x["horse_no"] = x.horse_no.astype(int)
    pair = expected.merge(
        got, on=["race_id", "horse_id", "horse_no", "horse_name"],
        validate="one_to_one", how="outer", indicator=True,
    )
    if not pair._merge.eq("both").all() or len(pair) != len(roster):
        raise ValueError("Pre-off odds/roster mismatch or cancelled runner")
    if pair.win_odds.isna().any() or pair.horse_no.duplicated().any():
        raise ValueError("Missing win odds or duplicate draw position")
    return pair.drop(columns="_merge").sort_values(
        "horse_no", kind="stable",
    ).reset_index(drop=True)


def _json_write_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verified_provider_time(html: str, timestamp: datetime) -> dict:
    """Preserve publisher's full datetime literal and its JST interpretation.

    2026-10-31 09:19 JST actual Yahoo source preflight proved the provider
    emits a calendar-date ISO timestamp WITHOUT offset. The adapter's
    parse_win_snapshot already interprets that literal as Japan local time.
    Record this inference explicitly; never present it as source-supplied TZ.
    """
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one("time.hr-horseResult__update")
    if tag is None:
        raise ValueError("Displayed provider update timestamp is missing")
    value = tag.get("datetime")
    if not value:
        raise ValueError("Provider has no independently encoded update time")
    if "T" not in value or len(value) < 16:
        raise ValueError("Odds updated-at has no full ISO calendar date/time")
    stated = datetime.fromisoformat(value)
    if stated.tzinfo is None:
        stated = stated.replace(tzinfo=JST)
        origin = "provider_local_calendar_datetime_interpreted_as_Asia_Tokyo"
    else:
        origin = "provider_explicit_timezone"
    if stated.astimezone(UTC) != timestamp.astimezone(UTC):
        raise ValueError("Parsed update time disagrees with provider source")
    return {
        "provider_datetime_literal": value,
        "timezone_interpretation": origin,
        "interpreted_provider_updated_at": stated.isoformat(),
    }


def record_preflight(config: dict, directory: Path) -> dict:
    """Check live-site access NOW; do not pretend morning odds were at T-10."""
    if directory.exists():
        raise FileExistsError("Never overwrite a completed preflight")
    roster_html, roster_meta = fetch_html(config["roster_url"])
    odds_html, odds_meta = fetch_html(config["odds_url"])
    roster, _ = parse_roster(roster_html, config)
    strict_roster(roster, config, config["race_id"])
    odds, asof = parse_win_snapshot(odds_html, config)
    provider_clock = verified_provider_time(odds_html, asof)
    complete = strict_same_market(roster, odds)
    p, overround = normalized_win_probabilities(
        complete.win_odds.to_numpy(float),
    )
    market = harville_top_k_probabilities(p)
    if not np.isclose(market.sum(), 3.0, atol=1e-8):
        raise ValueError("Preflight full-field Harville sum not equal three")
    snap = {
        "status": "PREFLIGHT_ONLY_NOT_AN_ACTUAL_T_MINUS_10_LOCK",
        "race_id": config["race_id"],
        "asof_provider": asof.isoformat(),
        "provider_clock_provenance": provider_clock,
        "roster": roster_meta,
        "odds": odds_meta,
        "starters": len(roster),
        "market_conversion": "normalized win odds + Harville top3",
        "overround": float(overround),
        "no_results_used": True,
        "publication_not_proven": True,
        "candidate_predictions_not_made": True,
    }
    directory.mkdir(parents=True)
    _json_write_exclusive(directory / "preflight.json", snap)
    return snap


def capture(
    config: dict, directory: Path, *,
    bundle_file: Path | None = None,
    bundle_manifest: Path | None = None,
    prior_features: Path | None = None,
    prior_manifest: Path | None = None,
    formal_dir: Path | None = None,
) -> dict:
    """Actual UTC system clock gates before AND after both HTTP requests."""
    off = datetime.fromisoformat(config["off_at"])
    started = datetime.now(UTC)
    ensure_lock_clock(off, started)
    if directory.exists():
        raise FileExistsError("Observations/failed attempts are append-only")
    roster_html, roster_meta = fetch_html(config["roster_url"])
    odds_html, odds_meta = fetch_html(config["odds_url"])
    acquired = datetime.fromisoformat(odds_meta["captured_at_utc"])
    roster, _ = parse_roster(roster_html, config)
    strict_roster(roster, config, config["race_id"])
    odds, provider = parse_win_snapshot(odds_html, config)
    provider_clock = verified_provider_time(odds_html, provider)
    matched = strict_same_market(roster, odds)
    locked_at = datetime.now(UTC)
    ensure_lock_clock(off, locked_at)
    ensure_provider_clock(provider, acquired, locked_at, off)

    odds_source_hash = sha_bytes(odds_html.encode("utf-8"))
    market_win, overround = normalized_win_probabilities(
        matched.win_odds.to_numpy(float),
    )
    pmarket = harville_top_k_probabilities(market_win)
    observed = matched[["race_id", "horse_id", "horse_no", "horse_name"]].copy()
    observed["p_M0_live_raw"] = pmarket
    observed = observed.sort_values("horse_no", kind="stable").reset_index(drop=True)
    directory.mkdir(parents=True)
    observed.to_csv(
        directory / "market_observed.csv", index=False, float_format="%.12g",
    )
    manifest = {
        "status": "OBSERVATION_ONLY_NOT_COMPLETE_MODEL_COMPARISON",
        "protocol": "tennosho_tminus10_prospective_v1",
        "race_id": config["race_id"],
        "off_at": off.isoformat(),
        "capture_started_at": started.isoformat(),
        "captured_at": acquired.isoformat(),
        "provider_updated_at": provider.isoformat(),
        "provider_clock_provenance": provider_clock,
        "locked_at": locked_at.isoformat(),
        "full_active_field": len(matched),
        "active_field_horse_numbers": observed.horse_no.to_list(),
        "active_field_roster_sha256": sha_bytes(
            observed[["race_id", "horse_id", "horse_no", "horse_name"]].to_csv(
                index=False, lineterminator="\n",
            ).encode(),
        ),
        "odds_source_html_sha256_only_not_redistributed": odds_source_hash,
        "source": {"roster": roster_meta, "market": odds_meta},
        "market_observation_sha256": sha256(directory / "market_observed.csv"),
        "conversion": "all-starter win odds normalized + Harville top3",
        "overround": float(overround),
        "target_outcome_known_at_capture": False,
        "formal_challenger_lock": False,
        "publication_to_main_verified": False,
        "production_changed": False,
    }
    _json_write_exclusive(directory / "observation.json", manifest)

    provided = [bundle_file, bundle_manifest, prior_features, prior_manifest]
    if any(value is None for value in provided):
        if formal_dir is not None:
            raise ValueError(
                "No formal B_raw/XGB800/CAT400 lock without frozen model "
                "AND verified pre-off historical F3 features"
            )
        return manifest
    assert bundle_file is not None
    assert bundle_manifest is not None
    assert prior_features is not None
    assert prior_manifest is not None
    if sha256(prior_features) != json.loads(
        prior_manifest.read_text(),
    ).get("feature_csv_sha256"):
        raise ValueError("Pre-audited F3 feature file hash missing/mismatched")
    prior = pd.read_csv(
        prior_features, dtype={"race_id": str, "horse_id": str},
    )
    past = json.loads(prior_manifest.read_text())
    bundle, model_info = locked_bundle(bundle_file, bundle_manifest)
    predictions, proof = predict_from_frozen(
        bundle, model_info, prior, past, matched,
    )
    if formal_dir is None:
        raise ValueError("Complete model prediction needs a unique formal lock path")
    if formal_dir.exists():
        raise FileExistsError("Formal model predictions are immutable")
    # Computation and I/O must ALSO finish in the fixed T-10..T-9 window.
    ensure_lock_clock(off, datetime.now(UTC))
    formal_dir.mkdir(parents=True)
    predictions.to_csv(
        formal_dir / "predictions.csv", index=False, float_format="%.12g",
    )
    verdict = {
        **manifest,
        **proof,
        "status": "MODEL_PREDICTIONS_CREATED_PUBLICATION_PENDING",
        "formal_challenger_lock": True,
        "predictions_sha256": sha256(formal_dir / "predictions.csv"),
        "features_csv_sha256": sha256(prior_features),
        "source_observation_sha256": sha256(directory / "market_observed.csv"),
        "model_manifest_sha256": sha256(bundle_manifest),
        "feature_manifest_sha256": sha256(prior_manifest),
        "protocol_spec_sha256": sha256(
            Path(config["spec"]),
        ),
        "published_to_main_before_off": False,
        "publication_to_main_verified": False,
    }
    _json_write_exclusive(formal_dir / "manifest.json", verdict)
    manifest["status"] = "OBSERVED_AND_MODEL_LOCK_CREATED_PUBLICATION_PENDING"
    # Append-only sidecar, no edits to previously written observation manifest.
    _json_write_exclusive(directory / "candidate_created.json", {
        "formal_path": str(formal_dir),
        "predictions_sha256": verdict["predictions_sha256"],
        "publication_pending": True,
    })
    return verdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("preflight", "capture"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--bundle-manifest", type=Path)
    p.add_argument("--prior-features", type=Path)
    p.add_argument("--prior-manifest", type=Path)
    p.add_argument("--formal-out", type=Path)
    a = p.parse_args()
    conf = json.loads(a.config.read_text())
    if a.mode == "preflight":
        if any((a.bundle, a.bundle_manifest, a.prior_features, a.prior_manifest,
                a.formal_out)):
            raise ValueError("Preflight is access-only and cannot generate locks")
        result = record_preflight(conf, a.output)
    else:
        result = capture(
            conf, a.output, bundle_file=a.bundle,
            bundle_manifest=a.bundle_manifest,
            prior_features=a.prior_features,
            prior_manifest=a.prior_manifest, formal_dir=a.formal_out,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
