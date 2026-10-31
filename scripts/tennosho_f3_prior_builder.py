"""Recompute 17 complete prior-only F3 rows using the ORIGINAL F3 source rules.

Never replace missing raw 2026 data with synthetic values. The frozen model
requires all JRA flat result runners/conditions to rebuild opponent Elo,
last-observed jockey/trainer rates and raw historical 3F/wet/mass statistics.
The model's SAME-RACE market fields are NOT part of this output: they are
calculated only from truly observed T-10 odds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from run_information_expansion_v1 import SOURCE_SHA
from run_input_learning_a_v1 import feature_registry
from tennosho_f3_2025_identity_repair import repair_frozen_2025_identities
from tennosho_f3_raw_audit import raw_result, umanity_raw_result

from keiba_place_lab import live_umanity
from keiba_place_lab.historical_panel import REQUIRED_COLUMNS, build_historical_panel
from keiba_place_lab.information_expansion_v1 import (
    _parse_3f,
    _parse_mass,
    _parse_passing,
    add_historical_result_features,
    add_opponent_elo,
    raw_observations,
)
from keiba_place_lab.input_learning_v1 import (
    MARKET_FIELD,
    MARKET_SPLITS,
    full_starter_relative,
)
from keiba_place_lab.live_history import (
    FROZEN_HISTORY_SHA256,
    RESULT_URL,
    CachedFetcher,
    load_frozen_history,
)
from keiba_place_lab.live_public import parse_roster
from keiba_place_lab.nonmarket import engineer_features

TARGET = "202605041111"
TARGET_DATE = pd.Timestamp("2026-11-01")
ALLOWED_LATEST_SOURCE = pd.Timestamp("2026-10-30")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_f3_git_checkpoint(root: Path) -> tuple[dict, pd.DataFrame]:
    """Read only the immutable history material actually persisted in Git.

    The full live-history snapshot has entry-audit and schedule files, but the
    quota-free branch checkpoints solely its verified history Parquet and
    current.json. Source-backed all-runner and all-race completeness is checked
    again by the separately audited 2026 raw manifest in rebuild().
    Do not silently reconstruct absent source metadata or change the hash.
    """
    pointer = root / "current.json"
    if not pointer.is_file():
        raise FileNotFoundError("Verified 2026 checkpoint pointer is missing")
    current = json.loads(pointer.read_text())
    if current.get("schema_version") != "jra_flat_live_history_v1":
        raise ValueError("2026 checkpoint has unrecognized schema")
    snapshot_id = current.get("snapshot_id", "")
    if not (
        isinstance(snapshot_id, str)
        and snapshot_id.startswith("jra_flat_live_history_v1-20261030-")
        and ".." not in snapshot_id
        and "/" not in snapshot_id
    ):
        raise ValueError("2026 checkpoint identity or cutoff mismatch")
    if current.get("complete_through") != "2026-10-30":
        raise ValueError("2026 checkpoint did not freeze at original cutoff")
    source = root / "snapshots" / snapshot_id / "jra_flat_history.parquet"
    if not source.is_file() or sha(source) != current.get("history_sha256"):
        raise ValueError("2026 Git checkpoint Parquet missing or SHA-256 changed")
    history = pd.read_parquet(source)
    if history.empty or not {"race_date", "race_id", "horse_id"}.issubset(history.columns):
        raise ValueError("Verified 2026 race history missing fields or rows")
    dates = pd.to_datetime(history.race_date, errors="raise")
    if (
        dates.isna().any()
        or not dates.dt.year.eq(2026).all()
        or dates.max() > ALLOWED_LATEST_SOURCE
        or history.duplicated(["race_id", "horse_id"]).any()
        or (history.race_id.astype(str) == TARGET).any()
    ):
        raise ValueError("Future, target, duplicated or unverified historic races")
    return current, history


def _jra_internal_id(value: object) -> str:
    """Map verified numeric JRA IDs to the frozen training source namespace."""
    horse_id = str(value)
    numeric_id = horse_id.removeprefix("jra:")
    if not re.fullmatch(r"\d{10}", numeric_id):
        raise ValueError(f"Invalid verified JRA horse identity: {horse_id}")
    return f"jra:{numeric_id}"


def _jra_external_id(value: object) -> str:
    """Expose the 10-digit ID expected by the target protocol."""
    return _jra_internal_id(value)[4:]


def _standard_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Keep ONLY the fields understood by the pre-existing historical formula."""
    out = source.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out:
            raise ValueError(f"Official standardized historical source absent: {col}")
    return out[sorted(REQUIRED_COLUMNS)].copy()


def _agent_key(value: str) -> str:
    """Canonical comparison key; never collapse different named people."""
    return re.sub(
        r"[\s.・]", "", unicodedata.normalize("NFKC", str(value))
    ).casefold()


def _verify_umanity_jockey_profiles(
    original: pd.DataFrame, extra: pd.DataFrame,
    raw_manifest: dict, cache: Path,
) -> tuple[dict[str, str], list[dict]]:
    """Resolve otherwise-ambiguous abbreviations from published profile IDs.

    Use two distinct dated historical race/horse observations per alias when
    available, requiring the same Umanity jockey profile ID and full name.
    Do not fetch tomorrow's race or infer identities from name similarity.
    """
    historical = {_agent_key(z) for z in original.jockey_name.dropna().astype(str)}
    active = {
        _agent_key(z) for z in extra.loc[
            extra.source_provider.eq("yahoo"), "jockey_name"
        ].dropna().astype(str)
    }
    unknown = []
    for alias in extra.loc[
        extra.source_provider.eq("umanity"), "jockey_name"
    ].dropna().astype(str).unique():
        key = _agent_key(alias)
        active_hits = {k for k in active if k.startswith(key)}
        known_hits = {k for k in historical if k.startswith(key)}
        if key not in historical and key not in active and (
            len(active_hits) != 1 and len(known_hits) != 1
        ):
            unknown.append(alias)
    if not unknown:
        return {}, []
    source_records = {
        str(rec["race_id"]): rec
        for rec in raw_manifest["source_records"]
    }
    fetcher = CachedFetcher(cache, pause_seconds=.25, max_attempts=4)
    verified: dict[str, str] = {}
    audit: list[dict] = []
    for alias in sorted(unknown):
        rows = extra.loc[
            extra.source_provider.eq("umanity")
            & extra.jockey_name.eq(alias)
            & extra.race_id.astype(str).isin(source_records)
        ].sort_values(["race_date", "race_id", "horse_no"])
        if rows.empty:
            raise ValueError(f"No prior race evidence for unknown jockey {alias}")
        samples = [rows.iloc[0]]
        if len(rows) > 1 and str(rows.iloc[-1].race_id) != str(rows.iloc[0].race_id):
            samples.append(rows.iloc[-1])
        identities = set()
        observations = []
        for row in samples:
            rid = str(row.race_id)
            date = pd.Timestamp(row.race_date)
            if rid == TARGET or date > ALLOWED_LATEST_SOURCE:
                raise ValueError("Future/target race in jockey profile verification")
            record = source_records[rid]
            race_url = record["source_url"]
            parsed = urlparse(race_url)
            if (
                parsed.scheme != "https" or parsed.netloc != "umanity.jp"
                or parsed.path != "/racing/result.php"
                or parse_qs(parsed.query).get("race_id")
                != [live_umanity.source_id(rid, date)]
            ):
                raise ValueError("Non-verified historical Umanity race source")
            html, race_record = fetcher.fetch(race_url)
            soup, meta = live_umanity.metadata(html, rid, date)
            if meta["race_id"] != rid or meta["race_date"] != date:
                raise ValueError("Published jockey race metadata mismatch")
            tables = [t for t in soup.select("table") if t.select_one("td.td_order")]
            if len(tables) != 1:
                raise ValueError("Published jockey race result table not unique")
            matching = []
            for tr in tables[0].select("tr"):
                cells = tr.find_all("td", recursive=False)
                if len(cells) < 11:
                    continue
                if int(cells[1].get_text(strip=True)) != int(row.horse_no):
                    continue
                hid, _ = live_umanity.identity(cells[3])
                if _jra_internal_id(hid) != _jra_internal_id(row.horse_id):
                    raise ValueError("Published jockey race/horse identity mismatch")
                matching.append(cells)
            if len(matching) != 1:
                raise ValueError("Published jockey starter is not unique")
            jockey = matching[0][5].find("a", href=True)
            if jockey is None or _agent_key(jockey.get_text(strip=True)) != _agent_key(alias):
                raise ValueError("Published jockey abbreviation mismatch")
            profile_url = urljoin(race_url, jockey["href"])
            p = urlparse(profile_url)
            profile_id = parse_qs(p.query).get("jockey_id", [])
            if (
                p.scheme != "https" or p.netloc != "umanity.jp"
                or p.path != "/racing/database/jockey.php"
                or len(profile_id) != 1
                or re.fullmatch(r"\d{4,8}", profile_id[0]) is None
            ):
                raise ValueError("Published jockey profile identity URL invalid")
            profile_record = None
            profile_status = "full_name_verified"
            try:
                profile_html, profile_record = fetcher.fetch(profile_url)
            except requests.HTTPError as exc:
                if (
                    exc.response is None or exc.response.status_code != 404
                    or (alias, profile_id[0]) != ("小谷哲", "05697")
                ):
                    raise
                # This local jockey's archived Umanity profile is 404.
                # Preserve the exact race-published alias and its stable
                # profile ID, never invent an unverified expanded name.
                full = alias
                profile_status = "published_alias_and_profile_id_only_404"
            else:
                title = BeautifulSoup(profile_html, "html.parser").title
                match = re.match(
                    r"^(.+?)騎手(?:\s|[-－―]|$)",
                    title.get_text(" ", strip=True) if title else "",
                )
                if match is None:
                    raise ValueError("Published jockey full name not found on profile")
                full = match.group(1).strip()
            if not _agent_key(full).startswith(_agent_key(alias)):
                raise ValueError(f"Jockey profile contradicts abbreviation {alias}")
            identities.add((profile_id[0], _agent_key(full), full))
            observations.append({
                "race_id": rid, "horse_id": _jra_external_id(row.horse_id),
                "race_source_url": race_record.url,
                "race_source_sha256": race_record.sha256,
                "profile_url": profile_url,
                "profile_sha256": profile_record.sha256 if profile_record else None,
                "profile_id": profile_id[0],
                "profile_status": profile_status,
            })
        if len(identities) != 1:
            raise ValueError(f"Historical races disagree on jockey alias {alias}")
        _, _, full_name = next(iter(identities))
        verified[alias] = full_name
        audit.append({
            "abbreviation": alias, "verified_full_name": full_name,
            "independent_race_profile_observations": observations,
        })
        print(json.dumps({
            "verified_umanity_jockey_alias": alias, "full": full_name,
            "observations": len(observations),
        }, ensure_ascii=False), flush=True)
    return verified, audit


def _canonicalize_extra_agents(
    original: pd.DataFrame, extra: pd.DataFrame,
    verified_jockey_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Resolve alternate abbreviated agents without merging distinct people.

    A verified historical result for the exact (race_id, horse_id) takes
    precedence over a global name-prefix lookup. This matters for current
    agents missing from 2010-2025 and for two different people sharing a
    published abbreviation. If that same-starter identity contradicts the
    alternate publication, fail closed rather than guessing an agent.
    """
    verified_jockey_aliases = verified_jockey_aliases or {}
    x = extra.copy()
    if not x.index.is_unique:
        raise ValueError("Duplicate raw result dataframe index")
    if "source_provider" not in x:
        x["source_provider"] = "yahoo"
    x["source_provider"] = x["source_provider"].fillna("yahoo")
    if not x.source_provider.isin(("yahoo", "umanity")).all():
        raise ValueError("Unknown raw agent identity source")
    audit: dict[str, object] = {}
    has_starter_keys = {"race_id", "horse_id"}.issubset(original.columns) and (
        {"race_id", "horse_id"}.issubset(x.columns)
    )
    for col in ("jockey_name", "trainer"):
        base_names = original[col].dropna().astype(str)
        historical = Counter(z for z in base_names if z.strip())
        exact: dict[str, str] = {}
        for name, _ in historical.most_common():
            exact.setdefault(_agent_key(name), name)
        current = Counter(
            x.loc[x.source_provider.eq("yahoo"), col].dropna().astype(str)
        )
        active_keys: dict[str, str] = {}
        for name, _ in current.most_common():
            if name.strip():
                active_keys.setdefault(
                    _agent_key(name), exact.get(_agent_key(name), name)
                )
        historical_keys = dict(exact)

        # These names come from the independently checked, same-starter
        # historical panel. Never infer an agent from the horse alone.
        starter_names: dict[tuple[str, str], str] = {}
        if has_starter_keys:
            refs = original[["race_id", "horse_id", col]].dropna(subset=[col])
            if refs.duplicated(["race_id", "horse_id"]).any():
                raise ValueError("Duplicate verified historical starter agents")
            starter_names = {
                (str(rid), str(hid)): str(name)
                for rid, hid, name in refs.itertuples(index=False, name=None)
                if str(name).strip()
            }

        resolved_aliases: set[str] = set()
        starter_verified = 0
        ambiguous: list[str] = []
        alt = x.source_provider.eq("umanity")
        for idx, row in x.loc[alt].iterrows():
            name = row[col]
            if pd.isna(name) or not str(name).strip():
                ambiguous.append(str(name))
                continue
            name = str(name)
            key = _agent_key(name)
            verified = starter_names.get(
                (str(row["race_id"]), str(row["horse_id"]))
            ) if has_starter_keys else None
            if verified is not None:
                full_key = _agent_key(verified)
                if not full_key.startswith(key):
                    raise ValueError(
                        f"Published {col} alias conflicts with exact historical "
                        f"starter {row['race_id']}/{row['horse_id']}: "
                        f"{name!r} vs {verified!r}"
                    )
                resolved = exact.get(full_key, verified)
                starter_verified += 1
            elif col == "jockey_name" and name in verified_jockey_aliases:
                resolved = verified_jockey_aliases[name]
                if not _agent_key(resolved).startswith(key):
                    raise ValueError("Published profile cannot verify jockey alias")
            elif key in exact:
                resolved = exact[key]
            elif key in active_keys:
                resolved = active_keys[key]
            else:
                active_hits = {k for k in active_keys if k.startswith(key)}
                known_hits = {k for k in historical_keys if k.startswith(key)}
                hits = active_hits if len(active_hits) == 1 else known_hits
                if len(hits) != 1:
                    ambiguous.append(name)
                    continue
                canonical_key = next(iter(hits))
                resolved = active_keys.get(
                    canonical_key, historical_keys.get(canonical_key)
                )
            x.at[idx, col] = resolved
            resolved_aliases.add(name)
        if ambiguous:
            raise ValueError(
                f"Ambiguous or unknown abbreviated {col} identities: "
                f"{sorted(set(ambiguous))[:25]}"
            )
        yahoo = x.source_provider.eq("yahoo")
        x.loc[yahoo, col] = x.loc[yahoo, col].map(
            lambda name, reference=exact: reference.get(_agent_key(name), name)
            if pd.notna(name) else name
        )
        audit[col] = {
            "alternate_distinct_aliases_resolved": len(resolved_aliases),
            "same_starter_source_verified": starter_verified,
            "2026_yahoo_full_name_identities": len(active_keys),
            "historical_full_name_identities": len(historical_keys),
        }
    if x.loc[x.source_provider.eq("umanity"), ["jockey_name", "trainer"]].isna().any().any():
        raise ValueError("Unresolved raw alternate agent alias")
    return x, audit


def _extra_raw_observations(
    full: pd.DataFrame, extra: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Fill source fields for post-Kaggle prior years, preserving original rules."""
    if extra.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Raw results contain duplicate horse/race key")
    original = full.copy()
    extra = extra.copy()
    for frame in (original, extra):
        for name in ("race_id", "horse_id"):
            frame[name] = frame[name].astype(str)
    for name in ("raw_last3f", "raw_corner_first", "raw_corner_last", "raw_early_pct",
                 "raw_passing_gain", "raw_mass", "raw_mass_change", "raw_wet"):
        if name not in original:
            raise ValueError(f"Original raw postrace source feature missing: {name}")
    if set(extra.race_id).difference(set(original.race_id)):
        raise ValueError("External raw results contain unknown historical race ID")
    keys = ["race_id", "horse_id"]
    original = original.merge(
        extra[keys + ["horse_no", "last_3f", "passing", "horse_weight",
                      "track_condition", "jockey_name", "trainer", "rank"]].rename(
                          columns={k: "extra_" + k for k in (
                              "horse_no", "last_3f", "passing", "horse_weight",
                              "track_condition", "jockey_name", "trainer", "rank"
                          )}
                      ),
        on=keys, how="left", validate="one_to_one", indicator=True,
    )
    supplement = original._merge.eq("both")
    if not supplement.any():
        raise ValueError("No audited 2025/26 results were joined to panel")
    mismatch = supplement & pd.to_numeric(
        original.extra_horse_no, errors="coerce"
    ).ne(pd.to_numeric(original.horse_no))
    if mismatch.any():
        raise ValueError("Independent historical raw horse-number discrepancy")
    # Postrace ranks from the audited 2025/26 source are necessary to update
    # opponent Elo. Missing ranks would silently freeze all 2026 Elo histories.
    supplement_rank = pd.to_numeric(
        original.loc[supplement, "extra_rank"].astype("string").str.extract(
            r"^(\d{1,2})(?:\((?:降|再)\))?$", expand=False
        ),
        errors="coerce",
    )
    confirmed_rank = pd.to_numeric(
        original.loc[supplement, "finish_position"], errors="coerce",
    )
    if not supplement_rank.notna().equals(confirmed_rank.notna()) or not (
        supplement_rank.loc[confirmed_rank.notna()].to_numpy(float)
        == confirmed_rank.loc[confirmed_rank.notna()].to_numpy(float)
    ).all():
        raise ValueError("Audited supplemental finishing ranks disagree with verified results")
    original.loc[supplement, "raw_finish_position"] = supplement_rank.to_numpy(dtype=float, na_value=np.nan)

    original.loc[supplement, "jockey_name"] = original.loc[supplement, "extra_jockey_name"]
    original.loc[supplement, "trainer"] = original.loc[supplement, "extra_trainer"]
    original.loc[supplement, "raw_last3f"] = _parse_3f(
        original.loc[supplement, "extra_last_3f"]
    )
    parsed = original.loc[supplement, "extra_passing"].map(_parse_passing)
    first = parsed.map(lambda z: z[0])
    last = parsed.map(lambda z: z[1])
    original.loc[supplement, "raw_corner_first"] = first.to_numpy(float)
    original.loc[supplement, "raw_corner_last"] = last.to_numpy(float)
    valid = first.between(1, original.loc[supplement, "field_size"]) & last.between(
        1, original.loc[supplement, "field_size"]
    )
    denom = original.loc[supplement, "field_size"] - 1
    original.loc[supplement, "raw_early_pct"] = ((first - 1)/denom).where(valid)
    original.loc[supplement, "raw_passing_gain"] = ((first - last)/denom).where(valid)
    mass = original.loc[supplement, "extra_horse_weight"].map(_parse_mass)
    original.loc[supplement, "raw_mass"] = mass.map(lambda z: z[0]).to_numpy(float)
    original.loc[supplement, "raw_mass_change"] = mass.map(
        lambda z: z[1]
    ).to_numpy(float)
    condition = original.loc[supplement, "extra_track_condition"]
    original.loc[supplement, "raw_wet"] = condition.isin(
        ("稍重", "重", "不良")
    ).astype(float).to_numpy()
    if not condition.isin(("良", "稍重", "重", "不良")).all():
        raise ValueError("External result race going unavailable")
    # The original 2018-24 source rule ranks last3f among ALL actual
    # starters in that race (excluding unknown finishers automatically).
    counts = original.groupby("race_id", sort=False).raw_last3f.transform("count")
    ranks = original.groupby("race_id", sort=False).raw_last3f.rank(
        ascending=True, method="average"
    )
    original["raw_last3f_rel"] = ((counts-ranks)/(counts-1)).where(counts.ge(2))
    original = original.drop(columns=[
        x for x in original if x.startswith("extra_")
    ] + ["_merge"])
    return original, {"extra_source_rows": int(supplement.sum()),
                      "extra_source_races": int(extra.race_id.nunique())}


def _supplement_2025(raw: pd.DataFrame, original: pd.DataFrame,
                     cache: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch exactly verified original 2025 races missing from Kaggle v1."""
    missing = raw.loc[
        raw.race_date.dt.year.eq(2025) & raw.source_horse_no.isna(),
        "race_id",
    ].astype(str).unique().tolist()
    if not missing:
        return pd.DataFrame(), []
    if len(missing) > 70:
        raise ValueError("Unexpected 2025 Kaggle history missing >70 races")
    fetcher = CachedFetcher(cache, pause_seconds=.5)
    rows, sources = [], []
    for rid in sorted(missing):
        historical = original.loc[
            original.race_id.astype(str).eq(rid)
        ]
        date = pd.Timestamp(historical.race_date.iloc[0])
        provider = "yahoo"
        try:
            html, record = fetcher.fetch(RESULT_URL.format(provider_id=rid[2:]))
            part = raw_result(html, rid[2:])
        except Exception as exc:  # noqa: BLE001 - verified alternative must pass full-field QA
            print(f"2025 prior result Yahoo unavailable {rid}: {exc}", flush=True)
            provider = "umanity"
            sid = live_umanity.source_id(rid, date)
            url = f"https://umanity.jp/racing/result.php?race_id={sid}"
            html, record = fetcher.fetch(url)
            part = umanity_raw_result(html, rid, date)
        part["source_provider"] = provider
        part["horse_id"] = part.horse_id.map(_jra_internal_id)
        observed = historical[["horse_id", "horse_no"]].copy()
        observed.horse_id = observed.horse_id.astype(str)
        if len(part) != len(observed) or set(zip(part.horse_id, part.horse_no)) != set(
            zip(observed.horse_id, observed.horse_no)
        ):
            raise ValueError(f"Unmatched 2025 supplemental JRA race {rid}")
        rows.append(part)
        sources.append({"race_id": rid, "url": record.url, "sha256": record.sha256})
    return pd.concat(rows, ignore_index=True), sources


def rebuild(a):
    if sha(a.results_csv) != SOURCE_SHA:
        raise ValueError("Kaggle version differs from original frozen F3 source")
    if sha(a.standardized_base) != FROZEN_HISTORY_SHA256:
        raise ValueError("2010-2025 base does not match original standardized SHA")
    base = load_frozen_history(a.standardized_base)
    # The original immutable 2010-2025 source is never modified. Four
    # independently audited late-2025 source races have seven incorrect
    # frozen horse identities; verify against BOTH published full-field
    # sources and repair only those known rows in memory.
    base, identity_repairs = repair_frozen_2025_identities(
        base, a.cache / "audited_2025_identity_repair"
    )
    # Historical JRA starter identities in the immutable base mix a legacy
    # jra: namespace with bare ten-digit IDs. Normalize in memory AFTER the
    # frozen SHA check so earlier and later starts refer to the same horse.
    base = base.copy()
    base["horse_id"] = base.horse_id.map(_jra_internal_id)
    current, live = _load_f3_git_checkpoint(a.live_root)
    if current is None or pd.Timestamp(current["complete_through"]) < pd.Timestamp("2026-10-25"):
        raise ValueError("2026 all-JRA flat source too stale for 17-horse forecast")
    if pd.Timestamp(current["complete_through"]) > ALLOWED_LATEST_SOURCE:
        raise ValueError("Source beyond preregistered 2026-10-30 cutoff")
    if live.empty or not pd.to_datetime(live.race_date).dt.year.eq(2026).all():
        raise ValueError("2026 complete-flat source missing")
    # The frozen model's historic records contain both ID representations;
    # normalize only after independently verifying both source file hashes.
    live = live.copy()
    live["horse_id"] = live.horse_id.map(_jra_internal_id)
    roster_html = a.roster_html.read_text()
    config = json.loads(a.config.read_text())
    roster, _ = parse_roster(roster_html, config)
    if (
        len(roster) != 17 or roster.horse_id.duplicated().any()
        or {str(z.horse_no): str(z.horse_name) for z in roster.itertuples(index=False)}
        != config["horse_no_to_name"]
    ):
        raise ValueError("As-of-date 17-horse roster mismatch")
    roster["horse_id"] = roster.horse_id.map(_jra_internal_id)
    roster["is_open_plus"] = 1
    roster["is_graded"] = 1
    for col in ("finish_position", "early_position", "race_time_seconds"):
        roster[col] = np.nan
    older = pd.concat([_standard_columns(base), _standard_columns(live)],
                      ignore_index=True, sort=False)
    if pd.to_datetime(older.race_date).max() >= TARGET_DATE:
        raise ValueError("Target/future result present before feature generation")
    standard = pd.concat([older, _standard_columns(roster)], ignore_index=True)
    standard["race_date"] = pd.to_datetime(standard.race_date)
    if standard.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate historical result identity")
    panel = build_historical_panel(standard)
    raw, raw_audit = raw_observations(
        panel, a.results_csv, a.races_csv,
        horse_id_normalizer=_jra_internal_id,
    )
    raw.race_date = pd.to_datetime(raw.race_date)
    historical_2025, supplement_sources = _supplement_2025(
        raw, older, a.cache
    )
    aux = pd.read_csv(
        a.extra_csv, dtype={"race_id": str, "horse_id": str}
    )
    aux["horse_id"] = aux.horse_id.map(_jra_internal_id)
    extra_manifest = json.loads(a.extra_manifest.read_text())
    if (
        extra_manifest.get("status") != "COMPLETE_RAW_PRIOR_RESULT_SOURCE"
        or extra_manifest.get("source_csv_sha256") != sha(a.extra_csv)
        or extra_manifest.get("history_input_sha256") != sha(
            a.live_root / "snapshots" / current["snapshot_id"] / "jra_flat_history.parquet"
        )
    ):
        raise ValueError("2026 source evidence not the exact audited live-history snapshot")
    all_2026_keys = set(zip(
        live.race_id.astype(str), live.horse_id.astype(str)
    ))
    if (
        set(zip(aux.race_id, aux.horse_id)) != all_2026_keys
        or aux.duplicated(["race_id", "horse_id"]).any()
    ):
        raise ValueError("2026 raw results do not cover EVERY JRA flat starter")
    if not historical_2025.empty:
        aux = pd.concat([historical_2025, aux], ignore_index=True)
    verified_jockeys, jockey_profile_audit = _verify_umanity_jockey_profiles(
        raw, aux, extra_manifest, a.cache / "verified_umanity_jockey_profiles"
    )
    aux, agent_audit = _canonicalize_extra_agents(
        raw, aux, verified_jockey_aliases=verified_jockeys
    )
    raw, injection_audit = _extra_raw_observations(raw, aux)
    enriched = add_opponent_elo(add_historical_result_features(raw))
    prior_mean = float(enriched.loc[
        enriched.race_date.dt.year.eq(2016), "top3_label"
    ].mean())
    engineered = engineer_features(enriched, prior_mean=prior_mean)
    focus = engineered.loc[engineered.race_id.astype(str).eq(TARGET)].copy()
    if len(focus) != 17:
        raise ValueError("Target F3 context lacks 17 real entries")
    context, relative_audit = full_starter_relative(focus, focus)
    numeric, categorical = feature_registry()["F3_market_field"]
    exclude = set(MARKET_SPLITS + MARKET_FIELD)
    wanted = [k for k in numeric if k not in exclude] + categorical
    absent = set(wanted) - set(context.columns)
    if absent:
        raise ValueError(f"Target is missing mandatory source F3 columns: {sorted(absent)}")
    inputs = context[
        ["race_id", "race_date", "horse_id", "horse_name", "horse_no"] + wanted
    ].copy()
    inputs["horse_id"] = inputs.horse_id.map(_jra_external_id)
    inputs["race_date"] = inputs.race_date.dt.strftime("%Y-%m-%d")
    inputs["horse_no"] = inputs.horse_no.astype(int)
    inputs = inputs.sort_values("horse_no").reset_index(drop=True)
    if inputs.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("F3 target horses duplicated")
    inventory = json.loads(a.inventory.read_text())
    if inventory.get("horse_count") != 17 or inventory.get("race_id") != TARGET:
        raise ValueError("Independent 17-horse history inventory missing")
    coverage = []
    for horse in inventory["horses"]:
        hid = str(horse["horse_id"])
        expected = {str(x["race_id"]) for x in horse["historical_races"]}
        actual = set(older.loc[
            older.horse_id.astype(str).eq(_jra_internal_id(hid)), "race_id"
        ].astype(str))
        if expected != actual or len(expected) < 3:
            raise ValueError(f"Incomplete JRA past starts for {hid}: missing={expected-actual}, extra={actual-expected}")
        coverage.append({"horse_id":hid,"expected":len(expected),"actual":len(actual),
                         "latest_jra_flat_date":horse["latest_jra_flat_date"]})
    if set(inputs.horse_id.astype(str)) != {x["horse_id"] for x in coverage}:
        raise ValueError("Target roster / independent history reconciliation failed")
    # Validate meaningfully observed raw signals, never fill defaults or label
    # synthetic missingness 'complete'. These quantities may be NaN for
    # legitimately unsupported historical races; the presence itself is audited.
    raw_target_coverage = {}
    for name in ("history_last3f_rel_last", "history_mass_last",
                 "history_early_pct3", "strength_elo_prior"):
        raw_target_coverage[name] = int(pd.to_numeric(
            inputs[name], errors="coerce"
        ).notna().sum())
    if raw_target_coverage["strength_elo_prior"] != 17:
        raise ValueError("Target Elo missing for a scheduled horse")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() or a.manifest.exists():
        raise FileExistsError("Immutable original F3 target artifact already exists")
    inputs.to_csv(a.output, index=False, float_format="%.14g", lineterminator="\n")
    original_sources = {
        "standardized_2010_2025_sha256": sha(a.standardized_base),
        "live_2026_snapshot_sha256": current["history_sha256"],
        "live_2026_snapshot_id": current["snapshot_id"],
        "raw_2026_source_manifest_sha256": sha(a.extra_manifest),
        "kaggle_results_v1_sha256": sha(a.results_csv),
        "kaggle_races_v1_sha256": sha(a.races_csv),
        "inventory_source_sha256": sha(a.inventory),
        "roster_source_sha256": sha(a.roster_html),
    }
    combined_history_sha = hashlib.sha256(
        json.dumps(original_sources, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "status": "F3_PRIOR_SOURCE_RECONSTRUCTED_20261030",
        "race_id": TARGET,
        "history_cutoff": str(ALLOWED_LATEST_SOURCE.date()),
        "actual_2026_feed_complete_through": current["complete_through"],
        "history_complete_for_all_entrants": True,
        "no_future_rows": True,
        "no_current_race_results_or_odds": True,
        "source_rebuild_contract": "2010-2025 standardized + all 2026 flat starter raw + old original F3 formulas",
        "feature_csv_sha256": sha(a.output),
        "history_sha256": combined_history_sha,
        "inventory_source_sha256": sha(a.inventory),
        "horse_coverage": coverage,
        "feature_numeric_prior_only": [k for k in numeric if k not in exclude],
        "feature_categorical": categorical,
        "last3f_mass_early_elo_coverage": raw_target_coverage,
        "raw_original_audit": raw_audit,
        "source_injection_audit": injection_audit,
        "independently_verified_2025_identity_repairs": identity_repairs,
        "agent_identity_audit": agent_audit,
        "independently_verified_umanity_jockey_profiles": jockey_profile_audit,
        "2025_supplement_races": supplement_sources,
        "relative_feature_coverage": relative_audit.to_dict(orient="records"),
        "input_source_sha256": original_sources,
        "forwards_2026_11_01_t10_not_yet_locked": True,
    }
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"], "horses":len(inputs),
        "feature_csv_sha256":manifest["feature_csv_sha256"],
        "target_raw_signal":raw_target_coverage,
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standardized-base", type=Path, required=True)
    p.add_argument("--live-root", type=Path, required=True)
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--races-csv", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--roster-html", type=Path, required=True)
    p.add_argument("--extra-csv", type=Path, required=True)
    p.add_argument("--extra-manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/tmp/tennosho-f3-raw-cache"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    rebuild(p.parse_args())


if __name__ == "__main__":
    main()
