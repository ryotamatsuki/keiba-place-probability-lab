"""Audited, versioned JRA live-history snapshots for forward prediction.

The live history is deliberately separate from model training artifacts.  A snapshot
contains every confirmed JRA flat starter for the requested year/period plus an
independent declared-entry audit used to prove field completeness.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .live_public import (
    _identity,
    _sex_age,
    _weight,
    canonical_id,
    metadata,
)
from .stage4_successor_phase3a import assert_full_field_context

SCHEMA_VERSION = "jra_flat_live_history_v1"
MONTHLY_URL = "https://sports.yahoo.co.jp/keiba/schedule/monthly/?year={year}&month={month}"
LIST_URL = "https://sports.yahoo.co.jp/keiba/race/list/{meeting_day_id}"
RESULT_URL = "https://sports.yahoo.co.jp/keiba/race/result/{provider_id}"
DENMA_URL = "https://sports.yahoo.co.jp/keiba/race/denma/{provider_id}"

_RESULT_STATUSES = {
    "中止": ("dnf", True),
    "失格": ("disqualified", True),
    "取消": ("scratched", False),
    "除外": ("excluded", False),
}


@dataclass(frozen=True)
class FetchRecord:
    url: str
    retrieved_at: str
    sha256: str
    cache_path: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class CachedFetcher:
    """HTTP fetcher with explicit refresh semantics and provenance sidecars."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str = "keiba-place-probability-lab research/0.1",
        pause_seconds: float = 0.50,
        timeout: tuple[int, int] = (15, 45),
        max_attempts: int = 6,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.pause_seconds = pause_seconds
        self.timeout = timeout
        self.max_attempts = max_attempts

    def fetch(self, url: str, *, force: bool = False) -> tuple[str, FetchRecord]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        html_path = self.cache_dir / f"{key}.html"
        meta_path = self.cache_dir / f"{key}.json"
        if force or not html_path.exists() or not meta_path.exists():
            last_error = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = requests.get(
                        url,
                        timeout=self.timeout,
                        headers={"User-Agent": self.user_agent},
                    )
                    response.raise_for_status()
                    data = response.content
                    html_path.write_bytes(data)
                    record = {
                        "url": url,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "sha256": sha256_bytes(data),
                        "cache_path": str(html_path),
                    }
                    meta_path.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
                    )
                    if self.pause_seconds:
                        time.sleep(self.pause_seconds)
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    status = getattr(exc.response, "status_code", None)
                    retryable = status in {429, 500, 502, 503, 504} or status is None
                    if not retryable or attempt == self.max_attempts:
                        raise
                    time.sleep(min(2 ** (attempt - 1), 16))
            else:  # pragma: no cover
                raise RuntimeError(f"Fetch retries exhausted: {url}") from last_error
        record = FetchRecord(**json.loads(meta_path.read_text()))
        return html_path.read_text(errors="replace"), record


def _date_from_text(text: str) -> pd.Timestamp:
    m = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if not m:
        raise ValueError("Race date absent")
    return pd.Timestamp(f"{m[1]}-{int(m[2]):02}-{int(m[3]):02}")


def parse_monthly_schedule(html: str, *, year: int, month: int) -> list[str]:
    """Return every JRA meeting-day id advertised on a monthly schedule page."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if str(year) not in text:
        raise ValueError(f"Monthly schedule does not identify year {year}")
    ids: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/keiba/race/list/(\d{8})(?:[/?#]|$)", a["href"])
        if m:
            ids.add(m[1])
        m = re.search(r"/keiba/race/(?:result|denma|index)/(\d{10})(?:[/?#]|$)", a["href"])
        if m:
            ids.add(m[1][:8])
    # The Yahoo id embeds two-digit year.
    ids = {x for x in ids if int(x[:2]) == year % 100}
    if not ids:
        raise ValueError(f"No meeting days found for {year}-{month:02}")
    return sorted(ids)


def parse_race_list(html: str, *, meeting_day_id: str) -> pd.DataFrame:
    """Enumerate every scheduled race on one JRA meeting-day.

    Obstacles are retained in the ledger but marked target_flat=False.
    """
    soup = BeautifulSoup(html, "html.parser")
    race_date = _date_from_text(soup.get_text(" ", strip=True))
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/keiba/race/(?:result|denma|index)/(\d{10})(?:[/?#]|$)", a["href"])
        if not m:
            continue
        pid = m[1]
        if not pid.startswith(meeting_day_id) or pid in seen:
            continue
        tr = a.find_parent("tr")
        row_text = tr.get_text(" ", strip=True) if tr else a.parent.get_text(" ", strip=True)
        is_obstacle = "障害" in row_text or "障" in row_text and "芝→" in row_text
        seen.add(pid)
        rows.append(
            {
                "provider_id": pid,
                "race_id": canonical_id(pid),
                "race_date": race_date,
                "meeting_day_id": meeting_day_id,
                "race_no": int(pid[-2:]),
                "target_flat": not is_obstacle,
                "schedule_row_text": row_text,
            }
        )
    if not rows:
        raise ValueError(f"No races found for meeting day {meeting_day_id}")
    frame = pd.DataFrame(rows).sort_values("race_no").reset_index(drop=True)
    if frame.provider_id.duplicated().any():
        raise ValueError("Duplicate scheduled race ids")
    return frame


def _status_from_rank(rank: str) -> tuple[str, bool, float]:
    rank = rank.strip()
    if rank.isdigit():
        return "finished", True, float(rank)
    for token, (status, starter) in _RESULT_STATUSES.items():
        if token in rank:
            return status, starter, float("nan")
    raise ValueError(f"Unsupported result status: {rank!r}")


def parse_result_audited(
    html: str,
    provider_id: str,
    *,
    source_url: str,
    retrieved_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse result rows while preserving cancellation and disqualification status.

    Returns:
      starters: actual starters only, suitable for V3 live history.
      result_entries: every declared horse shown in the result table.
    """
    soup, meta = metadata(html, provider_id)
    tables = [
        t
        for t in soup.select("table")
        if all(s in t.get_text() for s in ("着順", "馬名", "通過順位", "騎手名"))
    ]
    if len(tables) != 1:
        raise ValueError("Confirmed result table absent")
    entries: list[dict[str, object]] = []
    declared_field_size = 0
    for tr in tables[0].select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        rank = c[0].get_text(strip=True)
        number = int(c[2].get_text(strip=True))
        declared_field_size = max(declared_field_size, number)
        finish_status, is_starter, finish_position = _status_from_rank(rank)
        hid, name = _identity(c[3])
        sex, age = _sex_age(c[3])
        tm = re.match(r"\s*(\d+):(\d+\.\d+)", c[4].get_text())
        early = re.match(r"\s*(\d+)", c[5].get_text())
        if finish_status == "finished" and not tm:
            raise ValueError("Finished runner has no race time")
        row = {k: v for k, v in meta.items() if k != "off_at"}
        row.update(
            horse_id=hid,
            horse_name=name,
            horse_no=number,
            sex=sex,
            age=age,
            assigned_weight_kg=_weight(c[6]),
            finish_position=finish_position,
            finish_status=finish_status,
            is_starter=is_starter,
            race_time_seconds=(int(tm[1]) * 60 + float(tm[2])) if tm else float("nan"),
            early_position=float(early[1]) if early else float("nan"),
            outcome_confirmed=True,
            declared_field_size=declared_field_size,  # fixed below after full scan
            result_source_url=source_url,
            result_retrieved_at=retrieved_at,
        )
        entries.append(row)
    if not entries:
        raise ValueError("Empty result table")
    result_entries = pd.DataFrame(entries)
    result_entries["declared_field_size"] = declared_field_size
    if result_entries.horse_no.duplicated().any() or result_entries.horse_id.duplicated().any():
        raise ValueError("Duplicate result identities")
    starters = result_entries.loc[result_entries.is_starter].copy()
    if starters.empty or not starters.finish_position.eq(1).any():
        raise ValueError("No confirmed winning starter")
    starters["field_size"] = len(starters)
    result_entries["field_size"] = len(starters)
    assert_full_field_context(starters)
    return starters.reset_index(drop=True), result_entries.reset_index(drop=True)


def parse_declared_entry_audit(
    html: str,
    provider_id: str,
    *,
    source_url: str,
    retrieved_at: str,
) -> pd.DataFrame:
    """Parse the separately published entry/roster page for field verification."""
    soup, meta = metadata(html, provider_id)
    table = soup.select_one("#denma_list table")
    if table is None:
        raise ValueError("Declared roster absent")
    rows: list[dict[str, object]] = []
    declared_field_size = 0
    for tr in table.select("tr"):
        c = tr.find_all("td", recursive=False)
        if not c:
            continue
        number = int(c[1].get_text(strip=True))
        declared_field_size = max(declared_field_size, number)
        hid, name = _identity(c[2])
        text = tr.get_text(" ", strip=True)
        if "取消" in text:
            status = "scratched"
            is_starter = False
        elif "除外" in text:
            status = "excluded"
            is_starter = False
        else:
            status = "active"
            is_starter = True
        sex, age = _sex_age(c[2])
        row = {k: v for k, v in meta.items() if k not in ("off_at",)}
        row.update(
            horse_id=hid,
            horse_name=name,
            horse_no=number,
            sex=sex,
            age=age,
            assigned_weight_kg=_weight(c[3]),
            entry_status=status,
            is_starter=is_starter,
            declared_field_size=declared_field_size,  # fixed below
            entry_source_url=source_url,
            entry_retrieved_at=retrieved_at,
        )
        rows.append(row)
    if not rows:
        raise ValueError("Empty declared roster")
    out = pd.DataFrame(rows)
    out["declared_field_size"] = declared_field_size
    if out.horse_no.duplicated().any() or out.horse_id.duplicated().any():
        raise ValueError("Duplicate declared identities")
    return out.reset_index(drop=True)


def verify_full_field(
    result_entries: pd.DataFrame,
    roster_entries: pd.DataFrame,
) -> dict[str, object]:
    """Cross-check actual starters against the independently parsed entry page.

    The result table is allowed to omit cancelled/excluded horses. The declared
    roster is therefore the authority for declared-field identities and size,
    while the result table is the authority for actual starter outcomes.
    """
    key_cols = ["horse_no", "horse_id", "horse_name"]
    a = result_entries[key_cols + ["finish_status", "is_starter"]].copy()
    b = roster_entries[key_cols + ["entry_status", "is_starter"]].copy()
    for frame in (a, b):
        for col in ("horse_id", "horse_name"):
            frame[col] = frame[col].astype(str)

    a_keys = set(map(tuple, a[key_cols].to_records(index=False)))
    b_keys = set(map(tuple, b[key_cols].to_records(index=False)))
    unexpected = a_keys - b_keys
    if unexpected:
        raise ValueError(f"Result contains identities absent from declared roster: {sorted(unexpected)[:5]}")

    merged = b.merge(
        a,
        on=key_cols,
        how="left",
        validate="one_to_one",
        suffixes=("_entry", "_result"),
    )
    effective_scratched = 0
    effective_excluded = 0
    for row in merged.itertuples(index=False):
        if pd.isna(row.finish_status):
            # Some result tables omit a horse already marked cancelled/excluded on
            # the archived entry page.  That is a confirmed non-starter, not a gap.
            if row.entry_status == "scratched":
                effective_scratched += 1
                continue
            if row.entry_status == "excluded":
                effective_excluded += 1
                continue
            raise ValueError("Declared active horse missing from confirmed result/status table")
        if row.finish_status in ("scratched", "excluded"):
            # Conversely, the archived entry page may preserve the original active
            # declaration while the result page records the later non-starter event.
            if bool(row.is_starter_result):
                raise ValueError("Cancelled/excluded result row flagged as starter")
            if row.entry_status in ("scratched", "excluded") and row.entry_status != row.finish_status:
                raise ValueError("Cancellation/exclusion type conflicts across published views")
            if row.finish_status == "scratched":
                effective_scratched += 1
            else:
                effective_excluded += 1
        else:
            if row.finish_status not in ("finished", "dnf", "disqualified"):
                raise ValueError("Unsupported actual-starter result status")
            if not bool(row.is_starter_result):
                raise ValueError("Confirmed starter flagged non-starter in result")
            if row.entry_status in ("scratched", "excluded"):
                raise ValueError("Entry page marks a confirmed starter as non-starter")

    result_starters = int(result_entries.is_starter.sum())
    declared = int(roster_entries.declared_field_size.iloc[0])
    return {
        "declared_entries": len(roster_entries),
        "actual_starters": result_starters,
        "declared_field_size": declared,
        "scratched": effective_scratched,
        "excluded": effective_excluded,
        "dnf": int(result_entries.finish_status.eq("dnf").sum()),
        "disqualified": int(result_entries.finish_status.eq("disqualified").sum()),
    }


def build_expected_ledger(
    *,
    year: int,
    through: pd.Timestamp,
    fetcher: CachedFetcher,
    force_schedule: bool = True,
) -> tuple[pd.DataFrame, list[FetchRecord]]:
    provenance: list[FetchRecord] = []
    meeting_days: set[str] = set()
    for month in range(1, through.month + 1):
        url = MONTHLY_URL.format(year=year, month=month)
        html, rec = fetcher.fetch(url, force=force_schedule)
        provenance.append(rec)
        meeting_days.update(parse_monthly_schedule(html, year=year, month=month))
    frames = []
    for meeting_day_id in sorted(meeting_days):
        url = LIST_URL.format(meeting_day_id=meeting_day_id)
        html, rec = fetcher.fetch(url, force=force_schedule)
        provenance.append(rec)
        frame = parse_race_list(html, meeting_day_id=meeting_day_id)
        frame["schedule_source_url"] = rec.url
        frame["schedule_retrieved_at"] = rec.retrieved_at
        frames.append(frame)
    ledger = pd.concat(frames, ignore_index=True)
    ledger["race_date"] = pd.to_datetime(ledger.race_date)
    ledger = ledger.loc[ledger.race_date.le(pd.Timestamp(through).normalize())].copy()
    ledger = ledger.sort_values(["race_date", "meeting_day_id", "race_no"]).reset_index(drop=True)
    if ledger.race_id.duplicated().any():
        raise ValueError("Duplicate expected race ids across schedule")
    return ledger, provenance


def _read_current_snapshot(root: Path) -> tuple[dict | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pointer = root / "current.json"
    if not pointer.exists():
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    current = json.loads(pointer.read_text())
    snap = root / "snapshots" / current["snapshot_id"]
    manifest = json.loads((snap / "manifest.json").read_text())
    history = pd.read_parquet(snap / "jra_flat_history.parquet")
    entries = pd.read_parquet(snap / "entry_audit.parquet")
    ledger = pd.read_csv(snap / "race_ledger.csv", dtype={"race_id": "string", "provider_id": "string"})
    return manifest, history, entries, ledger


def _safe_json(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def publish_snapshot(
    *,
    root: Path,
    through: pd.Timestamp,
    expected_ledger: pd.DataFrame,
    starters: pd.DataFrame,
    entries: pd.DataFrame,
    race_qas: list[dict[str, object]],
    provenance: Iterable[FetchRecord],
) -> dict[str, object]:
    """Write a new immutable snapshot then atomically advance current.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    starters = starters.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    entries = entries.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    assert_full_field_context(starters)
    expected_flat = expected_ledger.loc[expected_ledger.target_flat].copy()
    confirmed_ids = set(starters.race_id.astype(str).unique())
    expected_ids = set(expected_flat.race_id.astype(str))
    missing = sorted(expected_ids - confirmed_ids)
    unexpected = sorted(confirmed_ids - expected_ids)
    if unexpected:
        raise ValueError(f"Unexpected race ids in history: {unexpected[:20]}")
    confirmed_races = len(confirmed_ids)
    expected_races = len(expected_ids)
    if missing:
        earliest = expected_flat.loc[expected_flat.race_id.astype(str).isin(missing), "race_date"].min()
        complete_through = (pd.Timestamp(earliest).normalize() - pd.Timedelta(days=1)).date().isoformat()
    else:
        complete_through = pd.Timestamp(through).normalize().date().isoformat()
    latest = pd.to_datetime(starters.race_date).max().date().isoformat() if len(starters) else None

    with tempfile.TemporaryDirectory(dir=root) as td:
        temp = Path(td)
        payload = temp / "payload"
        payload.mkdir()
        history_path = payload / "jra_flat_history.parquet"
        entries_path = payload / "entry_audit.parquet"
        ledger_path = payload / "race_ledger.csv"
        starters.to_parquet(history_path, index=False)
        entries.to_parquet(entries_path, index=False)
        expected_ledger.to_csv(ledger_path, index=False)

        history_sha = sha256_file(history_path)
        entries_sha = sha256_file(entries_path)
        ledger_sha = sha256_file(ledger_path)
        snapshot_id = f"{SCHEMA_VERSION}-{pd.Timestamp(through).strftime('%Y%m%d')}-{history_sha[:12]}"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "requested_through": pd.Timestamp(through).normalize().date().isoformat(),
            "complete_through": complete_through,
            "latest_race_date": latest,
            "expected_races": expected_races,
            "confirmed_races": confirmed_races,
            "missing_race_ids": missing,
            "expected_race_days": int(expected_flat.race_date.nunique()),
            "confirmed_race_days": int(starters.race_date.nunique()),
            "rows": len(starters),
            "entry_rows": len(entries),
            "history_sha256": history_sha,
            "entry_audit_sha256": entries_sha,
            "race_ledger_sha256": ledger_sha,
            "race_qa": race_qas,
            "source_fetches": [r.__dict__ for r in provenance],
            "generated_at": datetime.now(UTC).isoformat(),
        }
        if missing:
            raise ValueError(
                f"Snapshot incomplete: {confirmed_races}/{expected_races}; "
                f"missing {missing[:20]}"
            )
        manifest_path = payload / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_safe_json) + "\n")
        destination = root / "snapshots" / snapshot_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(payload), str(destination))

    pointer = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "complete_through": complete_through,
        "history_sha256": history_sha,
    }
    tmp_pointer = root / "current.json.tmp"
    tmp_pointer.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n")
    tmp_pointer.replace(root / "current.json")
    return manifest


def update_live_history(
    *,
    year: int,
    through: pd.Timestamp,
    root: Path,
    cache_dir: Path,
    recheck_days: int = 14,
    max_workers: int = 6,
) -> dict[str, object]:
    """Build/reconcile the year-to-date JRA flat history and publish only on full QA."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    through = pd.Timestamp(through).normalize()
    if through.year != year:
        raise ValueError("through must be in the requested year")
    if recheck_days < 0:
        raise ValueError("recheck_days must be nonnegative")
    fetcher = CachedFetcher(cache_dir)
    expected, provenance = build_expected_ledger(year=year, through=through, fetcher=fetcher)
    current_manifest, current_history, current_entries, _ = _read_current_snapshot(Path(root))
    current_races = set(current_history.race_id.astype(str)) if len(current_history) else set()
    refresh_from = through - pd.Timedelta(days=max(recheck_days - 1, 0))

    flat = expected.loc[expected.target_flat].copy()
    tasks = []
    for row in flat.itertuples(index=False):
        need = str(row.race_id) not in current_races or pd.Timestamp(row.race_date) >= refresh_from
        if need:
            tasks.append(row)

    collected_history: dict[str, pd.DataFrame] = {}
    collected_entries: dict[str, pd.DataFrame] = {}
    race_qas: dict[str, dict[str, object]] = {}
    fetch_records: list[FetchRecord] = list(provenance)

    def collect(row):
        result_url = RESULT_URL.format(provider_id=row.provider_id)
        denma_url = DENMA_URL.format(provider_id=row.provider_id)
        result_html, result_rec = fetcher.fetch(result_url, force=True)
        roster_html, roster_rec = fetcher.fetch(denma_url, force=True)
        starters, result_entries = parse_result_audited(
            result_html,
            row.provider_id,
            source_url=result_rec.url,
            retrieved_at=result_rec.retrieved_at,
        )
        roster_entries = parse_declared_entry_audit(
            roster_html,
            row.provider_id,
            source_url=roster_rec.url,
            retrieved_at=roster_rec.retrieved_at,
        )
        qa = verify_full_field(result_entries, roster_entries)
        declared_size = int(roster_entries.declared_field_size.iloc[0])
        starters["declared_field_size"] = declared_size
        result_entries["declared_field_size"] = declared_size
        if not starters.race_id.astype(str).eq(str(row.race_id)).all():
            raise ValueError("Result race identity mismatch")
        if not pd.to_datetime(starters.race_date).dt.normalize().eq(pd.Timestamp(row.race_date).normalize()).all():
            raise ValueError("Result race date mismatch")
        audit = roster_entries.merge(
            result_entries[
                [
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "finish_status",
                    "finish_position",
                    "outcome_confirmed",
                    "result_source_url",
                    "result_retrieved_at",
                ]
            ],
            on=["race_id", "horse_id", "horse_no"],
            how="left",
            validate="one_to_one",
        )
        nonstarter = audit["entry_status"].isin(["scratched", "excluded"])
        audit.loc[nonstarter & audit["finish_status"].isna(), "finish_status"] = audit.loc[
            nonstarter & audit["finish_status"].isna(), "entry_status"
        ]
        audit.loc[nonstarter & audit["outcome_confirmed"].isna(), "outcome_confirmed"] = True
        audit["result_source_url"] = audit["result_source_url"].fillna(result_rec.url)
        audit["result_retrieved_at"] = audit["result_retrieved_at"].fillna(result_rec.retrieved_at)
        qa.update(
            race_id=str(row.race_id),
            provider_id=str(row.provider_id),
            race_date=pd.Timestamp(row.race_date).date().isoformat(),
            result_sha256=result_rec.sha256,
            roster_sha256=roster_rec.sha256,
        )
        return str(row.race_id), starters, audit, qa, [result_rec, roster_rec]

    failures: dict[str, str] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(collect, row): row for row in tasks}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    rid, h, e, qa, recs = future.result()
                    collected_history[rid] = h
                    collected_entries[rid] = e
                    race_qas[rid] = qa
                    fetch_records.extend(recs)
                except Exception as exc:  # noqa: BLE001 - batch must record any per-race failure
                    failures[str(row.race_id)] = f"{type(exc).__name__}: {exc}"
    if failures:
        failure_path = Path(root) / "last_failure.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "requested_through": through.date().isoformat(),
                    "failed_races": failures,
                    "current_snapshot_preserved": current_manifest["snapshot_id"] if current_manifest else None,
                    "generated_at": datetime.now(UTC).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"failed_races": failures}, ensure_ascii=False, indent=2), flush=True)
        examples = list(failures.items())[:10]
        raise RuntimeError(
            f"Live history update failed for {len(failures)} races; examples={examples}"
        )

    # Replace refreshed races race-by-race; preserve older confirmed races.
    replace_ids = set(collected_history)
    if len(current_history):
        keep_history = current_history.loc[~current_history.race_id.astype(str).isin(replace_ids)].copy()
        keep_entries = current_entries.loc[~current_entries.race_id.astype(str).isin(replace_ids)].copy()
    else:
        keep_history = pd.DataFrame()
        keep_entries = pd.DataFrame()
    history = pd.concat([keep_history, *collected_history.values()], ignore_index=True, sort=False)
    entries = pd.concat([keep_entries, *collected_entries.values()], ignore_index=True, sort=False)
    history["race_id"] = history.race_id.astype("string")
    history["horse_id"] = history.horse_id.astype("string")
    entries["race_id"] = entries.race_id.astype("string")
    entries["horse_id"] = entries.horse_id.astype("string")
    history["race_date"] = pd.to_datetime(history.race_date)
    entries["race_date"] = pd.to_datetime(entries.race_date)

    # Keep only races in the current expected ledger; a schedule correction may remove a race.
    expected_flat_ids = set(flat.race_id.astype(str))
    history = history.loc[history.race_id.astype(str).isin(expected_flat_ids)].copy()
    entries = entries.loc[entries.race_id.astype(str).isin(expected_flat_ids)].copy()

    # Reconstruct QA for preserved races from stored audit rows when no refresh occurred.
    all_qas = []
    for rid in sorted(expected_flat_ids):
        if rid in race_qas:
            all_qas.append(race_qas[rid])
            continue
        h = history.loc[history.race_id.astype(str).eq(rid)]
        e = entries.loc[entries.race_id.astype(str).eq(rid)]
        if h.empty or e.empty:
            continue
        all_qas.append(
            {
                "race_id": rid,
                "provider_id": rid[2:],
                "race_date": pd.to_datetime(h.race_date).iloc[0].date().isoformat(),
                "declared_entries": len(e),
                "actual_starters": len(h),
                "declared_field_size": int(h.declared_field_size.iloc[0]),
                "scratched": int(e.entry_status.eq("scratched").sum()),
                "excluded": int(e.entry_status.eq("excluded").sum()),
                "dnf": int(e.finish_status.eq("dnf").sum()),
                "disqualified": int(e.finish_status.eq("disqualified").sum()),
                "reused_from_snapshot": current_manifest["snapshot_id"] if current_manifest else None,
            }
        )

    manifest = publish_snapshot(
        root=Path(root),
        through=through,
        expected_ledger=expected,
        starters=history,
        entries=entries,
        race_qas=all_qas,
        provenance=fetch_records,
    )
    failure = Path(root) / "last_failure.json"
    if failure.exists():
        failure.unlink()
    return manifest
