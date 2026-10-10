"""One-race strictly pre-start inference, using frozen Scope V3 model.

As-of caveat: official 2026 snapshot is certified only through 2026-10-03.
October 4-10 history is not complete; explicitly use 8-day freshness bound.
Never label these as market-calibrated production predictions.
"""
import json
from pathlib import Path
import pandas as pd
from keiba_place_lab.live_history import load_frozen_history

names = [
    "ニシノティアモ", "チェルビアット", "クイーンズウォーク",
    "クランフォード", "カネラフィーナ", "ムイ",
    "ミラビリスマジック", "ルージュソリテール",
    "ウイントワイライト", "セキトバイースト",
    "ラヴァンダ", "ミアネーロ", "ワタシマツワ",
    "ハワイアンティアレ", "ジョスラン", "カムニャック",
]
four_year_olds = {2, 5, 6, 8, 9, 13, 15, 16}
heavier = {11, 16}

out = Path("analysis/ireland_20261011_shadow")
out.mkdir(parents=True, exist_ok=True)
import glob
bases = glob.glob("/tmp/stage36/**/standardized_jra_flat_source_v1.parquet", recursive=True)
assert len(bases) == 1, f"Expected one historical base: {bases}"
base = load_frozen_history(Path(bases[0]))
pointer = json.loads(Path("data/live_history/2026/current.json").read_text())
assert pointer["complete_through"] == "2026-10-03", pointer
snapshot = Path("data/live_history/2026/snapshots") / pointer["snapshot_id"]
live = pd.read_parquet(snapshot / "jra_flat_history.parquet")
import hashlib
sha = hashlib.sha256((snapshot / "jra_flat_history.parquet").read_bytes()).hexdigest()
assert sha == pointer["history_sha256"]
for f in (base, live):
    f["race_id"] = f["race_id"].astype("string")
    f["horse_id"] = f["horse_id"].astype("string")
full = pd.concat([base, live], ignore_index=True, sort=False)
full["race_date"] = pd.to_datetime(full["race_date"])
full = full.loc[full.race_date.lt(pd.Timestamp("2026-10-11"))].copy()
assert not full.duplicated(["race_id", "horse_id"]).any()
assert full.race_date.max() == pd.Timestamp("2026-10-03")
rows = []
lookup_audit = []
for number, name in enumerate(names, start=1):
    age = 4 if number in four_year_olds else 5
    birth_year = str(2026 - age)
    found = full.loc[
        full.horse_name.eq(name)
        & full.horse_id.astype(str).str.startswith(birth_year)
    ].sort_values("race_date")
    if found.empty:
        raise ValueError(f"NO_REGISTERED_HISTORY_FOR {number}: {name}, birth_year={birth_year}")
    ids = found.horse_id.dropna().unique().tolist()
    if len(ids) != 1:
        raise ValueError(f"AMBIGUOUS_HORSE_ID {name}: {ids}")
    last_date = str(found.race_date.iloc[-1].date())
    rows.append(dict(
        race_id="202605040411",
        race_date="2026-10-11",
        racecourse="東京", surface="turf", distance_m=1800,
        turn_direction="left", race_class="Open",
        horse_id=str(ids[0]), horse_name=name, horse_no=number,
        sex="F", age=4 if number in four_year_olds else 5,
        assigned_weight_kg=56.0 if number in heavier else 55.0,
        field_size=16, declared_field_size=16,
    ))
    lookup_audit.append(dict(horse_no=number, horse_name=name,
        horse_id=str(ids[0]), historical_starts=len(found),
        last_race_date=last_date))
roster = pd.DataFrame(rows)
roster.to_csv(out / "roster.csv", index=False)
full.to_parquet("/tmp/ireland_prediction_history.parquet")
pd.DataFrame(lookup_audit).to_csv(out / "horse_ids_and_history.csv", index=False)
(out / "DATA_LIMITATIONS.md").write_text(
    "# Ireland Trophy 2026-10-11 shadow-only model forecast\\n\\n"
    "- Model: frozen stage4_scope_v3 turf_G0, production market-only remains canonical.\\n"
    "- Frozen pre-2026 standardized historical data + certified live snapshot through Oct 3.\\n"
    "- Explicit max-history-age-days override: 8 (default is 7).\\n"
    "- Oct 4-10 official results were not included. All 16 horses' latest starts"
    " should be verified against JRA before placing bets.\\n"
    "- Odds are not model inputs. Any market EV comparison requires time-locked"
    " actual place odds and independent calibration.\\n",
    encoding="utf-8",
)
print(json.dumps({"roster":len(roster),"history_rows":len(full),
  "last_history_date":str(full.race_date.max().date()),
  "target":"2026-10-11", "horses": lookup_audit}, ensure_ascii=False))
