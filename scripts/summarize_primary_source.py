from pathlib import Path
import pandas as pd
import kagglehub

DATASET="noriyukifurufuru/japan-horse-racing-2010-2025"
out=Path("docs/HISTORICAL_SOURCE_VALUE_SUMMARY.md")
root=Path(kagglehub.dataset_download(DATASET, output_dir="data/historical_raw/probe2", force_download=True))
if not root.is_dir():
    root=root.parent
races=pd.read_csv(root/"keiba_races.csv", encoding="utf-8-sig", low_memory=False)
results=pd.read_csv(root/"keiba_results.csv", encoding="utf-8-sig", low_memory=False)

lines=["# Historical Source Value Summary","","## races"]
for c in ["date","venue","course_type","turn","weather","track_condition","race_class","race_number"]:
    lines += ["",f"### {c}","", "~~~text"]
    if c in races.columns:
        vc=races[c].astype("string").fillna("<NA>").value_counts(dropna=False).head(80)
        lines += [str(vc)]
    else:
        lines += ["MISSING"]
    lines += ["~~~"]

lines += ["","## results"]
for c in ["rank","sex_age","weight","time","passing","last_3f","horse_weight","odds","popularity"]:
    lines += ["",f"### {c}","", "~~~text"]
    if c in results.columns:
        lines += [str(results[c].astype("string").fillna("<NA>").value_counts(dropna=False).head(40))]
    else:
        lines += ["MISSING"]
    lines += ["~~~"]

lines += ["","## basic ranges","","~~~text",
          f"races rows={len(races)}",
          f"results rows={len(results)}",
          f"race date min={races['date'].min()} max={races['date'].max()}",
          f"race_id unique races={races['race_id'].nunique()}",
          f"race_id unique results={results['race_id'].nunique()}",
          f"horse_id unique={results['horse_id'].nunique()}",
          "~~~"]
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
