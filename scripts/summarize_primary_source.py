from pathlib import Path
import csv
import pandas as pd
import kagglehub

DATASET="noriyukifurufuru/japan-horse-racing-2010-2025"
out=Path("docs/HISTORICAL_SOURCE_VALUE_SUMMARY.md")
root=Path(kagglehub.dataset_download(DATASET, output_dir="data/historical_raw/probe2", force_download=True))
if not root.is_dir():
    root=root.parent

races=pd.read_csv(root/"keiba_races.csv", encoding="utf-8-sig", low_memory=False)

result_path=root/"keiba_results.csv"
field_counts={}
bad_examples=[]
with result_path.open("r",encoding="utf-8-sig",newline="") as f:
    reader=csv.reader(f)
    header=next(reader)
    expected=len(header)
    total=0
    for line_no,row in enumerate(reader,start=2):
        total+=1
        field_counts[len(row)]=field_counts.get(len(row),0)+1
        if len(row)!=expected and len(bad_examples)<20:
            bad_examples.append((line_no,len(row),row))

# Parse the clean rows for distributions, skipping malformed rows only in this diagnostic.
results=pd.read_csv(
    result_path,
    encoding="utf-8-sig",
    low_memory=False,
    engine="python",
    on_bad_lines="skip",
)

lines=["# Historical Source Value Summary","","## parser diagnostics","",
       f"- header fields: {expected}",
       f"- raw data rows: {total}",
       f"- field-count distribution: {field_counts}",
       f"- pandas rows after malformed-line skip: {len(results)}",
       "",
       "### malformed examples","",
       "~~~text"]
for line_no,n,row in bad_examples:
    lines.append(f"line={line_no} fields={n} row={row}")
lines += ["~~~","","## races"]

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
          f"results raw rows={total}",
          f"results parseable rows={len(results)}",
          f"race date min={races['date'].min()} max={races['date'].max()}",
          f"race_id unique races={races['race_id'].nunique()}",
          f"race_id unique parseable results={results['race_id'].nunique()}",
          f"horse_id unique parseable={results['horse_id'].nunique()}",
          "~~~"]
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
