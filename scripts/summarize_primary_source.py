from pathlib import Path
import csv
from collections import Counter
import pandas as pd
import kagglehub

DATASET="noriyukifurufuru/japan-horse-racing-2010-2025"
out=Path("docs/HISTORICAL_SOURCE_VALUE_SUMMARY.md")
root=Path(kagglehub.dataset_download(DATASET, output_dir="data/historical_raw/probe2", force_download=True))
if not root.is_dir():
    root=root.parent

races=pd.read_csv(root/"keiba_races.csv", encoding="utf-8-sig", low_memory=False)

result_path=root/"keiba_results.csv"
field_counts=Counter()
bad_examples=[]
counters={c:Counter() for c in ["rank","sex_age","weight","time","passing","last_3f","horse_weight","odds","popularity"]}
unique_race_ids=set()
unique_horse_ids=set()
valid_rows=0

with result_path.open("r",encoding="utf-8-sig",newline="") as f:
    reader=csv.reader(f)
    header=next(reader)
    expected=len(header)
    idx={name:i for i,name in enumerate(header)}
    total=0
    for line_no,row in enumerate(reader,start=2):
        total+=1
        field_counts[len(row)]+=1
        if len(row)!=expected:
            if len(bad_examples)<20:
                bad_examples.append((line_no,len(row),row))
            continue
        valid_rows+=1
        unique_race_ids.add(row[idx["race_id"]])
        unique_horse_ids.add(row[idx["horse_id"]])
        for c in counters:
            counters[c][row[idx[c]] or "<EMPTY>"] += 1

lines=["# Historical Source Value Summary","","## parser diagnostics","",
       f"- header fields: {expected}",
       f"- raw data rows: {total}",
       f"- valid-width rows: {valid_rows}",
       f"- field-count distribution: {dict(sorted(field_counts.items()))}",
       "",
       "### malformed examples","",
       "~~~text"]
for line_no,n,row in bad_examples:
    lines.append(f"line={line_no} fields={n} row={row}")
lines += ["~~~","","## races"]

for c in ["date","venue","course_type","turn","weather","track_condition","race_class","race_number"]:
    lines += ["",f"### {c}","", "~~~text"]
    if c in races.columns:
        lines += [str(races[c].astype("string").fillna("<NA>").value_counts(dropna=False).head(80))]
    else:
        lines += ["MISSING"]
    lines += ["~~~"]

lines += ["","## results"]
for c,ctr in counters.items():
    lines += ["",f"### {c}","", "~~~text"]
    for value,count in ctr.most_common(40):
        lines.append(f"{value!r}: {count}")
    lines += ["~~~"]

lines += ["","## basic ranges","","~~~text",
          f"races rows={len(races)}",
          f"results raw rows={total}",
          f"results valid-width rows={valid_rows}",
          f"race date min={races['date'].min()} max={races['date'].max()}",
          f"race_id unique races={races['race_id'].nunique()}",
          f"race_id unique valid results={len(unique_race_ids)}",
          f"horse_id unique valid={len(unique_horse_ids)}",
          "~~~"]
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
