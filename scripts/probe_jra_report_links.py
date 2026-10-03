from urllib.request import Request,urlopen
from pathlib import Path
u="https://www.jra.go.jp/datafile/seiseki/report/2025.html"
h=urlopen(Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=60).read().decode("utf-8","replace")
needle="2025-1nakayama1.pdf"
i=h.find(needle)
Path("docs/JRA_REPORT_HTML_SNIPPET.txt").write_text(h[max(0,i-2500):i+1500],encoding="utf-8")
