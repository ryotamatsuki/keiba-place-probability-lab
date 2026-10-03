from urllib.request import Request,urlopen
from pathlib import Path
import fitz

url="https://www.jra.go.jp/datafile/seiseki/report/2025/2025-1nakayama1.pdf"
data=urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=60).read()
doc=fitz.open(stream=data,filetype="pdf")
text=doc[0].get_text()
Path("docs/JRA_SAMPLE_PDF_TEXT.txt").write_text(text[:12000],encoding="utf-8")
