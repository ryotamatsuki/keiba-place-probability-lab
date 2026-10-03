from urllib.request import Request,urlopen
from html.parser import HTMLParser
from pathlib import Path

class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag=="a":
            d=dict(attrs)
            if d.get("href"): self.links.append(d["href"])

lines=[]
for year in (2010,2015,2016,2017):
    u=f"https://www.jra.go.jp/datafile/seiseki/report/{year}.html"
    h=urlopen(Request(u,headers={"User-Agent":"Mozilla/5.0"}),timeout=60).read().decode("utf-8","replace")
    p=P(); p.feed(h)
    pdf=[x for x in p.links if ".pdf" in x.lower()]
    lines.append(f"YEAR {year} COUNT {len(pdf)}")
    lines.extend(pdf[:120])
    lines.append("")
Path("docs/JRA_OLD_PDF_LINK_SAMPLES.txt").write_text("\n".join(lines),encoding="utf-8")
