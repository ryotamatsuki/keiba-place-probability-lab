from urllib.request import Request, urlopen
from html.parser import HTMLParser
from pathlib import Path

URL="https://www.jra.go.jp/datafile/seiseki/report/2025.html"

class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag=="a":
            d=dict(attrs)
            if d.get("href"):
                self.links.append(d["href"])

req=Request(URL,headers={"User-Agent":"Mozilla/5.0"})
with urlopen(req,timeout=60) as r:
    html=r.read().decode("utf-8","replace")
p=P(); p.feed(html)
pdf=[x for x in p.links if ".pdf" in x.lower()]
Path("docs/JRA_2025_PDF_LINK_SAMPLES.txt").write_text("\n".join(pdf[:80])+"\n",encoding="utf-8")
print(len(pdf))
