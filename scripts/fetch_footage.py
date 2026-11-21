"""Download openly licensed real-world video clips from Wikimedia Commons for the eval set.

Only files whose licence is public domain, CC0 or CC BY (any version) are kept, so the
footage may be modified and the results published with attribution. Share-alike and
non-commercial licences are skipped. Every kept file is recorded in data/footage/manifest.csv
with its licence, author and source page. The clips themselves are never committed.

Standard library only, so it runs with any Python 3.9+:
    python3 scripts/fetch_footage.py            # up to 50 clips
    python3 scripts/fetch_footage.py --max 60
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "lookmatch-eval-fetcher/0.1 (https://github.com/omkar-k0/lookmatch; portfolio research)"
# Varied scenes: daylight, night, skin, foliage, water, sky, interiors, markets.
QUERIES = [
    "street walking", "city night", "beach waves", "forest walk", "mountain landscape", "sunset timelapse",
    "market people", "river boat", "snow", "desert", "festival crowd", "kitchen cooking", "train station",
    "flowers garden", "harbour", "portrait interview", "traffic road", "park children", "temple", "rain city",
]
OK_LICENCE = re.compile(r"^(cc0|public domain|pd|cc by [0-9.]+|cc-by-[0-9.]+|cc by|attribution)", re.I)
BAD_LICENCE = re.compile(r"(sa|nc|nd)\b|share|noncommercial|no derivatives", re.I)


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def candidates(query: str, limit: int = 20) -> list[dict]:
    data = get({
        "action": "query", "generator": "search", "gsrsearch": f"filetype:video {query}",
        "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
    })
    out = []
    for page in data.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        out.append({
            "title": page["title"], "url": info.get("url", ""), "page": info.get("descriptionurl", ""),
            "mime": info.get("mime", ""), "bytes": info.get("size", 0),
            "width": info.get("width", 0), "height": info.get("height", 0),
            "duration": float(info.get("duration") or 0),
            "licence": re.sub("<[^>]+>", "", lic).strip(),
            "author": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()[:200],
        })
    return out


def acceptable(c: dict) -> bool:
    lic = c["licence"]
    return (bool(OK_LICENCE.search(lic)) and not BAD_LICENCE.search(lic.replace("cc by", "")))\
        and c["mime"].startswith("video/") and c["width"] >= 1280 and 8 <= c["duration"] <= 900 \
        and 0 < c["bytes"] <= 250_000_000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=50)
    p.add_argument("--out", default="data/footage")
    a = p.parse_args()
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    seen = set()
    rows = []
    if manifest.exists():
        rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        seen = {r["title"] for r in rows}
    for q in QUERIES:
        if len(rows) >= a.max:
            break
        try:
            cands = candidates(q)
        except Exception as e:  # network hiccup: skip this query
            print(f"query failed: {q}: {e}")
            continue
        for c in cands:
            if len(rows) >= a.max or c["title"] in seen or not acceptable(c):
                continue
            seen.add(c["title"])
            ext = Path(urllib.parse.urlparse(c["url"]).path).suffix or ".webm"
            clip_id = f"c{len(rows) + 1:03d}"
            dst = out / "clips" / f"{clip_id}{ext}"
            print(f"[{len(rows) + 1}/{a.max}] {q}: {c['title']} ({c['licence']}, {c['bytes'] / 1e6:.0f} MB)", flush=True)
            try:
                req = urllib.request.Request(c["url"], headers={"User-Agent": UA})
                h = hashlib.sha256()
                with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        f.write(chunk)
                        h.update(chunk)
            except Exception as e:
                print(f"  download failed: {e}")
                dst.unlink(missing_ok=True)
                continue
            rows.append({"clip_id": clip_id, "file": dst.name, "title": c["title"], "query": q,
                         "licence": c["licence"], "author": c["author"], "source_page": c["page"],
                         "url": c["url"], "width": c["width"], "height": c["height"],
                         "duration_s": round(c["duration"], 1), "sha256": h.hexdigest()})
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            time.sleep(1)  # be polite to Wikimedia
    print(f"done: {len(rows)} clips in {out}/clips, manifest {manifest}")
    if len(rows) < 40:
        print("fewer than 40 clips: run again with other queries, or add own footage", file=sys.stderr)


if __name__ == "__main__":
    main()
