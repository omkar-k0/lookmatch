"""Write the README eval tables from eval/results/*.json. Numbers are never typed by hand."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "none": "Do nothing",
    "reinhard": "Reinhard (unpaired)",
    "histogram": "Histogram (unpaired)",
    "paired_lut": "Paired LUT",
}


def table(res: dict) -> str:
    s = res["summary"]
    lines = [
        f"**{res['split'].capitalize()} looks** ({', '.join(res['looks'])}): {res['clips']} clips, "
        f"{res['pairs']} scored pairs, {res['milestone']}, {res['date']}, "
        f"frames `{res['frames_fingerprint']}`",
        "",
        "| Method | Mean CIEDE2000 | Pairs under 3.0 | Pixels under 3.0 |",
        "|---|---:|---:|---:|",
    ]
    for m, v in s["all"].items():
        lines.append(
            f"| {NAMES[m]} | {v['mean_de']:.2f} | {v['pairs_under_3']:.1%} | {v['px_under_3']:.1%} |"
        )
    lines += [
        "",
        "| Look | " + " | ".join(NAMES[m] for m in s["all"]) + " |",
        "|---|" + "---:|" * len(s["all"]),
    ]
    for look, b in s["by_look"].items():
        lines.append(f"| {look} | " + " | ".join(f"{b[m]['mean_de']:.2f}" for m in s["all"]) + " |")
    sig = s["paired_lut_signals"]
    lines += [
        "",
        f"Paired LUT, does anything predict its error? Correlation with test error: "
        f"self-check {sig['corr_self_check_vs_test_error']:+.2f}, "
        f"LUT coverage {sig['corr_coverage_vs_test_error']:+.2f}.",
    ]
    return "\n".join(lines)


def main() -> None:
    parts = []
    for name in ("dev", "heldout"):
        f = ROOT / "eval" / "results" / f"{name}.json"
        if f.exists():
            parts.append(table(json.loads(f.read_text())))
    body = "\n\n".join(parts) if parts else "_No eval results yet._"
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = f"<!-- EVAL:START -->\n{body}\n<!-- EVAL:END -->"
    if "<!-- EVAL:START -->" in text:
        text = re.sub(r"<!-- EVAL:START -->.*?<!-- EVAL:END -->", lambda _: block, text, flags=re.S)
    else:
        text += "\n## Eval\n\n" + block + "\n"
    readme.write_text(text)
    print(body)


if __name__ == "__main__":
    main()
