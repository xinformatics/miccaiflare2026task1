"""Pull task + reported metrics out of every FLARE 2025 paper in 2025papers/.

Ranking them on one number would be wrong without this step: the 16 papers are
spread across three different challenge tasks with different targets and different
metrics, so a naive comparison mixes lesion DSC with organ DSC.

Extracts, per paper: title, which task it addresses, and the DSC/NSD figures it
reports for public and online (hidden) validation.
"""
import re
import sys
from pathlib import Path

PAPERS = Path("/media/shashank/NewVolume/PancancerCTSeg/2025papers")


def text_of(pdf):
    import subprocess
    r = subprocess.run(["pdftotext", "-q", str(pdf), "-"],
                       capture_output=True, text=True)
    return r.stdout


def title_of(txt):
    for line in txt.splitlines():
        s = line.strip()
        if len(s) > 18 and not s.lower().startswith(("abstract", "keywords")):
            return s[:78]
    return "?"


def task_of(txt):
    """Which FLARE 2025 task is this paper for?"""
    t = txt.lower()
    sig = [
        ("Task1 pan-cancer CT", ["pan-cancer", "pancancer", "whole-body tumor",
                                 "whole body cancer", "lesion segmentation"]),
        ("Task2 CPU organ seg", ["cpu", "laptop", "without using gpu", "non-gpu"]),
        ("Task3 UDA MRI/PET", ["mri", "pet", "domain adaptation", "cross-modality",
                               "unsupervised domain"]),
    ]
    scores = {name: sum(t.count(k) for k in keys) for name, keys in sig}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "?", scores


def metrics_of(txt):
    """Find reported DSC / NSD percentages near validation wording."""
    out = {}
    # e.g. "53.48 ± 38.06" or "DSC of 79.56%" or "59% DSC"
    pats = [
        (r"(\d{2}\.\d{1,2})\s*[±\+\-]\s*\d{1,2}\.\d{1,2}", "pm"),
        (r"DSC[^0-9]{0,18}(\d{2}\.\d{1,2})\s*%", "dsc"),
        (r"(\d{2})\s*%\s*DSC", "dsc"),
        (r"NSD[^0-9]{0,18}(\d{2}\.\d{1,2})\s*%", "nsd"),
        (r"(\d{2})\s*%\s*NSD", "nsd"),
    ]
    for pat, kind in pats:
        vals = [float(m) for m in re.findall(pat, txt)]
        if vals:
            out.setdefault(kind, []).extend(vals)
    return out


def main():
    rows = []
    for pdf in sorted(PAPERS.glob("*.pdf"),
                      key=lambda p: int(re.match(r"(\d+)", p.name).group(1))):
        txt = text_of(pdf)
        if not txt.strip():
            rows.append((pdf.name, "?", "NO TEXT (scanned?)", {}, ""))
            continue
        task, _ = task_of(txt)
        m = metrics_of(txt)
        # grab the sentence that states the headline result, for manual checking
        claim = ""
        for s in re.split(r"(?<=[.])\s+", txt):
            if re.search(r"(achiev|obtain|reach|score|lead|rank)", s, re.I) and \
               re.search(r"DSC|Dice|NSD", s, re.I) and re.search(r"\d\d", s):
                claim = " ".join(s.split())[:190]
                break
        rows.append((pdf.name, task, title_of(txt), m, claim))

    for name, task, title, m, claim in rows:
        print(f"\n### {name}")
        print(f"  task : {task}")
        print(f"  title: {title}")
        if m:
            for k, v in m.items():
                print(f"  {k:4s}: {sorted(set(v), reverse=True)[:8]}")
        if claim:
            print(f"  claim: {claim}")


if __name__ == "__main__":
    main()
