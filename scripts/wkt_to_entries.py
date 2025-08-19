# scripts/wkt_to_entries.py
# Converts Wiktextract JSONL(.gz) -> docs/entries.csv with columns term,text
# Supports: senses[*].glosses, senses[*].examples[{text}], top-level translations, sounds/ipa

import argparse, csv, gzip, io, json, sys, textwrap
from pathlib import Path

DEFAULT_LANGS = {"en", "uk", "pl"}
LANG_NAME_TO_CODE = {"english":"en", "ukrainian":"uk", "polish":"pl"}

def norm_lc(obj):
    lc = (obj.get("lang_code") or "").strip().lower()
    if lc:
        return lc
    ln = (obj.get("lang") or "").strip().lower()
    return LANG_NAME_TO_CODE.get(ln, "")

def first_gloss(sense: dict) -> str | None:
    # in your dump, definitions are in "glosses"
    g1 = sense.get("gloss")
    if isinstance(g1, str) and g1.strip():
        return g1.strip()
    gl = sense.get("glosses")
    if isinstance(gl, list):
        for g in gl:
            if isinstance(g, str) and g.strip():
                return g.strip()
    return None

def collect_examples(sense: dict, need=2):
    out = []
    ex = sense.get("examples") or []
    for item in ex:
        if isinstance(item, dict):
            t = item.get("text")
        else:
            t = item
        if isinstance(t, str) and t.strip():
            # slightly shorten very long quotes
            out.append(textwrap.shorten(t.strip().replace("\n"," "), width=220))
            if len(out) >= need:
                break
    return out

def collect_ipa(obj: dict):
    vals = []
    for s in obj.get("sounds") or []:
        ipa = s.get("ipa")
        if isinstance(ipa, str) and ipa.strip():
            vals.append(ipa.strip())
    return vals

def collect_translations(obj: dict, targets={"uk","pl"}, limit=8):
    # in your dump, "translations" are at the top level
    trs = []
    for tr in obj.get("translations") or []:
        if not isinstance(tr, dict):
            continue
        lc = (tr.get("lang_code") or "").strip().lower()
        w  = (tr.get("word") or "").strip()
        if lc in targets and w:
            trs.append(f"{lc}:{w}")
            if len(trs) >= limit:
                break
    return trs

def pack_row(obj, keep_langs):
    lc = norm_lc(obj)
    if keep_langs and keep_langs != {"any"} and lc not in keep_langs:
        return None

    term = (obj.get("word") or obj.get("title") or "").strip()
    if not term:
        return None

    senses = obj.get("senses") or []
    gloss = None
    examples = []
    for s in senses:
        if not gloss:
            gloss = first_gloss(s)
        if len(examples) < 2:
            examples += [e for e in collect_examples(s, need=2-len(examples)) if e]
        if gloss and len(examples) >= 2:
            break
    if not gloss:
        return None

    ipa_list = collect_ipa(obj)
    trs = collect_translations(obj) if lc == "en" else []

    parts = [f"{(lc or '').upper()}: {gloss}"]
    if ipa_list:
        parts.append(f"IPA: /{', '.join(ipa_list)}/")
    if examples:
        parts.append("Examples:\n- " + "\n- ".join(examples))
    if trs:
        parts.append("Translations: " + ", ".join(trs))
    text = " | ".join(parts)

    return term, text

def open_maybe_gz(p: Path):
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore")
    # sometimes gz file without extension
    with open(p, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.GzipFile(fileobj=open(p, "rb")), encoding="utf-8", errors="ignore")
    return open(p, "r", encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/wiktextract.jsonl.gz")
    ap.add_argument("--out", dest="out", default="docs/entries.csv")
    ap.add_argument("--langs", default="en,uk,pl", help="e.g. en,uk,pl or any")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    keep_langs = set(x.strip().lower() for x in args.langs.split(",")) if args.langs else set(DEFAULT_LANGS)
    if keep_langs == {""}:
        keep_langs = set(DEFAULT_LANGS)

    inp = Path(args.inp); out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert inp.exists(), f"File not found: {inp}"

    seen = written = 0
    print(f"→ Reading {inp} and writing to {out} …", file=sys.stderr)
    with open_maybe_gz(inp) as fin, open(out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["term","text"])
        for i, line in enumerate(fin, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            seen += 1
            row = pack_row(obj, keep_langs)
            if row:
                w.writerow(row)
                written += 1
                if args.limit and written >= args.limit:
                    break
            if i % 200_000 == 0:
                print(f"… processed: {i:,}, written: {written:,}", file=sys.stderr)
    print(f"✅ Done. Read: {seen:,}, written: {written:,}", file=sys.stderr)

if __name__ == "__main__":
    main()
