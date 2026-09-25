"""Match a list of named parameter values against the rate constants in a reactions folder.

Answers one question: given a parameter list that uses a different naming convention from
the YAML files, which entries exist on both sides, and which are missing from each?

Matching is by VALUE, not by name, because the two conventions do not share names. The
list's enzyme number disambiguates values that several keys share (k2 FabD, k3 FabH,
k4 FabG, k5 FabZ, k6 FabI, k7 TesA, k8 FabF, k9 FabA, k10 FabB -> the letter used in the
YAML keys). Values are compared with a relative tolerance, since a pasted list usually
carries fewer digits than the file.

Input format is forgiving: one entry per line, `name value`, `name = value`, `name: value`
or `name,value`; blank lines and #-comments ignored.

Usage: python match_params_to_reactions.py params.txt [--system C20+unsat+FBinit] [--rtol 1e-6]
"""
import argparse
import collections
import re
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent.parent
ENZ_FOR_NUM = {"2": "D", "3": "H", "4": "G", "5": "Z", "6": "I", "7": "T", "8": "F",
               "9": "A", "10": "B"}
LINE = re.compile(r"^\s*([A-Za-z0-9_{}\[\].+-]+)\s*[=:,\t ]\s*([-+0-9.eE]+)\s*$")


def load_list(path):
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = LINE.match(line)
        if m:
            try:
                out.append((m.group(1), float(m.group(2))))
            except ValueError:
                pass
    return out


def load_reactions(folder):
    entries = []
    for f in sorted(folder.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text())
        dflt = doc.get("defaults") or {}
        for r in doc.get("reactions", []):
            chains = list(dflt.get("sat_chain_lengths") or []) if r.get("chain_template") else None
            for kk, vk in (("rate_const_key", "rate_const_value"),
                           ("rvs_rate_const_key", "rvs_rate_const_value")):
                key = r.get(kk)
                if key is None:
                    continue
                val = r.get(vk)
                if isinstance(val, list):
                    for i, x in enumerate(val):
                        entries.append({"key": key, "value": float(x), "file": f.stem,
                                        "rxn": r["rxn_name"],
                                        "chain": chains[i] if chains and i < len(chains) else None})
                elif val is not None:
                    entries.append({"key": key, "value": float(val), "file": f.stem,
                                    "rxn": r["rxn_name"], "chain": None})
    return entries


def enzyme_of_list_name(name):
    m = re.match(r"^k(\d+)", name)
    return ENZ_FOR_NUM.get(m.group(1)) if m else None


def enzyme_of_key(key):
    m = re.search(r"(?:kcat|kon|koff|kfwd|krvs)_([A-Za-z]+?)_", key)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list_file")
    ap.add_argument("--system", default="C20+unsat+FBinit")
    ap.add_argument("--rtol", type=float, default=1e-6)
    a = ap.parse_args()

    folder = PROJECT / "Reactions" / "EC_FAS_ME1" / a.system
    entries = load_reactions(folder)
    listed = load_list(a.list_file)
    print(f"{len(listed)} parameters in the list; {len(entries)} rate-constant entries "
          f"({len({e['key'] for e in entries})} unique keys) in {a.system}\n")

    matched_keys, unmatched_list, ambiguous = set(), [], []
    for name, val in listed:
        want = enzyme_of_list_name(name)
        hits = [e for e in entries
                if abs(e["value"] - val) <= a.rtol * max(abs(val), abs(e["value"]), 1e-30)]
        if want:
            by_enz = [e for e in hits if enzyme_of_key(e["key"]) == want]
            if by_enz:
                hits = by_enz
        keys = sorted({e["key"] for e in hits})
        if not keys:
            unmatched_list.append((name, val))
        else:
            matched_keys.update(keys)
            if len(keys) > 1:
                ambiguous.append((name, val, keys))

    unmatched_keys = sorted({e["key"] for e in entries} - matched_keys)
    print(f"=== matched: {len(matched_keys)} of {len({e['key'] for e in entries})} YAML keys ===")
    print(f"\n=== in the LIST but not found in {a.system} ({len(unmatched_list)}) ===")
    for n, v in unmatched_list:
        print(f"  {n:<34} {v!r}")
    print(f"\n=== in {a.system} but NOT in the list ({len(unmatched_keys)}) ===")
    by_file = collections.defaultdict(list)
    for e in entries:
        if e["key"] in unmatched_keys:
            by_file[e["file"]].append(e)
    for f in sorted(by_file):
        seen = sorted({(e["key"], e["value"]) for e in by_file[f]})
        print(f"  {f}:")
        for k, v in seen:
            print(f"     {k:<40} {v}")
    if ambiguous:
        print(f"\n=== value matched more than one key ({len(ambiguous)}) ===")
        for n, v, ks in ambiguous:
            print(f"  {n} = {v}  ->  {ks}")


if __name__ == "__main__":
    main()
