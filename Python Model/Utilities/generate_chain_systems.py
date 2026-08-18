"""Generate chain-length-capped reaction systems from a verified reference set.

The scaling ladder used to vary ENZYME COUNT, which turned out to be a bad axis: a
reachability analysis showed 78-81% of the state vector was structurally dead in the
small systems (3 enzymes: 81 of 104 species) versus 0.6% in FullFAS. Most equations
were never exercised, which is what made the finite solve unpredictable, and it is
also why ``initial_condition_floor`` inverted -- the floor sets EVERY zero entry
positive, so in a system that is 80% dead it injects mass into inert subnetworks.

This generates the replacement axis: MAXIMUM FATTY-ACID CHAIN LENGTH, with all 9
enzymes present at every rung, so network size varies without varying deadness.

    python Utilities/generate_chain_systems.py --check     # verify against existing
    python Utilities/generate_chain_systems.py             # write the ladder

Reference is ``Reactions/EC_FAS_ME1/C20+unsat`` -- the full system with zero-rate
reactions already removed (318 species / 328 reactions, versus FullFAS's 320 with 2
dead). It is NOT the top-level ``Reactions/EC_FAS_ME1/*.yaml`` set, which still
contains the zero-rate reactions.

WHY THIS IS A SCRIPT AND NOT A HAND EDIT
----------------------------------------
Hand-editing produced 2 broken directories out of 4 on the first attempt, in two
distinct ways, both of which this script is structured to make impossible:

1. LOUD: a chain list and its rate-value list must stay positionally aligned.
   ``_as_list_and_shared`` (reaction_model_builder.py:1189) raises when they differ.
   The trap is that several reactions INHERIT ``sat_chain_lengths`` from the file's
   ``defaults`` block while carrying their own per-chain value list sized to it
   (FabA's sat_rate_const_value=9, FabH's rvs_rate_const_value=9, FabZ's
   rate_const_value=9). Capping ``defaults`` therefore forces a matching cut in
   reactions that do not mention a chain list at all. This module computes ONE index
   mask per (reaction, branch) and applies it to the chain list and every sibling
   value list together, so they cannot drift apart.

2. SILENT AND WORSE: an unstripped ``unsat_chain_lengths`` in a saturated-only
   variant builds perfectly and just quietly emits unsaturated species. Eight
   locations were missed this way. Here it is one flag, applied uniformly.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ENZYME_TOKENS = ("FabA", "FabB", "FabD", "FabF", "FabG", "FabH", "FabI", "FabZ", "TesA")

# Which value lists ride along with which chain list. The builder resolves the sat
# branch as reaction.get("rate_const_value", reaction.get("sat_rate_const_value")),
# so the unprefixed names pair with SAT, never with unsat.
VALUE_KEYS = {
    "sat": ("sat_rate_const_value", "sat_rvs_rate_const_value",
            "rate_const_value", "rvs_rate_const_value"),
    "unsat": ("unsat_rate_const_value", "unsat_rvs_rate_const_value"),
}
CHAIN_KEY = {"sat": "sat_chain_lengths", "unsat": "unsat_chain_lengths"}

# Saturated-only rungs stop at C20; unsaturated chains start at C12, so shorter rungs
# have no unsaturated branch to include and the +unsat ladder starts there.
SAT_CAPS = (4, 6, 8, 10, 12, 14, 16, 18, 20)
UNSAT_CAPS = (12, 14, 16, 18, 20)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def variant_dir(cap: int, unsat: bool) -> str:
    return f"C{cap}+unsat" if unsat else f"C{cap}"


# --------------------------------------------------------------------------- io

def read_yaml(path: Path) -> tuple[str, dict]:
    """Return (header_comment_block, parsed_body).

    Every comment in the reference set sits in the leading block, so preserving that
    verbatim preserves all of them -- checked, not assumed (see --check).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    first_code = next((i for i, l in enumerate(lines)
                       if l.strip() and not l.lstrip().startswith("#")), 0)
    return "\n".join(lines[:first_code]), yaml.safe_load(text)


def write_yaml(path: Path, header: str, data: dict) -> None:
    body = yaml.safe_dump(
        data,
        sort_keys=False,          # key order is meaningful to a human reading these
        default_flow_style=None,  # inline scalar collections: [4, 6, 8] / {FabG: 1}
        width=10**6,              # never wrap a chain or rate list across lines
        allow_unicode=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((header + "\n\n" if header else "") + body, encoding="utf-8")


# ---------------------------------------------------------------------- masking

REFERENCE_CAP = 20        # the reference set's own top rung
MAX_TOKEN_OFFSET = 2      # condensation adds 2 carbons; nothing in the model adds more


def mask_for(chains: list | None, cap: int) -> list[int] | None:
    """Indices of chains that survive the cap, or None when there is no list here.

    A flat ``chain <= cap`` is WRONG, and quietly so. FabB and FabF bind an acyl and
    elongate it by two, so their ``defaults.sat_chain_lengths`` stops at 18 in the
    C20 reference, not 20 -- binding a C20 acyl would produce C22. Capping such a
    list flat at 18 for a C18 system would let FabB make C20 in a system named C18.

    So each list keeps its own distance from the top of the ladder: a list ending
    within one condensation step of the reference cap is SHIFTED with the cap, while
    a list ending well below it is a deliberate low-chain band (TesA's [4, 6, 8, 10]
    rate band) and is merely clipped.
    """
    if not isinstance(chains, list) or not chains:
        return [] if isinstance(chains, list) else None
    offset = REFERENCE_CAP - max(int(c) for c in chains)
    effective = cap - offset if offset <= MAX_TOKEN_OFFSET else cap
    return [i for i, c in enumerate(chains) if int(c) <= effective]


def take(seq: list, idx: list[int]) -> list:
    return [seq[i] for i in idx]


def apply_branch(reaction: dict, branch: str, default_chains: list | None,
                 cap: int, keep: bool) -> None:
    """Cap one branch of one reaction, chain list and value lists in lockstep.

    ``default_chains`` is the file-level list this reaction inherits when it does not
    declare its own -- the case that makes this non-local, since the reaction's value
    lists are sized to a list stored elsewhere.
    """
    own = reaction.get(CHAIN_KEY[branch])
    effective = own if isinstance(own, list) else default_chains

    if not keep:                              # saturated-only: erase the branch
        # Empty the lists in place rather than deleting the keys. Both are equivalent
        # to the builder (an empty unsat_chain_lengths skips the branch entirely), but
        # emptying is what the hand-verified C18/C20 directories do, so --check stays a
        # real equality test instead of drowning in presence/absence noise.
        if CHAIN_KEY[branch] in reaction:
            reaction[CHAIN_KEY[branch]] = []
        for k in VALUE_KEYS[branch]:
            if k in reaction:
                reaction[k] = []
        return

    idx = mask_for(effective, cap)
    if idx is None:
        return
    if isinstance(own, list):
        reaction[CHAIN_KEY[branch]] = take(own, idx)
    for k in VALUE_KEYS[branch]:
        v = reaction.get(k)
        # Scalars and length-1 lists broadcast in the builder; only a genuinely
        # per-chain list (one entry per effective chain) may be cut.
        if isinstance(v, list) and len(v) == len(effective) and len(v) > 1:
            reaction[k] = take(v, idx)


# ------------------------------------------------------------------- filtering

def species_of(reaction: dict) -> list[str]:
    return [str(s) for s in list(reaction.get("reactants") or {})
            + list(reaction.get("products") or {})]


def is_unsaturated(name: str) -> bool:
    return "unsat" in name or "cis" in name


def free_acyl_chains(reaction: dict) -> list[int]:
    """Chain numbers of FREE acyl species only.

    Enzyme-complex species carry an index inflated by ``token_offsets`` (binding a C3
    malonyl to a C10 acyl-enzyme gives C13_FabB_Act_MalACP_cis3), so capping on those
    would delete the C12 unsaturated entry point at the C12 rung. The cap governs free
    species; complexes above it are the same terminal overhang the reference model
    already has at C21/C22.
    """
    out = []
    for s in species_of(reaction):
        m = re.match(r"C(\d+)_", s)
        if m and not any(e in s for e in ENZYME_TOKENS):
            out.append(int(m.group(1)))
    return out


def drop_reason(reaction: dict, default_sat: list | None, default_unsat: list | None,
                cap: int, keep_unsat: bool) -> str | None:
    """Why this reaction cannot exist at this rung, or None to keep it."""
    if reaction.get("chain_template"):
        sat = reaction.get("sat_chain_lengths")
        uns = reaction.get("unsat_chain_lengths")
        sat = sat if isinstance(sat, list) else default_sat
        uns = uns if isinstance(uns, list) else default_unsat
        n_sat = len(mask_for(sat, cap) or [])
        n_uns = len(mask_for(uns, cap) or []) if keep_unsat else 0
        if n_sat == 0 and n_uns == 0:
            return "no chain survives the cap"
        return None

    names = species_of(reaction)
    if not keep_unsat and any(is_unsaturated(s) for s in names):
        return "unsaturated species in a saturated-only system"
    over = [c for c in free_acyl_chains(reaction) if c > cap]
    if over:
        return f"free acyl species above the cap (C{max(over)})"
    return None


# ------------------------------------------------------------------ generation

def build_variant(ref_dir: Path, cap: int, keep_unsat: bool) -> tuple[dict[str, tuple[str, dict]], list[str]]:
    """Return {filename: (header, data)} plus a log of what was dropped."""
    files: dict[str, tuple[str, dict]] = {}
    log: list[str] = []

    for path in sorted(ref_dir.glob("*.yaml")):
        header, data = read_yaml(path)
        defaults = data.get("defaults") or {}
        # Snapshot the ORIGINAL defaults: reactions are masked against the list they
        # were sized to, so this must be read before defaults itself is capped.
        d_sat = defaults.get("sat_chain_lengths")
        d_unsat = defaults.get("unsat_chain_lengths")

        kept = []
        for reaction in data.get("reactions", []):
            why = drop_reason(reaction, d_sat, d_unsat, cap, keep_unsat)
            if why:
                log.append(f"      drop {path.name}:{reaction.get('rxn_name')} -- {why}")
                continue
            apply_branch(reaction, "sat", d_sat, cap, keep=True)
            apply_branch(reaction, "unsat", d_unsat, cap, keep=keep_unsat)
            kept.append(reaction)

        if defaults:
            apply_branch(defaults, "sat", d_sat, cap, keep=True)
            # defaults holds no value lists, so erasing the branch is just the list
            if isinstance(d_unsat, list):
                defaults["unsat_chain_lengths"] = (
                    take(d_unsat, mask_for(d_unsat, cap)) if keep_unsat else [])

        if not kept:
            log.append(f"      DROP FILE {path.name} -- no reactions survive")
            continue
        data["reactions"] = kept
        files[path.name] = (header, data)

    return files, log


def validate(files: dict[str, tuple[str, dict]]) -> list[str]:
    """Reproduce the builder's own preconditions before writing anything to disk."""
    errs: list[str] = []
    for name, (_, data) in files.items():
        appearing = set()
        for r in data["reactions"]:
            appearing |= set(species_of(r))
        for enzyme in data.get("enzymes") or []:
            # reaction_model_builder.py:771 raises if a declared enzyme never appears.
            # Templated names only resolve at build time, so match on the token.
            if not any(enzyme == s or enzyme in s for s in appearing):
                errs.append(f"{name}: declared enzyme {enzyme} appears in no reaction")
        for r in data["reactions"]:
            for branch in ("sat", "unsat"):
                chains = r.get(CHAIN_KEY[branch])
                if not isinstance(chains, list):
                    continue
                for k in VALUE_KEYS[branch]:
                    v = r.get(k)
                    if isinstance(v, list) and len(v) > 1 and len(v) != len(chains):
                        errs.append(f"{name}:{r.get('rxn_name')} {k} len {len(v)} "
                                    f"!= {CHAIN_KEY[branch]} len {len(chains)}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="Reactions/EC_FAS_ME1/C20+unsat")
    ap.add_argument("--out-root", default="Reactions/EC_FAS_ME1")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff against what is on disk; write nothing")
    ap.add_argument("--verbose", action="store_true", help="log every dropped reaction")
    a = ap.parse_args()

    root = project_root()
    ref_dir = root / a.reference
    if not ref_dir.is_dir():
        print(f"FATAL: reference not found: {ref_dir}", file=sys.stderr)
        return 1

    targets = [(c, False) for c in SAT_CAPS] + [(c, True) for c in UNSAT_CAPS]
    failures = 0

    for cap, unsat in targets:
        name = variant_dir(cap, unsat)
        files, log = build_variant(ref_dir, cap, unsat)
        errs = validate(files)
        out_dir = root / a.out_root / name

        if errs:
            failures += 1
            print(f"  {name:<12} INVALID -- not written")
            for e in errs[:6]:
                print(f"      {e}")
            continue

        if a.check:
            status = compare(out_dir, files)
        else:
            for fname, (header, data) in files.items():
                write_yaml(out_dir / fname, header, data)
            # A rung with fewer enzymes than the reference is a silent regression of
            # the whole point of this axis, so say so rather than writing quietly.
            status = f"{len(files)} files" + ("" if len(files) == 9 else "  <-- FEWER THAN 9 ENZYMES")
        print(f"  {name:<12} {status}")
        if a.verbose:
            print("\n".join(log))

    return 1 if failures else 0


def compare(out_dir: Path, files: dict[str, tuple[str, dict]]) -> str:
    """Semantic diff against what is already on disk (parsed, not textual)."""
    if not out_dir.is_dir():
        return "absent"
    on_disk = {p.name: read_yaml(p)[1] for p in sorted(out_dir.glob("*.yaml"))}
    if set(on_disk) != set(files):
        return f"file set differs: disk={sorted(set(on_disk)-set(files))} gen={sorted(set(files)-set(on_disk))}"
    diffs = [n for n in files if on_disk[n] != files[n][1]]
    return "MATCHES" if not diffs else f"differs in {diffs}"


if __name__ == "__main__":
    raise SystemExit(main())
