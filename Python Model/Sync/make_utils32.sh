#!/bin/bash
# Build a 32-bit copy of Utilities/ as Utilities32/.
#
# It does NOT rewrite float64 -> float32. Hardcoding float32 fails: pytensor and
# jax each carry their own default precision, and pinning one side to float32
# while the other still resolves its default produces a mixed graph that dies
# building the custom_vjp gradient bridge.
#
# Instead it STRIPS the explicit precision and lets the defaults decide. With
# jax_enable_x64=False (jax's default) and pytensor's floatX default, everything
# lands on float32 consistently, and there is nothing left to disagree.
#
#   dtype=jnp.float64 / "float64" / np.float64   -> removed
#   pt.cast(x, "float64")                        -> x
#   pt.matrix(dtype="float64")                   -> pt.matrix()
#   jax_enable_x64 True                          -> False
#
# Usage: Sync/make_utils32.sh     (run on whichever host will use it)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1   # project dir, wherever this repo lives

rm -rf Utilities32
cp -r Utilities Utilities32
rm -rf Utilities32/__pycache__
# precision_probe.py is the measuring instrument, not model code -- converting it
# would rewrite the tool we use to judge the conversion.
rm -f Utilities32/precision_probe.py

python3 - <<'PY'
import pathlib, re

changed = {}
for p in sorted(pathlib.Path("Utilities32").glob("*.py")):
    s = orig = p.read_text()

    # let jax pick float32 by default rather than being told float64
    s = s.replace('jax.config.update("jax_enable_x64", True)',
                  'jax.config.update("jax_enable_x64", False)')

    # pt.cast(<expr>, "float64") -> <expr>   (innermost-first, handles nesting)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'pt\.cast\(((?:[^()]|\([^()]*\))*?),\s*["\']float64["\']\s*\)', r'\1', s)

    # drop the dtype argument wherever it is given explicitly
    s = re.sub(r',\s*dtype\s*=\s*(?:jnp|np)\.float64', '', s)
    s = re.sub(r',\s*dtype\s*=\s*["\']float64["\']', '', s)
    s = re.sub(r'dtype\s*=\s*(?:jnp|np)\.float64\s*,\s*', '', s)
    s = re.sub(r'dtype\s*=\s*["\']float64["\']\s*,\s*', '', s)
    # .astype(np.float64) -> dropped entirely (numpy's own default is float64,
    # so removing the call changes nothing except that nothing FORCES 64 anymore)
    s = re.sub(r'\.astype\(\s*(?:jnp|np)\.float64\s*\)', '', s)
    s = re.sub(r'\.astype\(\s*["\']float64["\']\s*\)', '', s)

    # ...including when it is the ONLY argument: pt.matrix(dtype="float64") -> pt.matrix()
    s = re.sub(r'\(\s*dtype\s*=\s*(?:jnp\.|np\.)?["\']?float64["\']?\s*\)', '()', s)

    if s != orig:
        p.write_text(s)
        changed[p.name] = len(re.findall(r'float64', orig))

for name, n in changed.items():
    print(f"  {name}: {n} float64 site(s) stripped")

# every file must still parse -- a regex that eats a paren would otherwise only
# surface much later, inside a GPU job
import ast
bad = []
for p in sorted(pathlib.Path("Utilities32").glob("*.py")):
    try:
        ast.parse(p.read_text())
    except SyntaxError as e:
        bad.append(f"{p.name}:{e.lineno}: {e.msg}")
print("=== syntax check ===")
print("  all files parse" if not bad else "  BROKEN:\n    " + "\n    ".join(bad))
PY

echo "=== leftover float64 (should be none) ==="
grep -rn "float64" Utilities32/*.py || echo "  none"
echo "=== x64 setting ==="
grep -rhn "enable_x64" Utilities32/*.py | sort -u
