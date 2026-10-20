"""Compare the macros used in main.tex between an old and the current macros.tex (to re-check text claims)."""
import re, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
old_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript2", "_latex_package", "generated", "macros.tex")
new_p = os.path.join(ROOT, "manuscript", "generated", "macros.tex")


def load(p):
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", open(p, encoding="utf-8").read()))


old, new = load(old_p), load(new_p)
used = set(re.findall(r"\\([A-Za-z]+)", open(os.path.join(ROOT, "manuscript2", "main.tex"), encoding="utf-8").read()))
keys = sorted(k for k in used if k in old or k in new)
for k in keys:
    o, n = old.get(k, "-"), new.get(k, "-")
    flag = "" if o == n else "  *"
    print(f"{k:24s} {o:>14s} -> {n:<14s}{flag}")
