"""Replace \\footnotesize by \\figfont (a fixed 8-pt font defined in the manuscript preamble) in all TikZ and
CircuiTikz figure sources, so diagrams render identically in the 12-pt review and the 10-pt final layout."""
import glob, os
BS = chr(92)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manuscript2")
n = 0
for p in glob.glob(os.path.join(root, "tikz_*.tex")) + glob.glob(os.path.join(root, "ctikz_*.tex")):
    s = open(p, encoding="utf-8").read()
    t = s.replace(BS + "footnotesize", BS + "figfont")
    if t != s:
        n += s.count(BS + "footnotesize")
        open(p, "w", encoding="utf-8").write(t)
print("replaced", n)
