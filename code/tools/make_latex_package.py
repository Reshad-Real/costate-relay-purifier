"""Collect a self-contained LaTeX source package of the anonymised manuscript for the journal's typesetting
(1_LATEST_MANUSCRIPT_ISA_Transactions/Source_and_code/latex_source.zip, built in manuscript2/_latex_package): main.tex with flattened paths, the TikZ/CircuiTikz sources, photos,
generated macros and tables, the bibliography (.bib and .bst) and the data figures."""
import os, re, shutil, zipfile, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M2 = os.path.join(ROOT, "manuscript2")
DST = os.path.join(M2, "_latex_package")
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(os.path.join(DST, "generated"))
os.makedirs(os.path.join(DST, "figures"))
os.makedirs(os.path.join(DST, "photos"))
s = open(os.path.join(M2, "main.tex"), encoding="utf-8").read()
s = s.replace("{../figures2/}{../manuscript/Figures/}", "{figures/}")
s = s.replace("../manuscript/generated", "generated")
s = s.replace("../manuscript/refs_isa", "refs_isa")
open(os.path.join(DST, "main.tex"), "w", encoding="utf-8").write(s)
for f in ["title.tex", "isatrans.bst"] + [os.path.basename(p) for p in glob.glob(os.path.join(M2, "tikz_*.tex"))] + \
         [os.path.basename(p) for p in glob.glob(os.path.join(M2, "ctikz_*.tex"))]:
    shutil.copy(os.path.join(M2, f), os.path.join(DST, f))
for p in glob.glob(os.path.join(M2, "photos", "*.png")):
    shutil.copy(p, os.path.join(DST, "photos"))
gen = os.path.join(ROOT, "manuscript", "generated")
for p in glob.glob(os.path.join(gen, "*.tex")):
    shutil.copy(p, os.path.join(DST, "generated"))
shutil.copy(os.path.join(ROOT, "manuscript", "refs_isa.bib"), DST)
used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
for u in used:
    src = os.path.join(ROOT, "figures2", u if u.endswith(".pdf") else u + ".pdf")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST, "figures"))
zp = os.path.join(ROOT, "..", "1_LATEST_MANUSCRIPT_ISA_Transactions", "Source_and_code", "latex_source.zip")
os.makedirs(os.path.dirname(zp), exist_ok=True)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(DST):
        for f in files:
            p = os.path.join(base, f)
            z.write(p, os.path.relpath(p, DST))
print("package:", zp, sum(len(f) for _, _, f in os.walk(DST)), "files")
