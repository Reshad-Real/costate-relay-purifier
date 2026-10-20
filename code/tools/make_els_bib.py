"""Build manuscript/refs_els.bib for elsarticle-num: drop the note={\\url{https://doi.org/..}} duplicates
(elsarticle-num prints the doi field itself); keep notes of entries that have no DOI."""
import os, re

BS = chr(92)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manuscript")
s = open(os.path.join(root, "refs.bib"), encoding="utf-8").read()
pat = re.compile(r",\s*note = \{" + re.escape(BS) + r"url\{https://doi\.org/[^}]*\}\}")
out, n = [], 0
for entry in re.split(r"(?=^@)", s, flags=re.M):
    if re.search(r"\bdoi\s*=", entry):
        e2 = pat.sub("", entry)
        n += e2 != entry
        entry = e2
    out.append(entry)
txt = "".join(out)
open(os.path.join(root, "refs_els.bib"), "w", encoding="utf-8").write(txt)
print("doi notes removed:", n, "entries:", len(re.findall(r"^@", txt, flags=re.M)))
