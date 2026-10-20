"""Build the manuscript in the Elsevier elsarticle template (elsarticle-template-num: final, 5p, times, two-column,
numbered citations in the ISA Transactions style isatrans.bst) without any author information, from the master source
manuscript2/main.tex. Table captions stay above the tables (every \\caption precedes its table body).

Output: a self-contained LaTeX folder manuscript2/_template_package (main.tex + figures, TikZ sources, photos,
generated macros/tables, bibliography), compiled there, and copied to
  1_LATEST_MANUSCRIPT_ISA_Transactions/Manuscript_Elsevier_template_no_authors.pdf
  1_LATEST_MANUSCRIPT_ISA_Transactions/Source_and_code/latex_source_Elsevier_template.zip
"""
import os, re, glob, shutil, zipfile, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M2 = os.path.join(ROOT, "manuscript2")
DST = os.path.join(M2, "_template_package")
LATEST = os.path.join(ROOT, "..", "1_LATEST_MANUSCRIPT_ISA_Transactions")
TEX = os.environ.get("MIKTEX_BIN", r"C:\Users\Asus ROG Strix G16\AppData\Local\Programs\MiKTeX\miktex\bin\x64")
B = "\\"

src = open(os.path.join(M2, "main.tex"), encoding="utf-8").read()
title = re.search(r"\\newcommand\{\\PaperTitle\}\{(.*)\}\s*$", open(os.path.join(M2, "title.tex"), encoding="utf-8").read().strip()).group(1)


def between(a, b, s=src):
    i = s.index(a) + len(a)
    return s[i:s.index(b, i)]


abstract = between(r"\begin{abstract}", r"\end{abstract}").strip()
keywords = between(r"\begin{keyword}", r"\end{keyword}").strip()
body = between(r"\end{frontmatter}", r"\bibliographystyle").strip("\n")

# anonymised branch of the conditional end matter (no author-identifying statements)
body = re.sub(r"\\ifdefined\\FINALVERSION\n.*?\\else\n(.*?)\\fi(?:\n|$)", lambda m: m.group(1), body, flags=re.S)
assert "FINALVERSION" not in body
# author-year citations: parenthetical by default, textual in the 'Study' column of Table 1
# numbered citations: \cite is kept exactly as in the master source

highlights = [l[2:].strip() for l in open(os.path.join(M2, "highlights.txt"), encoding="utf-8") if l.startswith("- ")]

HEADER = r"""%%
%% Copyright 2007-2020 Elsevier Ltd
%%
%% This file is part of the 'Elsarticle Bundle'.
%% ---------------------------------------------
%%
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.2 of this license or (at your option) any
%% later version.  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%%
%% The list of all files belonging to the 'Elsarticle Bundle' is
%% given in the file `manifest.txt'.
%%
%% Template article for Elsevier's document class `elsarticle'
%% with numbered style bibliographic references
%%
%% GENERATED FILE: produced from manuscript2/main.tex by code/tools/make_template_version.py.
%% Edit the master source and rebuild instead of editing this file.

%\documentclass[preprint,12pt]{elsarticle}

%% Use the option review to obtain double line spacing
%% \documentclass[preprint,review,12pt]{elsarticle}

%% Use the options 1p,twocolumn; 3p; 3p,twocolumn; 5p; or 5p,twocolumn
%% for a journal layout:
%% \documentclass[final,1p,times]{elsarticle}
%% \documentclass[final,1p,times,twocolumn]{elsarticle}
%% \documentclass[final,3p,times]{elsarticle}
%% \documentclass[final,3p,times,twocolumn]{elsarticle}
%% \documentclass[final,5p,times]{elsarticle}
 \documentclass[final,5p,times,twocolumn]{elsarticle}

%% For including figures, graphicx.sty has been loaded in
%% elsarticle.cls. If you prefer to use the old commands
%% please give \usepackage{epsfig}

%% The amssymb package provides various useful mathematical symbols
\usepackage{amssymb}
%% The amsthm package provides extended theorem environments
\usepackage{amsthm}

%% The lineno packages adds line numbers. Start line numbering with
%% \begin{linenumbers}, end it with \end{linenumbers}. Or switch it on
%% for the whole article with \linenumbers.
%% \usepackage{lineno}

%% Packages used by this manuscript
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{bm}
\graphicspath{{figures/}}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{microtype}
\usepackage{adjustbox}
\usepackage{placeins}
\usepackage[siunitx,americancurrents]{circuitikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shadows}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\setlength{\emergencystretch}{2em}
\raggedbottom
\renewcommand{\dbltopfraction}{0.75}
\renewcommand{\textfraction}{0.1}
%% Table captions are placed above the tables, figure captions below the figures.
\captionsetup{font=footnotesize,labelfont=bf,skip=4pt}
\captionsetup[table]{position=top}

%% You might want to define your own abbreviated commands for common used terms, e.g.:
\newcommand{\figfont}{\fontsize{8}{9.5}\selectfont}   % all TikZ/CircuiTikz text: 8 pt
\newcommand{\singlefig}{\linewidth}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newtheoremstyle{colonplain}{\topsep}{\topsep}{\itshape}{0pt}{\bfseries}{:}{5pt plus 1pt minus 1pt}{}
\theoremstyle{colonplain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{assumption}{Assumption}
\newtheorem{remark}{Remark}
\renewcommand{\qedsymbol}{}   % no end-of-proof box
\renewcommand{\proofname}{Proof:}   % run-in heading "Proof:" (amsthm then adds no period)
\newcommand{\U}{\mathcal U}
\newcommand{\Z}{\mathcal Z}
\renewcommand{\P}{\mathcal P}
\newcommand{\GEN}{generated}
\input{generated/macros.tex}
\input{generated/fallback.tex}
\newcommand{\gentab}[1]{\IfFileExists{\GEN/tab_#1.tex}{\adjustbox{max width=\linewidth}{\input{\GEN/tab_#1.tex}}}{\textbf{[table #1 missing]}}}

\journal{ISA Transactions}
%% numbered citations: sorted, with consecutive numbers as ranges, e.g. [5-7]
\biboptions{sort&compress}


\begin{document}

\begin{frontmatter}

%% Title, authors and addresses
%% (author information removed for double-anonymised review; see the separate title page)

\title{TITLE}

\begin{abstract}
%% Text of abstract
ABSTRACT
\end{abstract}

%%Graphical abstract
%\begin{graphicalabstract}
%\includegraphics{graphical_abstract}
%\end{graphicalabstract}

%%Research highlights
%\begin{highlights}
HIGHLIGHTS
%\end{highlights}

\begin{keyword}
%% keywords here, in the form: keyword \sep keyword, up to a maximum of 6 keywords
KEYWORDS

%% PACS codes here, in the form: \PACS code \sep code

%% MSC codes here, in the form: \MSC code \sep code
%% or \MSC[2008] code \sep code (2000 is the default)

\end{keyword}


\end{frontmatter}

%\tableofcontents

%% \linenumbers

%% main text

BODY

%% If you have bibdatabase file and want bibtex to generate the
%% bibitems, please use
%%
%% numbered style of ISA Transactions (references numbered in order of citation, abbreviated journal
%% names, DOIs); Elsevier's generic numbered style is elsarticle-num
\bibliographystyle{isatrans}
\bibliography{refs_isa}

%% else use the following coding to input the bibitems directly in the
%% TeX file.

%%\begin{thebibliography}{00}

%% \bibitem{label}
%% Text of bibliographic item


%%\end{thebibliography}

\end{document}

\endinput
%%
%% End of file `elsarticle-template-num.tex'.
"""

doc = HEADER
for key, val in [("TITLE", title), ("ABSTRACT", abstract), ("KEYWORDS", keywords),
                 ("HIGHLIGHTS", "\n".join("%\\item " + h for h in highlights)), ("BODY", body)]:
    assert doc.count(key) == 1, key
    doc = doc.replace(key, val)

# ------------------------------------------------------------------ self-contained package
os.makedirs(DST, exist_ok=True)
for p in os.listdir(DST):                     # empty the build folder of its previous output
    q = os.path.join(DST, p)
    shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
for d in ("generated", "figures", "photos"):
    os.makedirs(os.path.join(DST, d))
open(os.path.join(DST, "main.tex"), "w", encoding="utf-8").write(doc)
for p in glob.glob(os.path.join(M2, "tikz_*.tex")) + glob.glob(os.path.join(M2, "ctikz_*.tex")):
    shutil.copy(p, DST)
for p in glob.glob(os.path.join(M2, "photos", "*.png")):
    shutil.copy(p, os.path.join(DST, "photos"))
for p in glob.glob(os.path.join(ROOT, "manuscript", "generated", "*.tex")):
    shutil.copy(p, os.path.join(DST, "generated"))
shutil.copy(os.path.join(ROOT, "manuscript", "refs_isa.bib"), DST)
shutil.copy(os.path.join(M2, "isatrans.bst"), DST)
shutil.copy(os.path.join(M2, "graphical_abstract.pdf"), DST)        # for the (commented) graphicalabstract block
active = "\n".join(l for l in doc.splitlines() if not l.lstrip().startswith("%"))
for u in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", active)):
    for base in (os.path.join(ROOT, "figures2"), os.path.join(ROOT, "manuscript", "Figures")):
        f = os.path.join(base, u if os.path.splitext(u)[1] else u + ".pdf")
        if os.path.exists(f):
            shutil.copy(f, os.path.join(DST, "figures"))
            break
    else:
        raise FileNotFoundError(u)

# ------------------------------------------------------------------ compile
env = dict(os.environ, PATH=TEX + os.pathsep + os.environ["PATH"])
run = lambda *a: subprocess.run(list(a), cwd=DST, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("bibtex", "main")
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("pdflatex", "-interaction=nonstopmode", "main.tex")
log = open(os.path.join(DST, "main.log"), encoding="latin-1").read()
pages = re.search(r"Output written on main\.pdf \((\d+) page", log)
errors = re.findall(r"(?m)^! .*", log)
undef = re.findall(r"Warning: (?:Reference|Citation) `([^']+)' .*undefined", log)
print(f"template version: {pages.group(1) if pages else 'NO OUTPUT'} pages, {len(errors)} errors, "
      f"{len(set(undef))} undefined refs/cites, {log.count('Overfull')} overfull boxes")

# ------------------------------------------------------------------ deliver
os.makedirs(os.path.join(LATEST, "Source_and_code"), exist_ok=True)
try:
    shutil.copy(os.path.join(DST, "main.pdf"), os.path.join(LATEST, "Manuscript_Elsevier_template_no_authors.pdf"))
except PermissionError:
    print("WARNING: Manuscript_Elsevier_template_no_authors.pdf is open in another program and was not updated; "
          "close it and rerun this script")
zp = os.path.join(LATEST, "Source_and_code", "latex_source_Elsevier_template.zip")
keep = lambda f: not re.search(r"\.(aux|log|blg|out|spl|synctex\.gz|fls|fdb_latexmk)$", f) and f != "main.pdf"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(DST):
        for f in files:
            rel = os.path.relpath(os.path.join(base, f), DST).replace(os.sep, "/")
            if keep(rel) or rel == "main.bbl":
                z.write(os.path.join(base, f), rel)
print("written:", os.path.normpath(zp))
