"""Capitalise the first word after a prose colon ("Name: Text", "...as follows: The ...") in the manuscript sources.

Skipped: colons in math, comments, label/ref/cite/url keys, clock times, TikZ option syntax (a colon with no space
after it, e.g. "south east:five"), and colons followed by math, a number or a word that is already capitalised.
Usage: python capitalize_after_colons.py [--apply]   (without --apply it only lists the changes)."""
import re, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M2 = os.path.join(ROOT, "manuscript2")
FILES = [os.path.join(M2, f) for f in ["main.tex", "supplement.tex", "title_page.tex", "graphical_abstract.tex",
                                       "title.tex", "hardware_validation.tex", "response_review_v1.tex"]]
FILES += sorted(glob.glob(os.path.join(M2, "*tikz_*.tex")))
KEYS = (r"label|ref|eqref|cref|Cref|autoref|cite|citep|citet|url|href|input|include|includegraphics|gentab|usepackage"
        r"|usetikzlibrary|bibliography|bibliographystyle|newcommand|renewcommand|providecommand|newtheoremstyle"
        r"|captionsetup|definecolor|colorlet|tikzset")


def mask(s):
    """Blank out everything that is not prose, keeping positions."""
    out = list(s)

    def blank(a, b):
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = " "
    for m in re.finditer(r"(?<!\\)%[^\n]*", s):            # comments (to the end of the line only)
        blank(m.start(), m.end())
    pats = [r"\\\[.*?\\\]", r"\$\$.*?\$\$", r"(?<!\\)\$.*?(?<!\\)\$",
            r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
            r"\\begin\{(Verbatim|verbatim)\}.*?\\end\{\1\}",           # literal file contents stay as they are
            r"\\(?:" + KEYS + r")\*?(?:\[[^\]]*\])?\{[^}]*\}",
            r"https?://\S+", r"\d:\d"]
    for pat in pats:
        for m in re.finditer(pat, s, flags=re.S):
            blank(m.start(), m.end())
    return "".join(out)


# colon, then at least one separator (space, newline, ~, closing brace of a run-in \emph{Name:}), then optional
# opening formatting, then a lower-case word
PAT = re.compile(r":((?:\}|\s|~)(?:\s|~|\\,|\}|\{|\\emph\{|\\textit\{|\\textbf\{)*)([a-z])")


def process(path, apply):
    s = open(path, encoding="utf-8").read()
    ms = mask(s)
    edits = []
    for m in PAT.finditer(ms):
        i = m.start(2)
        if s[m.start():i + 1] != ms[m.start():i + 1]:   # math or a key between the colon and the word, or in it
            continue
        edits.append(i)
        line = s.count("\n", 0, m.start()) + 1
        ctx = s[max(0, m.start() - 45):i + 25].replace("\n", " ")
        print(f"{os.path.basename(path)}:{line}: ...{ctx}...")
    if apply and edits:
        t = list(s)
        for i in edits:
            t[i] = t[i].upper()
        open(path, "w", encoding="utf-8").write("".join(t))
    return len(edits)


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    n = sum(process(f, apply) for f in FILES if os.path.exists(f))
    print(f"{n} {'changed' if apply else 'to change'}")
