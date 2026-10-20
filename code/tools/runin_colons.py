"""House style: run-in paragraph and bullet names end with a colon, "\\emph{Name:}", never "\\emph{Name.}".
Rewrites every \\emph{...} whose text ends with a period in the given LaTeX files and reports the count."""
import re, sys

for path in sys.argv[1:]:
    s = open(path, encoding="utf-8").read()
    s, k = re.subn(r"\\emph\{([^{}]*?)\.\}", lambda m: "\\emph{" + m.group(1) + ":}", s)
    open(path, "w", encoding="utf-8").write(s)
    print(f"{path}: {k} run-in name(s) changed to 'Name:'")
