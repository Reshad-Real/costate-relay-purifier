"""Geometric overlap check of the vector figures: (1) text labels whose boxes intersect, (2) line segments, box edges
or small markers that cross the inside of a text label. Prints one line per suspected overlap; renders each figure to a
PNG (argument 2, optional) for visual inspection.
Usage: python check_figure_overlaps.py <folder or pdf> [png_out_dir]"""
import sys, os, glob, itertools
import fitz

SHRINK = 0.22          # a crossing must reach this far inside the text box (fraction of its height)


def text_boxes(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"].strip()
                if t:
                    out.append((fitz.Rect(s["bbox"]), t))
    return out


def inner(r, f=SHRINK):
    d = f * r.height
    return fitz.Rect(r.x0 + d, r.y0 + d, r.x1 - d, r.y1 - d)


def seg_hits_rect(p, q, r):
    """Liang-Barsky clip of segment pq against rectangle r."""
    if r.is_empty:
        return False
    x0, y0, dx, dy = p.x, p.y, q.x - p.x, q.y - p.y
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - r.x0), (dx, r.x1 - x0), (-dy, y0 - r.y0), (dy, r.y1 - y0)):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def check(pdf):
    issues = []
    page = fitz.open(pdf)[0]
    tb = text_boxes(page)
    for (r1, t1), (r2, t2) in itertools.combinations(tb, 2):
        i = r1 & r2
        if not i.is_empty and i.get_area() > 0.12 * min(r1.get_area(), r2.get_area()):
            if abs(r1.y0 - r2.y0) < 0.5 and (r1.x1 <= r2.x0 + 1.0 or r2.x1 <= r1.x0 + 1.0):
                continue                       # adjacent spans of one text line
            issues.append(f"text/text   '{t1}'  x  '{t2}'")
    for d in page.get_drawings():
        segs, small = [], None
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1], it[2]))
            elif it[0] == "re":
                r = it[1]
                segs += [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
            elif it[0] == "qu":
                q = it[1]
                segs += [(q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)]
            elif it[0] == "c":
                segs.append((it[1], it[4]))
        rr = d["rect"]
        if max(rr.width, rr.height) < 6:   # markers and arrow heads
            small = rr
        for r, t in tb:
            ir = inner(r)
            if small is not None and not (small & ir).is_empty:
                issues.append(f"mark/text   marker at ({small.x0:.0f},{small.y0:.0f})  x  '{t}'")
                continue
            if any(seg_hits_rect(p, q, ir) for p, q in segs):
                issues.append(f"line/text   line near ({rr.x0:.0f},{rr.y0:.0f})-({rr.x1:.0f},{rr.y1:.0f})  x  '{t}'")
    return sorted(set(issues))


if __name__ == "__main__":
    target = sys.argv[1]
    pngdir = sys.argv[2] if len(sys.argv) > 2 else None
    pdfs = sorted(glob.glob(os.path.join(target, "*.pdf"))) if os.path.isdir(target) else [target]
    for pdf in pdfs:
        iss = check(pdf)
        print(f"== {os.path.basename(pdf)}: {len(iss)} suspected overlap(s)")
        for s in iss:
            print("   " + s)
        if pngdir:
            os.makedirs(pngdir, exist_ok=True)
            fitz.open(pdf)[0].get_pixmap(dpi=170).save(os.path.join(pngdir, os.path.basename(pdf)[:-4] + ".png"))
