"""Build manuscript/refs_isa.bib for the ISA Transactions reference style (isatrans.bst = vancouver.bst with
DOIs printed as https://doi.org/...): refs_els.bib + refs_isa_extra.bib, journal names abbreviated to
ISO 4 / LTWA (without full stops, as in the journal's examples), note={\\url{doi}} duplicates dropped."""
import os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "manuscript")
ABBR = {
    "ACS ES\\&T Air": "ACS ES T Air",
    "AIChE Journal": "AIChE J",
    "American Industrial Hygiene Association Journal": "Am Ind Hyg Assoc J",
    "Annals of Occupational Hygiene": "Ann Occup Hyg",
    "Annual Reviews in Control": "Annu Rev Control",
    "Applied Energy": "Appl Energy",
    "Annals of Discrete Mathematics": "Ann Discrete Math",
    "Atmospheric Environment": "Atmos Environ",
    "Automatica": "Automatica",
    "Building and Environment": "Build Environ",
    "Control Engineering Practice": "Control Eng Pract",
    "Critical Reviews in Toxicology": "Crit Rev Toxicol",
    "Energy and Buildings": "Energy Build",
    "European Journal of Control": "Eur J Control",
    "Foundations and Trends in Systems and Control": "Found Trends Syst Control",
    "IEEE Control Systems Magazine": "IEEE Control Syst Mag",
    "IEEE Control Systems Letters": "IEEE Control Syst Lett",
    "IEEE Transactions on Control Systems Technology": "IEEE Trans Control Syst Technol",
    "IFAC-PapersOnLine": "IFAC-PapersOnLine",
    "IEEE Open Journal of Industry Applications": "IEEE Open J Ind Appl",
    "IEEE Transactions on Automatic Control": "IEEE Trans Autom Control",
    "IEEE Transactions on Industrial Electronics": "IEEE Trans Ind Electron",
    "IEEE Transactions on Power Electronics": "IEEE Trans Power Electron",
    "Indoor Air": "Indoor Air",
    "International Journal of Robust and Nonlinear Control": "Int J Robust Nonlinear Control",
    "ISA Transactions": "ISA Trans",
    "Journal of Building Engineering": "J Build Eng",
    "Journal of Process Control": "J Process Control",
    "Journal of the Air \\& Waste Management Association": "J Air Waste Manag Assoc",
    "Journal of the IEST": "J IEST",
    "Optimal Control Applications and Methods": "Optim Control Appl Methods",
    "Science and Technology for the Built Environment": "Sci Technol Built Environ",
    "Annals of Work Exposures and Health": "Ann Work Expo Health",
    "Applied Occupational and Environmental Hygiene": "Appl Occup Environ Hyg",
    "Atmosphere": "Atmosphere",
    "Atmospheric Measurement Techniques": "Atmos Meas Tech",
    "Building Services Engineering Research and Technology": "Build Serv Eng Res Technol",
    "Computer Physics Communications": "Comput Phys Commun",
    "Environment International": "Environ Int",
    "Environmental Science \\& Technology": "Environ Sci Technol",
    "Indoor Environments": "Indoor Environ",
    "International Journal of Environmental Research and Public Health": "Int J Environ Res Public Health",
    "Journal of Aerosol Science": "J Aerosol Sci",
    "Journal of Occupational and Environmental Hygiene": "J Occup Environ Hyg",
    "Journal of the Royal Statistical Society: Series A (Statistics in Society)": "J R Stat Soc Ser A Stat Soc",
    "Mathematics and Computers in Simulation": "Math Comput Simul",
    "Renewable and Sustainable Energy Reviews": "Renew Sustain Energy Rev",
    "Reviews in Environmental Science and Bio/Technology": "Rev Environ Sci Biotechnol",
    "Science of The Total Environment": "Sci Total Environ",
    "Social Indicators Research": "Soc Indic Res",
    "Thorax": "Thorax",
    "SIAM Review": "SIAM Rev",
    "Mathematical Programming": "Math Program",
    "Mathematical Programming Computation": "Math Program Comput",
    "Systems & Control Letters": "Syst Control Lett",
    "Environmental Pollution": "Environ Pollut",
    "Nonlinear Analysis: Hybrid Systems": "Nonlinear Anal Hybrid Syst",
    "Foundations and Trends® in Systems and Control": "Found Trends Syst Control",
}
BS = chr(92)
s = open(os.path.join(ROOT, "refs_els.bib"), encoding="utf-8").read()
s += "\n" + open(os.path.join(ROOT, "refs_isa_extra.bib"), encoding="utf-8").read()
s += "\n" + open(os.path.join(ROOT, "refs_isa_extra2.bib"), encoding="utf-8").read()
s = s.replace("Systems &amp; Control Letters", "Systems & Control Letters")
pat_note = re.compile(r",\s*note = \{" + re.escape(BS) + r"url\{https://doi\.org/[^}]*\}\}")
unknown = set()


def abbr(m):
    name = m.group(2)
    if name in ABBR:
        return m.group(1) + ABBR[name] + m.group(3)
    unknown.add(name)
    return m.group(0)


out = []
seen = set()
for entry in re.split(r"(?=^@)", s, flags=re.M):
    m = re.match(r"@\w+\{([^,]+),", entry.strip())
    if m:
        if m.group(1) in seen:
            continue                       # keep the first definition of a key
        seen.add(m.group(1))
    entry = pat_note.sub("", entry)
    entry = re.sub(r"(journal\s*=\s*\{)([^}]*(?:\}[^,}\n]*)?)(\})", abbr, entry)
    out.append(entry)
txt = "".join(out)
# grey literature: keep the URL, drop the internal "No DOI" annotations
for a in ["No DOI; verified via author/publisher page ", "Standard, no DOI; ", "No DOI; "]:
    txt = txt.replace(a, "")
# handbook chapter without page numbers: cite as a book chapter title (avoids an empty-pages entry)
txt = txt.replace("@incollection{ashrae2020fans,", "@book{ashrae2020fans,").replace(
    "title = {Fans}, booktitle = {2020 ASHRAE Handbook---HVAC Systems and Equipment}, chapter = {21},",
    "title = {2020 {ASHRAE} Handbook---{HVAC} Systems and Equipment, Chapter 21: Fans},")
open(os.path.join(ROOT, "refs_isa.bib"), "w", encoding="utf-8").write(txt)
print("entries:", len(re.findall(r"^@", txt, flags=re.M)), "unmapped journals:", sorted(unknown))
