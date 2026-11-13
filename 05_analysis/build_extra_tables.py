"""Append supporting tables S3–S8 to the existing supporting_tables.docx."""

import csv
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOCX = "outputs/supporting_tables.docx"
ANALYSIS = Path("data/derived")

doc = Document(DOCX)

# Display name normalisation
SYNOPSIS_PRETTY = {
    "AmphibianConservation": "Amphibian",
    "BatConservation": "Bat",
    "BeeConservation": "Bee",
    "BiodiversityofMarineArtificialStructures": "Marine artificial structures",
    "ButterflyandMothConservation": "Butterfly & Moth",
    "ControlofFreshwaterInvasiveSpecies": "Freshwater invasives",
    "FarmlandConservation": "Farmland",
    "ForestConservation": "Forest",
    "GrasslandConservation": "Grassland",
    "ManagementofCaptiveAnimals": "Captive animals",
    "MarineFishConservation": "Marine fish",
    "MarineandFreshwaterMammalConservation": "Marine & freshwater mammal",
    "MediterraneanFarmland": "Mediterranean farmland",
    "NaturalPestControl": "Natural pest control",
    "PeatlandConservation": "Peatland",
    "PrimateConservation": "Primate",
    "ShrublandandHeathlandConservation": "Shrubland & heathland",
    "SoilFertility": "Soil fertility",
    "SustainableAquaculture": "Sustainable aquaculture",
}

FAIL_TAGS = [
    "absent_or_contradicted_fact",
    "overgeneralisation",
    "wrong_quantity",
    "unsupported_comparative_claim",
    "wrong_effect_direction_or_magnitude",
    "mismerged_studies",
    "unsupported_evidence_summary",
]

# ---------- helpers ----------
def shade_cell(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tc_pr.append(shd)

def format_header_row(row):
    for cell in row.cells:
        shade_cell(cell, 'D9E2F3')
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True

def add_caption(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    p.space_after = Pt(6)

def add_footnote(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)

def right_align(row, cols):
    for j in cols:
        for p in row.cells[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

def shrink_table_font(table, size=9):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)

def set_col_widths(table, widths_inches):
    """Force per-column widths and disable autofit; widths_inches is a list of floats."""
    table.autofit = False
    tblPr = table._element.tblPr
    # Force fixed layout
    for existing in tblPr.findall(qn('w:tblLayout')):
        tblPr.remove(existing)
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblPr.append(layout)
    # Set total table width
    total_w = sum(widths_inches)
    for existing in tblPr.findall(qn('w:tblW')):
        tblPr.remove(existing)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(int(total_w * 1440)))
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    # Add a tblGrid that defines column widths (LibreOffice requires this)
    tbl = table._element
    for existing in tbl.findall(qn('w:tblGrid')):
        tbl.remove(existing)
    grid = OxmlElement('w:tblGrid')
    for w in widths_inches:
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), str(int(w * 1440)))
        grid.append(gc)
    # Insert tblGrid right after tblPr
    tblPr_elem = tbl.find(qn('w:tblPr'))
    tblPr_elem.addnext(grid)
    # Set per-cell widths
    for row in table.rows:
        for j, w in enumerate(widths_inches):
            if j < len(row.cells):
                row.cells[j].width = Inches(w)
                tcPr = row.cells[j]._tc.get_or_add_tcPr()
                for ex in tcPr.findall(qn('w:tcW')):
                    tcPr.remove(ex)
                tcW = OxmlElement('w:tcW')
                tcW.set(qn('w:w'), str(int(w * 1440)))
                tcW.set(qn('w:type'), 'dxa')
                tcPr.append(tcW)

def read_csv(path):
    with open(path, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f))

# Locate the Appendix A page-break to insert before it
appendix_para = None
for i, p in enumerate(doc.paragraphs):
    if "Appendix A" in p.text and p.style.name.startswith("Heading"):
        appendix_para = p
        break

# Standard 'document end' approach: we'll just append S3–S8 at the end,
# then move the Appendix A heading + everything after to after the new tables.
# Simpler: append everything before the appendix by re-saving order.
# python-docx doesn't support easy re-ordering, so we will instead REBUILD
# the appendix at the end. Easier path: we append tables now, then re-add the
# appendix block by reading prompts again.
# Actually simplest: we add S3–S8 right at the end of the body (Appendix A is
# already there). The Appendix A page break already exists before it, so we
# need to insert *before* the page break. We'll use the low-level API.

# Find the page-break paragraph immediately before the Appendix heading
def find_page_break_before(target_para):
    body = target_para._element.getparent()
    target = target_para._element
    # Walk backwards from target to find paragraph containing a w:br type="page"
    prev = target.getprevious()
    while prev is not None:
        if prev.tag.endswith('}p'):
            brs = prev.findall(".//" + qn('w:br'))
            for br in brs:
                if br.get(qn('w:type')) == 'page':
                    return prev
        prev = prev.getprevious()
    return None

page_break_para = find_page_break_before(appendix_para) if appendix_para is not None else None
insert_anchor = page_break_para if page_break_para is not None else appendix_para

# We need a way to add new content BEFORE insert_anchor.
# Strategy: collect new elements by adding them at end of document via doc.add_*,
# then move them before insert_anchor.
body = insert_anchor.getparent()
new_elements_start_idx = len(list(body)) - 1  # remember snapshot

# ---------- Build S3–S8 at end of doc, then relocate ----------

# =============================================================
# Table S3 — Per-model headline metrics with 95% CIs
# =============================================================
add_caption("Table S3 — Per-model headline metrics with 95% bootstrap confidence intervals. Each row reports the mean across summaries (n = number of summaries scored for that model–metric cell). Relevance, citation rate, CC-when-cited and faithfulness are bounded in [0, 1]; values are reported to four decimal places.")

mh = read_csv(ANALYSIS / "model_headline.csv")
# pivot: model × metric
models = list(dict.fromkeys(r["model"] for r in mh))
metrics_order = ["Relevance", "Citation rate", "CC (all stmts)", "CC (when cited)", "Faithfulness"]

t3 = doc.add_table(rows=1 + len(models), cols=1 + len(metrics_order))
t3.style = 'Light Grid Accent 1'
hdr = t3.rows[0].cells
hdr[0].text = "Model"
for j, m in enumerate(metrics_order, 1):
    hdr[j].text = f"{m}\nmean [95% CI]  (n)"
format_header_row(t3.rows[0])

idx = {(r["model"], r["metric"]): r for r in mh}
for i, model in enumerate(models, 1):
    row = t3.rows[i].cells
    row[0].text = model
    for j, m in enumerate(metrics_order, 1):
        rec = idx.get((model, m))
        if rec is None:
            row[j].text = "—"
        else:
            row[j].text = f"{float(rec['mean']):.3f} [{float(rec['lo']):.3f}, {float(rec['hi']):.3f}]  (n={rec['n_summaries']})"
    right_align(t3.rows[i], range(1, 1 + len(metrics_order)))
shrink_table_font(t3, 9)
set_col_widths(t3, [0.95, 1.15, 1.15, 1.15, 1.15, 1.15])

doc.add_paragraph()

# =============================================================
# Table S4 — Per-model summary-distribution detail
# =============================================================
add_caption("Table S4 — Per-model distribution of summary-level scores. mean, median, SD, IQR and coefficient of variation are computed across summaries (n per model–metric cell). %≥90, %≥80 are right-tail fractions; %<70, %<50 are left-tail fractions. Composite metric available from Figure 2 source data.")

sd = read_csv(ANALYSIS / "summary_distribution.csv")
metrics_sd = ["Relevance", "Citation rate", "CC (all stmts)", "CC (when cited)", "Faithfulness"]
models_sd = list(dict.fromkeys(r["model"] for r in sd))

t4 = doc.add_table(rows=1 + len(models_sd) * len(metrics_sd), cols=10)
t4.style = 'Light Grid Accent 1'
hdr = t4.rows[0].cells
for j, h in enumerate(["Model", "Metric", "n", "Mean", "Median", "SD", "IQR", "%≥90", "%<70", "%<50"]):
    hdr[j].text = h
format_header_row(t4.rows[0])
set_col_widths(t4, [0.85, 1.0, 0.4, 0.5, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5])

idx = {(r["model"], r["metric"]): r for r in sd}
i = 1
for model in models_sd:
    for k, m in enumerate(metrics_sd):
        rec = idx.get((model, m))
        if rec is None:
            continue
        row = t4.rows[i].cells
        row[0].text = model if k == 0 else ""
        row[1].text = m
        row[2].text = rec["n"]
        row[3].text = f"{float(rec['mean']):.3f}"
        row[4].text = f"{float(rec['median']):.3f}"
        row[5].text = f"{float(rec['sd']):.3f}"
        row[6].text = f"{float(rec['iqr']):.3f}"
        row[7].text = f"{float(rec['pct_ge_90'])*100:.1f}%"
        row[8].text = f"{float(rec['pct_lt_70'])*100:.1f}%"
        row[9].text = f"{float(rec['pct_lt_50'])*100:.1f}%"
        right_align(t4.rows[i], range(2, 10))
        i += 1
shrink_table_font(t4, 8)

doc.add_paragraph()

# =============================================================
# Table S5 — Variance decomposition
# =============================================================
add_caption("Table S5 — Variance decomposition of summary-level scores using two-way ANOVA with model, synopsis, and their interaction as fixed effects. SS = type-II sum of squares; η² = SS divided by total SS (proportion of variance attributable to each term). Residual = within-cell variance across summaries. Rows sum to 1 within each metric.")

vd = read_csv(ANALYSIS / "variance_decomp.csv")
metrics_vd = list(dict.fromkeys(r["metric"] for r in vd))
terms_vd = ["model", "synopsis", "model:synopsis", "Residuals"]

t5 = doc.add_table(rows=1 + len(metrics_vd) * len(terms_vd), cols=4)
t5.style = 'Light Grid Accent 1'
hdr = t5.rows[0].cells
for j, h in enumerate(["Metric", "Term", "SS", "η²"]):
    hdr[j].text = h
format_header_row(t5.rows[0])

idx = {(r["metric"], r["term"]): r for r in vd}
i = 1
for m in metrics_vd:
    for k, term in enumerate(terms_vd):
        rec = idx.get((m, term))
        if rec is None:
            continue
        row = t5.rows[i].cells
        row[0].text = m if k == 0 else ""
        row[1].text = term
        row[2].text = f"{float(rec['ss']):.3f}"
        row[3].text = f"{float(rec['eta_sq']):.4f}"
        right_align(t5.rows[i], (2, 3))
        i += 1
shrink_table_font(t5, 9)
set_col_widths(t5, [1.2, 1.6, 1.4, 1.4])

doc.add_paragraph()

# =============================================================
# Table S6 — Failure-mode prevalence by model
# =============================================================
add_caption("Table S6 — Prevalence of each failure-tag category by summary-author model, expressed as the number of statements assigned that tag per 100 statements scored. n_stmts = total statements scored for that model (denominator). A single statement may receive multiple tags. Tag definitions are given in Table S1.")

fp = read_csv(ANALYSIS / "failure_reason_profile.csv")
models_fp = list(dict.fromkeys(r["model"] for r in fp))

t6 = doc.add_table(rows=1 + len(FAIL_TAGS), cols=1 + len(models_fp))
t6.style = 'Light Grid Accent 1'
hdr = t6.rows[0].cells
hdr[0].text = "Failure tag"
for j, mdl in enumerate(models_fp, 1):
    # find n_stmts_scored for this model (constant across its tags)
    n_stmts = next((r["n_stmts_scored"] for r in fp if r["model"] == mdl), "—")
    hdr[j].text = f"{mdl}\n(n={n_stmts})"
format_header_row(t6.rows[0])
set_col_widths(t6, [2.9] + [0.75] * len(models_fp))

idx = {(r["model"], r["failure_reason"]): r for r in fp}
for i, tag in enumerate(FAIL_TAGS, 1):
    row = t6.rows[i].cells
    # render tag in Courier
    row[0].text = ""
    p_tag = row[0].paragraphs[0]
    run = p_tag.add_run(tag)
    rpr = run._element.get_or_add_rPr()
    rf = OxmlElement('w:rFonts')
    rf.set(qn('w:ascii'), 'Courier New')
    rf.set(qn('w:hAnsi'), 'Courier New')
    rf.set(qn('w:cs'), 'Courier New')
    rpr.append(rf)
    run.font.size = Pt(8)
    for j, mdl in enumerate(models_fp, 1):
        rec = idx.get((mdl, tag))
        if rec is None:
            row[j].text = "—"
        else:
            rate = float(rec["rate_per_stmt"]) * 100
            row[j].text = f"{rate:.2f}"
        right_align(t6.rows[i], (j,))
shrink_table_font(t6, 9)

doc.add_paragraph()

# =============================================================
# Table S7 — Extended LLM-judge × human agreement with
# per-summariser signed bias (S7a Relevance, S7b CC-cited, S7c Faithfulness)
# =============================================================
add_caption("Table S7 — Extended LLM-as-judge validation against consensus human labels, broken out by summariser model. Three candidate judges (Gemini 3.1 Pro Preview, Claude Opus 4.1, GPT-5.4) were evaluated under four prompt regimes (refined manual; GEPA seed 1234; GEPA seed 5678; seed v7) on the held-out human-evaluated subset. Agreement = % of statements matching consensus human label; |bias| = absolute mean Yes-rate gap (judge − human) averaged across summariser models; κ = Cohen's kappa. The signed per-summariser columns report the judge − human Yes-rate gap (percentage points) for each summary author; positive values mean the judge is more lenient than humans, negative values mean stricter. A judge with small |bias| but a wide signed spread across summarisers is unsuitable because it favours/disfavours specific authors. The production judge (bold) is the refined-manual configuration of Gemini 3.1 Pro Preview; CC-all is omitted because its agreement is largely mechanical (uncited statements auto-failed by both judge and humans) and was not part of judge selection.")

# Per-summariser signed bias (pp) from analysis_outputs/judge_per_summariser_bias.csv
# Row order: (Judge, Prompt, [overall_agree, overall_|bias|, κ], [Claude, Gemini, GPT-5, Kimi signed bias pp], is_prod)
_S7_RELEVANCE = [
    ("Gemini 3.1 Pro Preview", "Refined manual",      "94.1%", "2.7%",  "0.400", "+2.7",  "+2.5",  "+3.1",  "+2.4",  True),
    ("Gemini 3.1 Pro Preview", "GEPA seed 1234",      "84.4%", "13.4%", "0.343", "-16.8", "-16.2", "-10.5", "-10.0", False),
    ("Gemini 3.1 Pro Preview", "GEPA seed 5678",      "85.6%", "11.6%", "0.354", "-15.0", "-11.1", "-8.4",  "-11.8", False),
    ("Gemini 3.1 Pro Preview", "Seed v7 (original)",  "92.3%", "1.6%",  "0.370", "-1.4",  "-2.0",  "+2.1",  "+0.9",  False),
    ("Claude Opus 4.1",        "Refined manual",      "93.8%", "2.3%",  "0.383", "+2.3",  "+1.0",  "+2.6",  "+3.3",  False),
    ("Claude Opus 4.1",        "GEPA seed 1234",      "88.1%", "8.6%",  "0.397", "-11.4", "-11.6", "-5.3",  "-6.2",  False),
    ("Claude Opus 4.1",        "GEPA seed 5678",      "86.7%", "9.9%",  "0.357", "-10.5", "-12.6", "-8.9",  "-7.6",  False),
    ("Claude Opus 4.1",        "Seed v7 (original)",  "91.9%", "1.7%",  "0.334", "-2.8",  "-1.0",  "+2.1",  "+1.0",  False),
    ("GPT-5.4",                "Refined manual",      "93.8%", "2.3%",  "0.383", "+1.8",  "+2.0",  "+3.7",  "+1.9",  False),
    ("GPT-5.4",                "GEPA seed 1234",      "82.2%", "15.8%", "0.312", "-16.8", "-20.2", "-13.1", "-13.3", False),
    ("GPT-5.4",                "GEPA seed 5678",      "81.0%", "16.5%", "0.283", "-21.4", "-19.7", "-14.1", "-10.9", False),
    ("GPT-5.4",                "Seed v7 (original)",  "93.3%", "2.4%",  "0.334", "+0.9",  "+1.0",  "+4.7",  "+2.8",  False),
]
_S7_CC_CITED = [
    ("Gemini 3.1 Pro Preview", "Refined manual",      "94.3%", "2.6%", "0.377", "-3.3",  "+1.1",  "+1.1",  "-4.9",  True),
    ("Gemini 3.1 Pro Preview", "GEPA seed 1234",      "92.4%", "5.1%", "0.412", "-13.3", "-1.1",  "-1.1",  "-4.9",  False),
    ("Gemini 3.1 Pro Preview", "GEPA seed 5678",      "91.7%", "4.8%", "0.347", "-11.7", "-0.5",  "-2.1",  "-4.9",  False),
    ("Gemini 3.1 Pro Preview", "Seed v7 (original)",  "93.0%", "2.6%", "0.282", "-5.6",  "+0.5",  "+1.1",  "-3.3",  False),
    ("Claude Opus 4.1",        "Refined manual",      "94.0%", "1.1%", "0.271", "-1.1",  "-0.5",  "+1.6",  "+1.6",  False),
    ("Claude Opus 4.1",        "GEPA seed 1234",      "93.3%", "3.2%", "0.421", "-8.4",  "-1.6",  "-1.1",  "-1.6",  False),
    ("Claude Opus 4.1",        "GEPA seed 5678",      "93.3%", "2.1%", "0.354", "-6.1",  "+0.5",  "-1.6",  "+0.0",  False),
    ("Claude Opus 4.1",        "Seed v7 (original)",  "93.5%", "1.1%", "0.276", "-2.2",  "+0.0",  "+0.5",  "-1.6",  False),
    ("GPT-5.4",                "Refined manual",      "93.8%", "1.1%", "0.264", "-1.7",  "-0.5",  "+2.1",  "+0.0",  False),
    ("GPT-5.4",                "GEPA seed 1234",      "93.5%", "2.8%", "0.396", "-8.9",  "+0.0",  "+0.5",  "-1.6",  False),
    ("GPT-5.4",                "GEPA seed 5678",      "90.0%", "5.2%", "0.233", "-13.3", "-2.1",  "-0.5",  "-4.9",  False),
    ("GPT-5.4",                "Seed v7 (original)",  "93.4%", "1.5%", "0.246", "-3.9",  "+0.0",  "+2.1",  "+0.0",  False),
]
_S7_FAITH = [
    ("Gemini 3.1 Pro Preview", "Refined manual",      "83.5%", "4.3%",  "0.400", "-4.5",  "+5.1",  "+4.2",  "-3.3",  True),
    ("Gemini 3.1 Pro Preview", "GEPA seed 1234",      "82.0%", "5.8%",  "0.355", "-5.5",  "+6.1",  "+4.7",  "-7.1",  False),
    ("Gemini 3.1 Pro Preview", "GEPA seed 5678",      "81.0%", "4.4%",  "0.379", "-8.6",  "+0.0",  "-1.0",  "-8.1",  False),
    ("Gemini 3.1 Pro Preview", "Seed v7 (original)",  "84.4%", "5.7%",  "0.343", "+4.1",  "+6.6",  "+7.3",  "+4.7",  False),
    ("Claude Opus 4.1",        "Refined manual",      "83.6%", "5.1%",  "0.426", "-3.7",  "+5.6",  "+1.6",  "-9.5",  False),
    ("Claude Opus 4.1",        "GEPA seed 1234",      "84.2%", "4.4%",  "0.366", "+3.2",  "+6.6",  "+7.3",  "-0.5",  False),
    ("Claude Opus 4.1",        "GEPA seed 5678",      "82.9%", "3.7%",  "0.360", "+0.5",  "+5.1",  "+4.7",  "-4.7",  False),
    ("Claude Opus 4.1",        "Seed v7 (original)",  "84.6%", "8.8%",  "0.289", "+9.6",  "+7.1",  "+10.5", "+8.1",  False),
    ("GPT-5.4",                "Refined manual",      "72.7%", "15.3%", "0.281", "-20.5", "-4.0",  "-12.0", "-24.6", False),
    ("GPT-5.4",                "GEPA seed 1234",      "81.5%", "3.1%",  "0.342", "-4.1",  "+2.0",  "+2.6",  "-3.8",  False),
    ("GPT-5.4",                "GEPA seed 5678",      "81.1%", "3.7%",  "0.346", "-5.9",  "+1.5",  "+2.1",  "-5.2",  False),
    ("GPT-5.4",                "Seed v7 (original)",  "82.9%", "4.5%",  "0.305", "+0.0",  "+5.1",  "+7.3",  "+5.7",  False),
]

def _build_s7_subtable(label, metric_name, rows):
    sub = doc.add_paragraph()
    sub_run = sub.add_run(f"{label}  •  {metric_name}")
    sub_run.bold = True
    sub_run.font.size = Pt(10)

    # Cols: Judge, Prompt, Agree, |bias|, κ, Claude, Gemini, GPT-5, Kimi  -> 9 cols
    tt = doc.add_table(rows=2 + len(rows), cols=9)
    tt.style = 'Light Grid Accent 1'
    h1 = tt.rows[0].cells
    h2 = tt.rows[1].cells
    h1[0].text = "Judge"
    h1[1].text = "Prompt"
    h1[2].text = "Overall"
    h1[3].text = ""
    h1[4].text = ""
    h1[5].text = "Signed bias by summariser (pp)"
    h1[6].text = ""
    h1[7].text = ""
    h1[8].text = ""
    # Merge group-label cells across their sub-columns
    h1[2].merge(h1[3]).merge(h1[4])
    h1[5].merge(h1[6]).merge(h1[7]).merge(h1[8])
    # Merge Judge / Prompt vertically across both header rows
    h1[0].merge(h2[0]).text = "Judge"
    h1[1].merge(h2[1]).text = "Prompt"
    # Sub-headers in row 2 (cells 2-8 still distinct in row 1 logically; reaccess by index)
    h2 = tt.rows[1].cells  # refresh after merges
    h2[2].text = "Agree"
    h2[3].text = "|bias|"
    h2[4].text = "κ"
    h2[5].text = "Claude"
    h2[6].text = "Gemini"
    h2[7].text = "GPT-5"
    h2[8].text = "Kimi"
    # Centre-align the merged group labels
    for idx in (2, 5):
        for p in tt.rows[0].cells[idx].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_header_row(tt.rows[0])
    format_header_row(tt.rows[1])

    prev_judge = None
    for ri, r in enumerate(rows, start=2):
        judge_name, prompt, agree, abias, kappa, c, g, gp, k, is_prod = r
        cells = tt.rows[ri].cells
        cells[0].text = judge_name if judge_name != prev_judge else ""
        cells[1].text = prompt
        cells[2].text = agree
        cells[3].text = abias
        cells[4].text = kappa
        cells[5].text = c
        cells[6].text = g
        cells[7].text = gp
        cells[8].text = k
        if is_prod:
            for c_ in cells:
                for p in c_.paragraphs:
                    for run in p.runs:
                        run.bold = True
        right_align(tt.rows[ri], range(2, 9))
        prev_judge = judge_name
    shrink_table_font(tt, 8)
    # Total width ~ 9.0" works in landscape; in portrait it will compress
    set_col_widths(tt, [1.05, 1.25, 0.55, 0.55, 0.50, 0.55, 0.55, 0.55, 0.55])
    doc.add_paragraph()

_build_s7_subtable("S7a", "Relevance", _S7_RELEVANCE)
_build_s7_subtable("S7b", "CC (when cited)", _S7_CC_CITED)
_build_s7_subtable("S7c", "Faithfulness", _S7_FAITH)

add_footnote("Production judge in bold (refined-manual Gemini 3.1 Pro Preview). 'Overall' columns repeat the values shown in main-text Table 1. 'Signed bias by summariser' columns report (judge Yes-rate − human Yes-rate) in percentage points, computed within each summariser model on the human-evaluated subset; positive = judge is more lenient than humans. Note GPT-5.4 refined-manual faithfulness shows a 20.6 pp spread across summarisers (Claude −20.5, Kimi −24.6 vs Gemini −4.0), confirming it would systematically disfavour some summarisers if used as judge. CC-all is omitted because uncited statements are auto-failed by construction and judge selection used CC-cited only.")

doc.add_paragraph()

# =============================================================
# Table S8 — Synopsis × model performance
# =============================================================
add_caption("Table S8 — Mean summary-level score by synopsis × model for each metric. Values are mean fractions in [0, 1] across the question-level summaries within each synopsis–model cell; 19 synopses × 4 models per metric. Cells with no eligible summaries are shown as '—'. This table is the numeric companion to Figure 3.")

sx = read_csv(ANALYSIS / "synopsis_x_model.csv")
synopses = list(dict.fromkeys(r["synopsis"] for r in sx))
models_sx = list(dict.fromkeys(r["model"] for r in sx))
metrics_sx = ["Relevance", "Citation rate", "CC (all stmts)", "CC (when cited)", "Faithfulness"]

# Build one sub-table per metric (otherwise too wide)
idx = {(r["metric"], r["synopsis"], r["model"]): r for r in sx}

for m in metrics_sx:
    sub = doc.add_paragraph()
    sub_run = sub.add_run(f"S8{['a','b','c','d','e'][metrics_sx.index(m)]}  •  {m}")
    sub_run.bold = True
    sub_run.font.size = Pt(10)

    tt = doc.add_table(rows=1 + len(synopses), cols=1 + len(models_sx))
    tt.style = 'Light Grid Accent 1'
    hdr = tt.rows[0].cells
    hdr[0].text = "Synopsis"
    for j, mdl in enumerate(models_sx, 1):
        hdr[j].text = mdl
    format_header_row(tt.rows[0])
    for i, syn in enumerate(synopses, 1):
        row = tt.rows[i].cells
        row[0].text = SYNOPSIS_PRETTY.get(syn, syn)
        for j, mdl in enumerate(models_sx, 1):
            rec = idx.get((m, syn, mdl))
            if rec is None:
                row[j].text = "—"
            else:
                row[j].text = f"{float(rec['mean_score']):.3f}"
        right_align(tt.rows[i], range(1, 1 + len(models_sx)))
    set_col_widths(tt, [1.8] + [1.05] * len(models_sx))
    shrink_table_font(tt, 9)
    doc.add_paragraph()

# =============================================================
# Move the just-added elements before the Appendix A page break
# =============================================================
all_body = list(body)
# Everything added is at the end. Find the slice from new_elements_start_idx+1 onward
# except the final paragraph that might come after the appendix (none in our case).
appendix_idx = all_body.index(insert_anchor)
new_block = all_body[new_elements_start_idx + 1:]
# Filter out items that are actually before the appendix anchor (shouldn't be any, but be safe)
for el in new_block:
    body.remove(el)
# Re-insert immediately before insert_anchor
for el in new_block:
    insert_anchor.addprevious(el)

doc.save(DOCX)
print(f"Wrote {DOCX}")
