"""Build supporting tables S1 (failure-tag definitions), S2 (inter-rater κ),
plus Appendix A (judge prompts) as a single .docx file."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

doc = Document()

# --- Set base style ---
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

def add_caption(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    p.space_after = Pt(6)

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

# =============================================================
# Table S1 — Failure-tag definitions
# =============================================================
add_caption("Table S1 — Definitions of the seven failure-tag categories returned by the LLM-as-judge for statements assessed as not fully faithful. Tags are drawn from a controlled vocabulary; multiple tags may be assigned to a single statement when more than one failure mode applies.")

tags = [
    ("absent_or_contradicted_fact",
     "The statement makes a specific factual claim (about effects, sample sizes, locations, species, contexts, study design, etc.) that is absent from or directly contradicted by the cited evidence documents."),
    ("overgeneralisation",
     "The statement extends the cited evidence beyond its stated scope — for example, attributing a finding to 'most', 'all', 'typically', or 'generally' when the evidence is mixed, limited, or only from a small number of studies."),
    ("wrong_quantity",
     "The statement reports a specific number, count, or quantifier that is wrong or not clearly supported by the cited evidence (e.g. says 'several studies' when only one or two exist)."),
    ("unsupported_comparative_claim",
     "The statement makes a comparative claim (e.g. 'X is more effective than Y', 'no difference from Z') that is not explicitly supported by at least one cited document."),
    ("wrong_effect_direction_or_magnitude",
     "The statement misrepresents the direction, size, or certainty of an effect — for example, asserting 'increased' when a document reports 'no significant effect', or 'strong evidence' when the document indicates very limited or mixed evidence."),
    ("mismerged_studies",
     "The statement combines findings from different studies into a single claim that does not match what any individual study actually found."),
    ("unsupported_evidence_summary",
     "The statement makes a claim about the overall state of the evidence base (e.g. 'no studies', 'only one study', 'many studies') that is not clearly supported by the number and type of studies in the cited documents."),
]

t1 = doc.add_table(rows=1+len(tags), cols=2)
t1.style = 'Light Grid Accent 1'
t1.autofit = False
t1.columns[0].width = Inches(2.6)
t1.columns[1].width = Inches(3.9)
# Set per-cell widths to enforce column ratio in Word
for row in t1.rows:
    row.cells[0].width = Inches(2.6)
    row.cells[1].width = Inches(3.9)

hdr = t1.rows[0].cells
hdr[0].text = "Failure tag"
hdr[1].text = "Definition"
format_header_row(t1.rows[0])

for i, (tag, defn) in enumerate(tags, 1):
    row = t1.rows[i].cells
    # Render tag with Consolas explicitly via rPr
    row[0].text = ""
    p_tag = row[0].paragraphs[0]
    run = p_tag.add_run(tag)
    rpr = run._element.get_or_add_rPr()
    rfonts = OxmlElement('w:rFonts')
    rfonts.set(qn('w:ascii'), 'Courier New')
    rfonts.set(qn('w:hAnsi'), 'Courier New')
    rfonts.set(qn('w:cs'), 'Courier New')
    rpr.append(rfonts)
    run.font.size = Pt(9)
    run.bold = False
    row[1].text = defn

doc.add_paragraph()

# =============================================================
# Table S2 — Inter-rater reliability
# =============================================================
add_caption("Table S2 — Inter-rater reliability between the three human reviewers (R1, R2, R3) who scored the LLM-judge validation subset. By design, 19 statements were dual-scored within each reviewer pair, with no statement scored by all three reviewers (so Fleiss' κ across the three reviewers is not estimable). For relevance R1 × R2, all 19 statements were labelled Yes by both reviewers, so Cohen's κ is undefined (zero variance in both rater margins). Cohen's κ can also appear pathologically low on small subsets when one label class dominates (e.g. relevance R2 × R3: 94.7% raw agreement, κ = 0.000 because nearly all statements were labelled Yes).")

irr_rows = [
    # metric, comparison, n, raw agree %, kappa
    ("Relevance",           "R1 × R2",   "19", "100.0%", "undefined†"),
    ("Relevance",           "R1 × R3",   "19", "84.2%",  "−0.075"),
    ("Relevance",           "R2 × R3",   "19", "94.7%",  "0.000"),
    ("CC (cited only)",     "R1 × R2",   "19", "100.0%", "1.000"),
    ("CC (cited only)",     "R1 × R3",   "19", "94.7%",  "0.855"),
    ("CC (cited only)",     "R2 × R3",   "19", "100.0%", "1.000"),
    ("Faithfulness",        "R1 × R2",   "19", "89.5%",  "0.457"),
    ("Faithfulness",        "R1 × R3",   "19", "89.5%",  "0.457"),
    ("Faithfulness",        "R2 × R3",   "19", "84.2%",  "0.000"),
]

t2 = doc.add_table(rows=1+len(irr_rows), cols=5)
t2.style = 'Light Grid Accent 1'

hdr = t2.rows[0].cells
hdr[0].text = "Metric"
hdr[1].text = "Comparison"
hdr[2].text = "n"
hdr[3].text = "Raw agreement"
hdr[4].text = "κ"
format_header_row(t2.rows[0])

prev_metric = None
for i, row_data in enumerate(irr_rows, 1):
    metric, comp, n, agree, kappa = row_data
    row = t2.rows[i].cells
    # blank repeating metric label for readability
    row[0].text = metric if metric != prev_metric else ""
    row[1].text = comp
    row[2].text = n
    row[3].text = agree
    row[4].text = kappa
    # right-align numeric columns
    for j in (2, 3, 4):
        for p in row[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    prev_metric = metric

# Footnote for Table S2
footnote = doc.add_paragraph()
fn_run = footnote.add_run(
    "† All 19 statements were labelled Yes by both reviewers; Cohen's κ is undefined when one rater has zero variance."
)
fn_run.italic = True
fn_run.font.size = Pt(9)

doc.add_paragraph()

# =============================================================
# Appendix A — Production judge prompts
# =============================================================
doc.add_page_break()
h = doc.add_heading("Appendix A — Production LLM-as-judge prompts", level=1)

intro = doc.add_paragraph(
    "The three prompts below were used by the production LLM judge "
    "(google/gemini-3.1-pro-preview, temperature = 0.0) to score every statement "
    "in the main evaluation. Variable placeholders are shown in curly braces "
    "({statement}, {question}, etc.) and were substituted at runtime. "
    "Each prompt instructs the judge to return a JSON object with explicit "
    "reasoning and a binary verdict (Yes / No), enforced via a JSON-schema-"
    "constrained response format."
)
for run in intro.runs:
    run.italic = True

def add_prompt_block(label, source_file, body):
    h2 = doc.add_heading(label, level=2)
    meta = doc.add_paragraph()
    meta.add_run("Source: ").bold = True
    code = meta.add_run(source_file)
    code_rpr = code._element.get_or_add_rPr()
    code_rfonts = OxmlElement('w:rFonts')
    code_rfonts.set(qn('w:ascii'), 'Courier New')
    code_rfonts.set(qn('w:hAnsi'), 'Courier New')
    code_rfonts.set(qn('w:cs'), 'Courier New')
    code_rpr.append(code_rfonts)
    code.font.size = Pt(10)

    # body in monospace, smaller font, single-paragraph blocks separated by blank lines
    for block in body.strip().split("\n\n"):
        p = doc.add_paragraph()
        run = p.add_run(block)
        # Force Consolas through rPr only (run.font.name creates a duplicate rFonts element)
        rpr = run._element.get_or_add_rPr()
        rfonts = OxmlElement('w:rFonts')
        rfonts.set(qn('w:ascii'), 'Courier New')
        rfonts.set(qn('w:hAnsi'), 'Courier New')
        rfonts.set(qn('w:cs'), 'Courier New')
        rpr.append(rfonts)
        run.font.size = Pt(9)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(6)

# Read prompt files
prompt_dir = Path("03_llm_judge/prompt_files")  # run from the repository root
relevance = (prompt_dir / "relevance/manual_relevance_v16.txt").read_text()
cc = (prompt_dir / "citation_correctness/manual_citation_correctness_v17.txt").read_text()
faithfulness = (prompt_dir / "faithfulness/manual_faithfulness_v14-concat.txt").read_text()

add_prompt_block("A.1 Relevance prompt (manual_relevance_v16)",
                 "prompt_files/relevance/manual_relevance_v16.txt",
                 relevance)
add_prompt_block("A.2 Citation correctness prompt (manual_citation_correctness_v17)",
                 "prompt_files/citation_correctness/manual_citation_correctness_v17.txt",
                 cc)
add_prompt_block("A.3 Faithfulness prompt (manual_faithfulness_v14-concat)",
                 "prompt_files/faithfulness/manual_faithfulness_v14-concat.txt",
                 faithfulness)

out_path = "outputs/supporting_tables.docx"
doc.save(out_path)
print(f"Wrote {out_path}")
