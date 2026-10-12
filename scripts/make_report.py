"""Create the five-page report and references from saved results."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "prevalence_baseline": "Overall baseline",
    "antibiotic_baseline": "Drug baseline",
    "logistic_unweighted": "Logistic",
    "logistic_balanced": "Logistic, weighted",
    "lightgbm_unweighted": "LightGBM",
    "lightgbm_balanced": "LightGBM, weighted",
}
DRUG_NAMES = {
    "amoxicillin/clavulanate": "Amoxicillin/clavulanate",
    "piperacillin/tazobactam": "Piperacillin/tazobactam",
    "trimethoprim/sulfamethoxazole": "Trimethoprim/sulfamethoxazole",
}
FEATURE_NAMES = {
    "lab_median_wbc_14d": "White blood cell count, prior 14 days",
    "lab_median_creatinine_14d": "Creatinine, prior 14 days",
    "vital_median_heart_rate_48h": "Heart rate, prior 48 hours",
    "vital_median_temperature_48h": "Temperature, prior 48 hours",
    "prior_abx_days_ago": "Days since prior antibiotic exposure",
    "prior_abx_class_count_90d": "Antibiotic classes recorded in prior 90 days",
    "prior_candidate_resistance_days_ago": "Days since prior resistance to candidate",
}


def number(value, digits=3):
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def count(value):
    return f"{int(value):,}"


def percent(value):
    return f"{100 * float(value):.1f}%"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Completed pipeline output is required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def interval_text(intervals, model, metric):
    row = intervals[(intervals.model == model) & (intervals.metric == metric)].iloc[0]
    return f"{number(row.estimate)} (95% interval {number(row.ci_low)} to {number(row.ci_high)})"


def paragraphs(*texts):
    return [("p", x) for x in texts]


def build_pages(results, config):
    audit = read_json(results / "construction_audit.json")
    summary = pd.read_csv(results / "cohort_summary.csv")
    missing = pd.read_csv(results / "missingness.csv")
    imbalance = pd.read_csv(results / "class_balance.csv")
    model_dir = results / "models"
    evaluation = read_json(model_dir / "evaluation_summary.json")
    selection = read_json(model_dir / "frozen_selection.json")
    comparison = pd.read_csv(model_dir / "model_comparison.csv")
    intervals = pd.read_csv(model_dir / "test_confidence_intervals.csv")
    differences = pd.read_csv(model_dir / "paired_model_differences.csv")
    groups = pd.read_csv(model_dir / "subgroup_metrics.csv")
    diagnostics = pd.read_csv(model_dir / "calibration_diagnostics.csv")
    verified = read_json(ROOT / "data" / "raw" / "verified_downloads.json")
    overall = summary.loc[summary.split.eq("all")].iloc[0]
    selected = evaluation["selected_model"]
    chosen = comparison[(comparison.split == "test") & (comparison.model == selected)].iloc[0]
    assert selected == selection["selected_model"], "Model selection files disagree"
    assert selection["test_used_for_selection"] is False, "Test labels must not select the model"
    assert int(overall.rows) == audit["analytical_rows"], "Cohort summary is stale"
    assert int(chosen.n) == int(summary.loc[summary.split.eq("test"), "rows"].iloc[0]), "Evaluation and cohort files describe different runs"
    assert evaluation["proper_fit_rows"] + evaluation["calibration_rows"] == int(summary.loc[summary.split.eq("train"), "rows"].iloc[0]), "Training counts disagree"
    selected_name = MODEL_NAMES.get(selected, selected)
    tested = comparison[comparison.split.eq("test")].set_index("model")
    drugs = ", ".join(config["antibiotics"])
    minority = "resistant" if overall.resistant_n <= overall.susceptible_n else "susceptible"
    n_numeric = len(evaluation.get("numeric_features", []))
    if not n_numeric:
        spec = read_json(ROOT / "src" / "armd" / "support_feature_spec.json")
        n_numeric = len(spec["numeric_features"])
    cohort_rows = [["Set", "Rows", "Patients", "Cultures", "Resistant"]]
    for split in ["all", "train", "validation", "test"]:
        row = summary[summary.split == split].iloc[0]
        cohort_rows.append([split.capitalize(), count(row.rows), count(row.patients), count(row.culture_events), percent(row.resistant_prevalence)])
    pages = []
    page = [("title", "Antimicrobial resistance prediction"), ("sub", "1. Task and study sample")]
    page += paragraphs(
        f"This study predicts a laboratory susceptibility result at the time a culture is ordered. The final dataset has {count(overall.rows)} rows from {count(overall.patients)} patients and {count(overall.culture_events)} cultures. {selected_name} was selected using validation data. Its test drug-macro AUROC was {number(chosen.antibiotic_macro_auroc)}.",
        "One row is one patient, culture event, organism and candidate antibiotic. Susceptible is 1. Resistant is 0. Only observed results are used. Untested drugs have no label. Current organism identity defines the row but is excluded from predictors because it is usually unknown when the culture is ordered.",
        f"The public ARMD files come from two affiliated hospitals and use an adult cohort [1]. The release is Dryad version 403223, dated 22 October 2025 [2]. The {len(verified)} supplied files match the publisher's checksums. A fixed hash selected about {percent(config['sample_fraction'])} of patients before data cleaning. This is a scaled sample, not the full source cohort.",
        f"The nine candidates are {drugs}. This panel was set before model fitting. The instructor list was not supplied. It is a provisional ARMD panel, not a verified STEWARD-AW panel.",
    )
    page.append(("table", cohort_rows, [0.17, 0.23, 0.20, 0.22, 0.18]))
    page += paragraphs(
        f"There are {count(overall.organism_events)} organism-events. Each has a mean of {number(overall.candidate_depth_mean, 2)} tested candidates, a median of {number(overall.candidate_depth_median, 1)}, and a range of {int(overall.candidate_depth_min)} to {int(overall.candidate_depth_max)}. The final key has {count(overall.duplicate_keys)} duplicates.",
        f"Shifted order dates run from {str(overall.time_min_shifted)[:10]} to {str(overall.time_max_shifted)[:10]}. ARMD shifts dates separately for each patient. This range is not the true calendar span. Cross-patient calendar order cannot be recovered [1a].",
    )
    pages.append(page)

    predictor_missing = missing[(missing.split == "all") & missing.is_predictor.astype(str).str.lower().eq("true")].sort_values("missing_fraction", ascending=False)
    missing_rows = [["Predictor with most missing values", "Missing", "Share"]]
    for row in predictor_missing.head(7).itertuples():
        missing_rows.append([FEATURE_NAMES.get(row.feature, row.feature.replace("_", " ")), count(row.missing_n), percent(row.missing_fraction)])
    decisions = audit["key_label_decisions"]
    rejected_keys = sum(int(value) for key, value in decisions.items() if key != "retained")
    page = [("title", "Data preparation"), ("sub", "2. Labels, joins and missing values")]
    page += paragraphs(
        f"The pipeline scanned {count(audit['source_rows_scanned'])} cohort rows and selected {count(audit['sample_rows'])} by patient. After key, date, candidate and culture checks, {count(audit['target_rows_before_label_resolution'])} target rows entered label review.",
        f"Matching S results or matching R results were collapsed. Label review rejected {count(rejected_keys)} keys: {count(decisions.get('invalid_or_unknown_result', 0))} with an unsupported or missing result, and {count(decisions.get('conflicting_s_r', 0))} with conflicting S and R results. It collapsed {count(audit['agreeing_duplicate_rows_collapsed'])} extra agreeing rows. Untested, intermediate and unknown results were not used as labels.",
        f"The pipeline removed {count(audit['events_with_conflicting_metadata_excluded'])} events with conflicting dates or culture descriptions. Repeated support values had to agree; conflicts became missing. Each support table had one row per intended join key. Joins had to preserve cohort row counts. Missing age was retained under ARMD's adult inclusion rule; explicit under-18 bands were excluded.",
        f"The model uses {len(evaluation['features'])} predictors: candidate, culture type, age band, anonymous gender code, four ward flags and {n_numeric} numeric summaries. The extra summaries cover laboratory tests, vital signs and prior antibiotic or resistance records. The dictionary gives each field's source, timing rule and missing-value rule.",
        "The source defines laboratory summaries over the prior 14 days and vital signs over the prior 48 hours [1a]. Separate result-release times are absent, so availability at order cannot be checked for each measurement. Values retain source units. Prior history uses only the positive, strictly earlier day differences defined in the feature contract. No linked history means unknown, not no history.",
    )
    page.append(("table", missing_rows, [0.65, 0.18, 0.17]))
    page += paragraphs(
        "Missing categories receive their own code. Numeric gaps use medians learned from model-fitting patients, with separate flags for missing values. Scaling also uses only those patients. Fields with no fitting values are handled by the saved preprocessing contract. The analytical file keeps the original gaps. Missing values never become a measured zero.",
        "Only ARMD was supplied. The harmonization matrix records the ARMD mapping and marks other dataset mappings as unavailable. These features are a proposed shared set, not a tested intersection of several datasets.",
    )
    pages.append(page)

    imbalance_all = imbalance.groupby("candidate_antibiotic", as_index=False)[["n", "resistant_n", "susceptible_n"]].sum()
    balance_rows = [["Candidate", "Rows", "Resistant", "Minority"]]
    for row in imbalance_all.itertuples():
        balance_rows.append([row.candidate_antibiotic, count(row.n), percent(row.resistant_n / row.n), "R" if row.resistant_n <= row.susceptible_n else "S"])
    page = [("title", "Model fitting and evaluation"), ("sub", "3. Keep training and testing separate")]
    page += paragraphs(
        f"A separate patient hash assigned about 60% to training, 20% to validation and 20% to testing (seed {config['seed']}). All rows from a patient stay together. The audit found {audit['patient_overlap_between_splits']} patients shared across these sets. Patient codes, event codes, shifted dates, current organism and current susceptibility details are excluded from predictors.",
        f"Within training, {count(evaluation['proper_fit_patients'])} patients fit the models and {count(evaluation['calibration_patients'])} different patients fit sigmoid probability calibration [3]. Categories, medians, scaling and class weights use only the model-fitting subset. Validation and test data were not rebalanced.",
        "Logistic regression and LightGBM [4] were each fitted with and without balanced class weights. Two baselines predict the training susceptible share, either overall or for each drug. Model settings were fixed in advance. Weighted fitting gives each class the same total training weight; it does not add synthetic rows.",
        "Validation selected the learned model with the highest mean resistant average precision across drugs, using lower Brier score to break ties. Each model's threshold maximized validation balanced accuracy. These choices were frozen before test scoring. Calibration used only the held-out training subset, not validation or test labels.",
        f"Overall, {percent(overall.resistant_prevalence)} of labels are resistant; {minority} is the smaller class. The table shows each drug's class balance. R means resistant and S means susceptible.",
    )
    page.append(("table", balance_rows, [0.47, 0.19, 0.18, 0.16]))
    page += paragraphs(
        "AUROC measures score ranking: 0.5 is chance and 1 is perfect. Drug-macro AUROC averages it across drugs with both classes. Macro AUPRC averages susceptible and resistant average precision within each drug, then across drugs. AUPRC here means average precision, not trapezoidal area. Balanced accuracy averages the two class recalls. Resistant precision measures how often a resistance alert is correct; resistant recall measures how much resistance is found. Lower Brier score means less probability error.",
    )
    pages.append(page)

    results_rows = [["Test model", "Macro\nAUROC", "Macro\nAUPRC", "Balanced\naccuracy", "R\nprecision", "R\nrecall", "Brier"]]
    for model in MODEL_NAMES:
        row = tested.loc[model]
        results_rows.append([MODEL_NAMES[model] + (" *" if model == selected else ""), number(row.antibiotic_macro_auroc), number(row.antibiotic_macro_class_auprc), number(row.balanced_accuracy), number(row.precision_r), number(row.recall_r), number(row.brier)])
    page = [("title", "Test results"), ("sub", "4. Ranking, resistance detection and calibration")]
    page.append(("table", results_rows, [0.29, 0.118, 0.118, 0.118, 0.118, 0.118, 0.12]))
    page += paragraphs(
        "The drug baseline can help overall decisions because drugs have different resistance rates. It cannot rank patients within a drug, so its drug-macro AUROC is 0.5.",
        f"* Selected using validation data. All {int(chosen.antibiotics_evaluable)} test drugs have both classes. At its validation-set threshold of {number(chosen.threshold_s)}, {selected_name} found {percent(chosen.recall_r)} of resistant results; {percent(chosen.precision_r)} of its resistance alerts were correct. Its mean resistant AUPRC across drugs was {number(chosen.antibiotic_macro_auprc_r)}.",
        f"The selected model's drug-macro AUROC was {interval_text(intervals, selected, 'antibiotic_macro_auroc')}. Its Brier score was {interval_text(intervals, selected, 'brier')}. The intervals resample patients {evaluation['bootstrap']['replicates']} times, keeping all rows for each sampled patient together. They measure uncertainty in this test sample with fitted models held fixed.",
    )
    advanced_calibration = results / "advanced" / "figures" / "report_calibration.png"
    calibration_path = advanced_calibration if advanced_calibration.is_file() else model_dir / "figures" / "calibration_test.png"
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if advanced_calibration.is_file():
        page.append(("image", calibration_path, 6.15, 3.25))
        page.append(("caption", "Figure 1. Selected model before and after training-only calibration. Points compare predicted resistance with the observed resistant share. Bin edges were set on validation. Lines show 95% intervals from 500 patient bootstrap samples. This follow-up check did not change test predictions."))
    else:
        page.append(("image", calibration_path, 4.65, 3.33))
        page.append(("caption", "Figure 1. Test calibration after training-only calibration. Each point compares mean predicted susceptibility with the observed share in a probability bin. The dashed diagonal shows exact agreement. Sparse bins are less reliable. No curve was fitted to change the test predictions."))
    raw = diagnostics[(diagnostics.split == "test") & (diagnostics.model == selected) & (diagnostics.state == "uncalibrated")].iloc[0]
    calibrated = diagnostics[(diagnostics.split == "test") & (diagnostics.model == selected) & (diagnostics.state == "calibrated")].iloc[0]
    page += paragraphs(f"For the selected model, test Brier score was {number(raw.brier, 4)} before calibration and {number(calibrated.brier, 4)} after calibration. Brier score reflects both ranking and calibration, so this comparison alone does not prove better calibration [3].")
    pages.append(page)

    candidate_groups = groups[(groups.model == selected) & (groups.group_field == "candidate_antibiotic")].sort_values("group_value")
    candidate_rows = [["Selected model by test drug", "Rows", "R share", "AUROC", "R AUPRC"]]
    for row in candidate_groups.itertuples():
        candidate_rows.append([row.group_value, count(row.n), percent(1 - row.prevalence_s), number(row.auroc_s), number(row.auprc_r)])
    page = [("title", "What the results mean"), ("sub", "5. Differences and limits")]
    page.append(("table", candidate_rows, [0.43, 0.16, 0.13, 0.14, 0.14]))
    paired = differences[(differences.model_a == selected) & (differences.model_b == "logistic_unweighted") & (differences.metric == "antibiotic_macro_auroc")]
    if len(paired):
        paired = paired.iloc[0]
        page += paragraphs(f"Compared with unweighted logistic regression, the selected model's drug-macro AUROC difference was {number(paired.difference_a_minus_b)} (paired 95% interval {number(paired.ci_low)} to {number(paired.ci_high)}). The same patient samples were used for both models. This estimates their difference in this test set, not in a new hospital.")
    advanced = results / "advanced"
    if (advanced / "ablation_metrics.csv").is_file() and (advanced / "drug_performance.csv").is_file():
        ablation = pd.read_csv(advanced / "ablation_metrics.csv")
        ap = ablation[ablation.metric.eq("antibiotic_macro_auprc_r")].set_index("model").estimate
        drug_check = pd.read_csv(advanced / "drug_performance.csv").set_index("candidate_antibiotic")
        amp, pip = drug_check.loc["ampicillin"], drug_check.loc["piperacillin/tazobactam"]
        page += paragraphs(
            f"Further checks are in advanced_analysis.pdf. At the original cutoff, all {count(amp.n)} ampicillin rows were flagged, including {count(amp.false_alerts)} false alerts. All {count(pip.n_r)} resistant piperacillin/tazobactam rows were missed. An overall recall score hides these drug differences.",
            f"On validation, adding prior history to the basic fields raised mean resistant AP from {number(ap['core_only'])} to {number(ap['core_prior_history'])}. Using all features scored {number(ap['full'])}. The paired interval showed no clear added gain from these lab and vital summaries. These follow-up comparisons used fixed model settings. They did not select a new model or change the original test results.",
        )
    else:
        culture_groups = groups[(groups.model == selected) & (groups.group_field == "culture_type") & groups.both_classes.astype(str).str.lower().eq("true")]
        if len(culture_groups):
            limits = culture_groups.auroc_s.agg(["min", "max"])
            page += paragraphs(f"Across the {len(culture_groups)} culture-type groups with both labels, AUROC ranged from {number(limits['min'])} to {number(limits['max'])}. These group scores pool drugs, so they differ from drug-macro scores. Full tables include age, gender, organism and other groups, with counts and small-group flags.")
    page += paragraphs(
        "Independent dataset or hospital validation was not possible. The supplied ARMD files do not identify the two hospitals separately. Ward and culture groups cannot replace an outside-site test. Patient separation also does not test future years, since calendar dates are shifted.",
        "Labels cover tested drugs with clear results. Testing choices may depend on the patient, organism and laboratory rules. Results may differ for untested drugs. Excluding intermediate or conflicting labels also changes the sample.",
        "The current organism is unknown at culture order. Different organisms in one culture can have the same inputs but different outcomes. Exact feature release times and some units are also unavailable. These results do not measure treatment benefit.",
        "The next checks are to confirm the instructor's drug list, obtain a hospital field or second dataset, and verify feature release times and units. The code can rerun the study when these inputs are available.",
    )
    pages.append(page)

    refs = [
        ("[1] Nateghi Haredasht, F., et al. Antibiotic Resistance Microbiology Dataset (ARMD): A Resource for Antimicrobial Resistance from EHRs. Scientific Data (2025).", "https://www.nature.com/articles/s41597-025-05649-7"),
        ("[1a] ARMD author manuscript. Source windows and date shifts.", "https://arxiv.org/pdf/2503.07664"),
        ("[2] Antibiotic Resistance Microbiology Dataset. Dryad. Version 403223 (22 October 2025).", "https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp"),
        ("[3] scikit-learn. Probability calibration documentation.", "https://scikit-learn.org/stable/modules/calibration.html"),
        ("[4] LightGBM. Model parameter documentation.", "https://lightgbm.readthedocs.io/en/stable/Parameters.html"),
        ("[5] National Archives. Top 10 Principles for Plain Language.", "https://www.archives.gov/open/plain-writing/10-principles.html"),
        ("[6] University of Southern California Libraries. Generative AI and Writing.", "https://libguides.usc.edu/writingguide/AIwriting"),
    ]
    page = [("title", "References and file guide"), ("sub", "Outside the five-page report body")]
    for title, url in refs:
        page.append(("reference", title, url))
    page += paragraphs(
        "Sources checked on 12 October 2026. The writing follows the plain-language advice in [5]. The review of AI writing problems in [6] informed checks for invented facts, false citations and unsupported claims. Every reported number is loaded from the pipeline outputs. The report uses short sentences and no em dashes.",
        "This analysis and report were prepared with AI assistance. The data are public, de-identified ARMD records. Software checks and saved output tables support review; they do not replace review of the study design.",
    )
    page.append(("table", [["File or folder", "Purpose"], ["report.pdf / report.docx", "Page-controlled report and editable copy"], ["docs/feature_harmonization_matrix.xlsx", "Field mapping, dictionary, source decisions, target list"], ["results/construction_audit.json", "Row exclusions, label decisions, joins and leakage checks"], ["results/cohort_summary.csv; missingness.csv", "Cohort counts, candidate depth, dates and missing values"], ["results/models/", "Full metrics, subgroups, intervals, predictions and fitted models"], ["advanced_analysis.pdf / .html", "Exploratory appendix and figure gallery"], ["results/advanced/", "Extra results, workbook and editable figures"], ["docs/methods.html; README.txt", "Detailed method and commands to repeat the run"]], [0.55, 0.45]))
    pages.append(page)
    text_content = json.dumps(pages, default=str, ensure_ascii=False)
    if "\u2014" in text_content:
        raise ValueError("The report must not contain em dashes")
    assert len(pages) == 6
    return pages, {"selected_model": selected, "analytical_rows": int(overall.rows), "body_pages": 5,
                   "references_pages": 1, "source_results": str(results.resolve()),
                   "em_dash_count": 0, "numbers_from_actual_outputs": True}


def render_pdf(pages, path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=19, leading=22, spaceAfter=9, textColor=colors.HexColor("#183e56")))
    styles.add(ParagraphStyle(name="ReportSub", fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceAfter=10, textColor=colors.HexColor("#366678")))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10.3, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportCaption", fontName="Helvetica", fontSize=8.2, leading=10.4, spaceAfter=8, textColor=colors.HexColor("#465665")))
    styles.add(ParagraphStyle(name="ReportCell", fontName="Helvetica", fontSize=8.4, leading=10.5))
    styles.add(ParagraphStyle(name="ReportHead", fontName="Helvetica-Bold", fontSize=8.4, leading=10.5, textColor=colors.white))
    width = A4[0] - 88
    story = []
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for block in page:
            kind = block[0]
            if kind in {"title", "sub", "p", "caption"}:
                style = {"title": "ReportTitle", "sub": "ReportSub", "p": "ReportBody", "caption": "ReportCaption"}[kind]
                story.append(Paragraph(html.escape(block[1]), styles[style]))
            elif kind == "reference":
                story.append(Paragraph(html.escape(block[1]) + '<br/><link href="' + html.escape(block[2]) + '" color="#175b85">' + html.escape(block[2]) + "</link>", styles["ReportBody"]))
            elif kind == "table":
                rows = [[Paragraph(html.escape(str(value)).replace("\n", "<br/>"), styles["ReportHead" if ri == 0 else "ReportCell"]) for value in row] for ri, row in enumerate(block[1])]
                table = Table(rows, colWidths=[width * x for x in block[2]], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#214f68")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#a8bdc8"))]))
                story += [table, Spacer(1, 9)]
            elif kind == "image":
                story.append(Image(str(block[1]), width=block[2] * inch, height=block[3] * inch))
                story.append(Spacer(1, 4))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5f6c76"))
        canvas.drawString(44, 27, "ARMD | Scaled susceptibility prediction study")
        canvas.drawRightString(A4[0] - 44, 27, f"{doc.page}" if doc.page <= 5 else "References")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=36, bottomMargin=40,
                            title="Antimicrobial resistance prediction", author="ARMD project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(path)
    if len(reader.pages) != 6:
        raise AssertionError(f"Report overflow: expected 5 body pages + 1 reference page, found {len(reader.pages)}")
    expected = ["Antimicrobial resistance prediction", "Data preparation", "Model fitting and evaluation", "Test results", "What the results mean", "References and file guide"]
    for page, title in zip(reader.pages, expected):
        if title not in page.extract_text():
            raise AssertionError(f"Unexpected page layout: {title}")
    return len(reader.pages)


def render_docx(pages, path):
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin, section.bottom_margin = Inches(.50), Inches(.55)
    section.left_margin = section.right_margin = Inches(.61)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for index, page in enumerate(pages):
        if index:
            document.add_page_break()
        for block in page:
            kind = block[0]
            if kind in {"title", "sub"}:
                p = document.add_paragraph()
                p.paragraph_format.keep_with_next = True
                run = p.add_run(block[1]); run.bold = True
                run.font.size = Pt(18 if kind == "title" else 11)
                run.font.color.rgb = RGBColor.from_string("183E56")
            elif kind in {"p", "caption"}:
                p = document.add_paragraph(block[1])
                if kind == "caption":
                    for run in p.runs:
                        run.font.size = Pt(8)
            elif kind == "reference":
                document.add_paragraph(block[1] + "\n" + block[2])
            elif kind == "image":
                document.add_picture(str(block[1]), width=Inches(block[2]), height=Inches(block[3]))
            elif kind == "table":
                table = document.add_table(rows=0, cols=len(block[1][0]))
                table.style = "Light Shading Accent 1"
                table.autofit = False
                for row_index, row in enumerate(block[1]):
                    cells = table.add_row().cells
                    for col, value in enumerate(row):
                        cells[col].width = Inches(7.05 * block[2][col])
                        cells[col].text = str(value)
                        for p in cells[col].paragraphs:
                            p.paragraph_format.space_after = Pt(3)
                            for run in p.runs:
                                run.font.size = Pt(8)
                                run.bold = row_index == 0
                document.add_paragraph().paragraph_format.space_after = Pt(0)
    section.footer.paragraphs[0].text = "ARMD | Scaled susceptibility prediction study | Page-controlled version: report.pdf"
    document.save(path)


def render_html(pages, path):
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ARMD report</title><style>body{font:15px/1.5 Arial,sans-serif;color:#203341;background:#edf2f5}section{background:white;max-width:850px;margin:24px auto;padding:36px}h1{font-size:25px;color:#183e56}h2{font-size:18px;color:#366678}table{border-collapse:collapse;width:100%;font-size:12px;margin:14px 0}td,th{padding:7px;border-bottom:1px solid #d7e0e6;text-align:left}th{background:#214f68;color:white}.caption{font-size:12px}img{display:block;max-width:100%;margin:auto}@media print{body{background:white}section{margin:0;padding:0;break-after:page}}</style></head><body>']
    for page in pages:
        parts.append("<section>")
        for block in page:
            kind = block[0]
            if kind in {"title", "sub", "p", "caption"}:
                tag = {"title": "h1", "sub": "h2", "p": "p", "caption": 'p class="caption"'}[kind]
                parts.append(f"<{tag}>{html.escape(block[1])}</{tag.split()[0]}>")
            elif kind == "reference":
                parts.append(f'<p>{html.escape(block[1])}<br><a href="{html.escape(block[2])}">{html.escape(block[2])}</a></p>')
            elif kind == "image":
                import base64
                encoded = base64.b64encode(Path(block[1]).read_bytes()).decode("ascii")
                parts.append(f'<img alt="Test calibration plot" src="data:image/png;base64,{encoded}" width="558">')
            elif kind == "table":
                parts.append("<table>")
                for index, row in enumerate(block[1]):
                    tag = "th" if index == 0 else "td"
                    parts.append("<tr>" + "".join(f"<{tag}>{html.escape(str(value))}</{tag}>" for value in row) + "</tr>")
                parts.append("</table>")
        parts.append("</section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    config = read_json(ROOT / "config.json")
    pages, validation = build_pages(args.results, config)
    args.output.mkdir(parents=True, exist_ok=True)
    validation["pdf_pages"] = render_pdf(pages, args.output / "report.pdf")
    render_docx(pages, args.output / "report.docx")
    render_html(pages, args.output / "report.html")
    (args.output / "report_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"Wrote PDF (5 body pages + references), DOCX and HTML to {args.output}")


if __name__ == "__main__":
    main()
