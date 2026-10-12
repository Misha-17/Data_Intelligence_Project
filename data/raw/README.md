# Antibiotic Resistance Microbiology Dataset (ARMD): A resource for antimicrobial resistance from EHRs

## Background

Antimicrobial resistance (AMR) represents a pressing global health challenge, exacerbated by the overuse and misuse of antibiotics. Efforts to mitigate AMR require high-quality datasets to analyze trends in microbial susceptibility, guide clinical decision-making, and inform stewardship programs. Electronic health records (EHR) are a rich source of real-world data that can be leveraged to study antimicrobial use and resistance patterns. However, constructing meaningful datasets from EHR data requires rigorous curation and preprocessing to ensure accuracy, relevance, and usability. ARMD aims to facilitate research in antimicrobial stewardship, with applications in identifying resistance patterns, evaluating treatment practices, and informing public health interventions. By leveraging de-identified EHR data from Stanford Healthcare, this dataset provides a unique opportunity to generate insights that can help improve infectious disease management and curb the spread of AMR. The dataset includes detailed information on culture positivity, organism identification, and antibiotic susceptibility across 55 antibiotics. By supporting the development of algorithms for predicting susceptibility and selecting effective treatments, ARMD offers researchers the tools to optimize empiric antibiotic therapy while minimizing the overuse of broad-spectrum antibiotics. Additionally, ARMD enables the exploration of broader questions in causal inference and policy learning by leveraging antibiotic susceptibility testing as a proxy for counterfactual outcomes under different treatments. With its large cohort of over 283,000 adult patients and a diverse set of microbiological cultures, this dataset supports a range of research applications, from evaluating resistance patterns to improving clinical guidelines for antimicrobial stewardship.

---

## Description of the data and file structure

### Data Files

#### 1. microbiology_cultures_cohort.csv

**Description:** Contains primary information about microbiological cultures, including culture type, organism identified, and antibiotic susceptibility.

**Key Columns:**

* `culture_description`: Description of the culture (e.g., Urine, Blood).
* `was_positive`: Indicates whether the culture was positive for an organism.
* `organism`: Identified microorganism.
* `antibiotic`: Antimicrobial agent tested.
* `susceptibility`: Categorized susceptibility result (e.g., Susceptible, Resistant, Intermediate).

#### 2. microbiology_cultures_ward_info.csv

**Description:** Provides information about the ward setting where cultures were collected (e.g., ICU, ER).

**Key Columns:**

* `hosp_ward_IP`: Indicates if the culture was taken in an inpatient ward.
* `hosp_ward_OP`: Indicates if the culture was taken in an outpatient setting.
* `hosp_ward_ER`: Indicates if the culture was taken in the emergency department.
* `hosp_ward_ICU`: Indicates if the culture was taken in the ICU.

#### 3. microbiology_cultures_prior_med.csv

**Description:** Details prior medication exposure for patients relative to the culture order.

**Key Columns:**

* `medication_name`: Generic name of the medication.
* `medication_time_to_culturetime`: Days between medication start and culture order.
* `medication_category`: Category of the medication.

#### 4. microbiology_cultures_microbial_resistance.csv

**Description:** Contains data on microbial resistance and the timeline of resistance development.

**Key Columns:**

* `organism`: Microorganism identified.
* `antibiotic`: Antimicrobial agent tested.
* `resistant_time_to_culturetime`: Days between resistance confirmation and culture order.

#### 5. microbiology_cultures_demographics.csv

**Description:** Includes patient demographic information at the time of the culture order.

**Key Columns:**

* `age`: Patient age categorized into bins (e.g., 18–24, 25–34, etc.). Patients aged 89 years or older are grouped into a single category (90+).
* `gender`: Gender is encoded as binary values (0 or 1) without indicating which value corresponds to male or female.

#### 6. microbiology_cultures_labs.csv

**Description:** Includes detailed laboratory results recorded within specific time windows relative to culture orders, including summary statistics (median, Q25, Q75) for key laboratory metrics.

**Key Columns:**

* `Period_Day`: Window frame representing the number of days from the culture order during which lab measurements were recorded.
* **Laboratory Metrics (summary statistics: Q25, Median, Q75):**
  * White blood cell count (wbc), neutrophils (neutrophils), lymphocytes (lymphocytes).
  * Hemoglobin (hgb), platelets (plt), sodium (na), bicarbonate (hco3), blood urea nitrogen (bun), creatinine (cr).
  * Lactate (lactate), procalcitonin (procalcitonin).
* **Laboratory Metrics (first and last recorded values):**
  * `first_wbc`, `last_wbc`, `first_hgb`, `last_hgb`, `first_cr`, `last_cr`, etc.

#### 7. microbiology_cultures_vitals.csv

**Description:** Contains detailed vital sign measurements near the time of the culture order, including summary statistics and first and last recorded values.

**Key Columns:**

* **Vital Signs (summary statistics: Q25, Median, Q75):**
  * Heart rate (`heartrate`), respiratory rate (`resprate`), body temperature (`temp`), systolic blood pressure (`sysbp`), diastolic blood pressure (`diasbp`).
* **Vital Signs (first and last recorded values):**
  * `first_heartrate`, `last_heartrate`, `first_resprate`, `last_resprate`, `first_temp`, `last_temp`, `first_sysbp`, `last_sysbp`, `first_diasbp`, `last_diasbp`.

#### 8. microbiology_cultures_antibiotic_class_exposure.csv

**Description:** Tracks prior exposure to antibiotic classes.

**Key Columns:**

* `antibiotic_class`: The antibiotic class.
* `time_to_culturetime`: Days between antibiotic exposure and culture order.

#### 9. microbiology_cultures_antibiotic_subtype_exposure.csv

**Description:** Details prior exposure to antibiotic subclasses.

**Key Columns:**

* `antibiotic_subtype`: The antibiotic subclass.
* `time_to_culturetime`: Days between exposure and culture order.

#### 10. microbiology_culture_prior_infecting_organism.csv

**Description:** Contains data on prior infecting organisms identified in previous microbiological cultures for each patient.

**Key Columns:**

* `prior_organism`: Indicates the presence of a prior infection caused by this organism.
* `prior_infecting_organism_days_to_culture`: Days between the previously recorded infection and the culture order.

#### 11. microbiology_cultures_comorbidity.csv

**Description:** Contains detailed information on comorbidities.

**Key Columns:**

* `comorbidity_component`: Specific component of either the AHRQ CCSR diagnosis or Elixhauser comorbidity index.
* `comorbidity_component_start_days_culture`: Days between the start of the component and the culture order.
* `comorbidity_component_end_days_culture`: Days between the end of the component and the culture order (NULL indicates ongoing conditions).

#### 12. microbiology_cultures_priorprocedures.csv

**Description:** Lists procedures performed on patients before culture orders.

**Key Columns:**

* `procedure_name`: Name of the procedure (e.g., Central Venous Catheter, Mechanical Ventilation).
* `procedure_time_to_culturetime`: Days between procedure and culture order.

#### 13. microbiology_cultures_adi_scores.csv

**Description:** Contains Area Deprivation Index (ADI) data mapped to cohort ZIP codes.

**Key Columns:**

* `adi_score`: ADI score representing socioeconomic disadvantage.
* `adi_state_rank`: State-level rank for the ADI score.

#### 14. microbiology_cultures_nursing_home_visits.csv

**Description:** Tracks nursing home visits relative to the culture order date.

**Key Columns:**

* `nursing_home_visit_culture`: Days between the nursing home visit and the culture order.

#### 15. microbiology_cultures_implied_susceptibility.csv

**Description:** Determines the implied susceptibility of organisms to antibiotics.

**Key Columns:**

* `Implied_susceptibility`: The inferred susceptibility of organisms to an antibiotic.

#### 16. implied_susceptibility_rules.csv

**Description:** Defines the rules used to infer susceptibility relationships between antibiotics.

**Key Columns:**

* `Organism`: The organism for which the susceptibility rules apply.
* `Antibiotic`: The antibiotic for which susceptibility is inferred.
* `Rule`: The logic or condition used to determine inferred susceptibility.

### Linking Across Files

The files are linked using:

* `anon_id`: De-identified patient identifier.
* `pat_enc_csn_id_coded`: Patient encounter identifier.
* `order_proc_id_coded`: Unique culture order identifier.
* `order_time_jittered_utc`: Jittered timestamp.

---

## De-Identification

To ensure compliance with HIPAA guidelines and protect patient privacy:

* Patient IDs are anonymized, and all identifiable information is removed.
* Ages are grouped into bins (e.g., "18–24", "25–35") to prevent exact identification.
* Gender is anonymized as binary values (0 and 1) without further specification.
* Temporal data (e.g., culture collection times) are adjusted using jittering to maintain meaningful relationships without revealing exact dates.

---

## Sharing/Access Information

* This dataset is publicly available on Dryad for research purposes.
* Users are encouraged to contact the dataset creators for support or further clarification if needed.

---

## Code/Software

* No scripts are included in the dataset. However, users may contact the authors if they require guidance on using the data or developing analytical workflows.
* Analysis software such as Python, R, or statistical platforms like SPSS or SAS can be used to interpret and analyze the dataset.

---

## Usage Notes

This dataset is ideal for researchers and clinicians aiming to:

* Analyze trends in antimicrobial resistance.
* Develop predictive models for empirical antibiotic selection.
* Study the impact of comorbidities, nursing home visits, and clinical variables on resistance patterns.

### Handling Missing Data

* Empty cells in the dataset are represented as "null" to ensure clarity.

### Ethical Considerations

* This study was approved by the Stanford University Institutional Review Board (IRB) under eProtocol #70466. The IRB determined the study involves minimal risk, and patient consent was waived due to the use of de-identified retrospective data.

---

## Acknowledgments

This dataset was developed with support from Stanford University and funded by the National Institutes of Health (NIH) R01 grant. The authors acknowledge all contributors who assisted in creating and validating this dataset.

---

## Citation

When using this dataset, please cite:

Nateghi Haredasht, F., et al. *Antibiotic Resistance Microbiology Dataset (ARMD).* Stanford Healthcare, 2025. [arXiv:2503.07664](https://doi.org/10.48550/arXiv.2503.07664).

## Contact

For questions, clarifications, or further support, please contact:

**Fateme Nateghi Haredasht, PhD**
Stanford University
[fnateghi@stanford.edu](mailto:fnateghi@stanford.edu)
