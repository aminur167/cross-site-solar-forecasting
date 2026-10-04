# STEP-BY-STEP JOURNAL SUBMISSION GUIDE FOR *ENERGY AND AI* (ELSEVIER)

**Manuscript Title:** *Data-Efficient and Uncertainty-Calibrated Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts*  
**Principal Investigator:** Aminur Rahman (HSTU, Bangladesh)  
**Target Journal:** *Energy and AI* (Elsevier) — ISSN: 2666-5468

---

## Phase 1: 1-Click Verification on Overleaf

Before submitting to the journal portal, verify that the LaTeX project compiles cleanly into a formatted PDF:

1. Download or locate `paper_submission_package.zip` in your project directory (or click the **Download (.zip)** button in Tab 8 of your local Streamlit dashboard).
2. Go to **[Overleaf.com](https://www.overleaf.com/)** and log in.
3. Click **New Project** $\to$ **Upload Project**.
4. Select `paper_submission_package.zip`.
5. Overleaf will automatically extract all files (`main.tex`, `references.bib`, tables, and figures).
6. Click **Recompile** in Overleaf. The complete, two-column Elsevier article PDF will be generated immediately.
7. Download the compiled PDF to keep a reference copy on your computer.

---

## Phase 2: Elsevier Editorial Manager Portal Setup

1. Open the official journal submission portal:  
   👉 **[https://www.editorialmanager.com/egai/](https://www.editorialmanager.com/egai/)**
2. If you do not have an account, click **Register Now** (use your institutional email `aminurhstu23@gmail.com` or HSTU email and link your ORCID ID if available).
3. Log in as an **Author**.
4. Click **"Submit New Manuscript"**.

---

## Phase 3: Step-by-Step Submission Workflow

### Step 1: Article Type Selection
* Select **"Full Length Article"** (or "Research Paper").

### Step 2: Enter Manuscript Title & Running Head
* **Full Title:** `Data-Efficient and Uncertainty-Calibrated Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts`
* **Short Title / Running Head:** `Cross-Site Solar PV Transfer Forecasting`

### Step 3: Add Author Information
* **First Name:** Aminur
* **Last Name:** Rahman
* **Institution:** Hajee Mohammad Danesh Science and Technology University (HSTU)
* **Department:** Department of Computer Science and Engineering
* **City:** Dinajpur, **Country:** Bangladesh
* Mark yourself as the **Corresponding Author**.

### Step 4: Paste Abstract & Keywords
* Copy the abstract directly from `paper/manuscript/main.tex` or `README.md`.
* **Keywords:**
  1. Solar PV power forecasting
  2. Cross-site transfer learning
  3. Few-shot domain adaptation
  4. Conformal prediction
  5. Uncertainty quantification
  6. Sensor-availability shifts

### Step 5: Upload Submission Files
Upload the following items from your `paper/` directory:
1. **Cover Letter:** Upload `paper/cover_letter.tex` (or export as PDF/DOCX).
2. **Highlights:** Upload `paper/highlights.txt` (Contains 5 bullet points, each strictly $\le 85$ characters).
3. **Manuscript File:** Upload `paper/manuscript/main.tex` (along with `references.bib`).
4. **Figure Files:** Upload `figures/fig1_baseline_benchmark.pdf` through `fig4_shap_feature_importance.pdf` (marked as "Figure").
5. **Table Files:** Upload `tables/table1_baseline_benchmark.tex` through `table5_deep_learning_benchmark.tex` (marked as "Table").
6. **Supplementary Material:** Upload `paper/supplementary/supplementary.tex`.
7. **Author Statements:** Upload `paper/author_statements.txt` (Conflict of Interest and CRediT contributions).

---

## Phase 4: APC Waiver Request (Crucial for Bangladeshi Authors)

* *Energy and AI* is an Open Access journal, but Elsevier provides **Research4Life / Equity Waivers** for eligible developing countries including Bangladesh.
* When the portal asks: *"How will the Article Publishing Charge (APC) be handled?"*:
  * Select: **"I request an APC Waiver / Discount under the Research4Life / Equity Program"**.
  * Enter Institution: **Hajee Mohammad Danesh Science and Technology University (HSTU), Bangladesh**.
  * Note: *As an independent undergraduate researcher from an eligible institution, an automatic 50% to 100% waiver is standardly granted upon peer-reviewed acceptance.*

---

## Phase 5: Suggested Peer Reviewers

The portal will ask you to suggest 3 to 4 independent reviewers who are active experts in renewable energy forecasting. Here are 3 ideal, non-conflicted researchers in this field:

1. **Dr. Yang Zhang**  
   *Institution:* University of Calgary, Canada  
   *Field:* Multi-site solar transfer learning and renewable forecasting.  
   *Reason:* Leading author of cross-site transfer learning benchmarks in *Applied Energy*.

2. **Dr. Chao Huang**  
   *Institution:* Tsinghua University / University of Hong Kong  
   *Field:* Conformalized Quantile Regression and probabilistic smart grid operations.  
   *Reason:* Published conformal prediction for renewable generation in *Energy and AI*.

3. **Dr. Mihai-Cristian Mihaescu**  
   *Institution:* University of Craiova / UPB Bucharest, Romania  
   *Field:* Open photovoltaic generation benchmarking and solar inverter telemetry.  
   *Reason:* Co-creator of the open-source UPB Romanian PV benchmark dataset.

---

## Phase 6: PDF Building & Final Approval

1. Click **Build PDF for Approval**.
2. The system will compile the LaTeX source and figures into a single unified proof PDF.
3. Download and inspect the proof PDF (check that author name, HSTU affiliation, tables, and figures appear properly).
4. Click **"I accept this proof"** and **"Submit Manuscript"**.
5. You will receive an official confirmation email with your manuscript tracking number (e.g., `EGAI-D-26-XXXXX`).
