"""Central configuration: paths, column groups, sentinel patterns, plausibility ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = ROOT / "merged_all_years_normalized.xlsx"
RAW_SHEET = "All Data"

DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PAPER = ROOT / "paper"

for _p in (INTERIM, PROCESSED, TABLES, FIGURES, PAPER):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260801

# ---------------------------------------------------------------- raw columns
COL_SEX = "جنسیت"
COL_VISIT_DATE = "تاریخ آخرین پذیرش"
COL_BIRTH_DATE = "تاریخ تولد"
COL_MARITAL = "وضعیت تاهل"
COL_EMPLOYER = "آخرین نقش فرد"
COL_BMI = "شاخص توده بدنی"
COL_SBP = "فشار خون سیستول"
COL_DBP = "فشار خون دیاستول"
COL_DX_CURRENT = "ابتلا به بیماری در حال حاضر"
COL_DX_HISTORY = "سابقه ابتلاء به بیماری"
COL_DX_FAMILY = "سابقه ابتلا به بیماری در خانواده"

SEX_MALE = "مرد"
SEX_FEMALE = "زن"

# Values that encode "this test was not performed / not recorded" rather than a
# measurement. Stored as text in the source export, so isna() alone understates
# missingness by a large margin (e.g. carotid Doppler reads as 1% missing but is
# actually 95% missing).
SENTINEL_PATTERNS = [
    r"not registered",
    r"not_registered",
    r"ثبت نشده",
    r"انجام نداده",
    r"انجام نشده",
    r"نامشخص",
]

# --------------------------------------------------- cheap "core" feature panel
# Available to (nearly) every attendee of the programme: venous blood chemistry,
# haematology, and bedside anthropometry/vitals. These are the inputs the triage
# model is allowed to use.
CORE_CHEMISTRY = [
    "Fasting Blood Sugar", "Blood Urea", "Creatinine", "Uric Acid",
    "Cholesterol", "Triglycerides", "HDL cholestrol", "LDL cholestrol",
    "Calcium", "Phosphorus", "Total Bilirubin", "Direct bilirubin",
    "SGOT (AST)", "SGPT (ALT)", "Alkaline phosphatase", "ESR",
    "TSH", "VT D3", "FER", "IRON", "TIBC", "A1C",
]
CORE_HAEMATOLOGY = [
    "HB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDW.CV", "RDW.SD",
    "WBC.1", "RBC.1", "N", "L", "M", "EO", "BAS", "MPV", "PDW",
]
CORE_VITALS = [COL_BMI, COL_SBP, COL_DBP]
CORE_DEMOGRAPHICS = ["age", "male"]

CORE_NUMERIC = CORE_CHEMISTRY + CORE_HAEMATOLOGY + CORE_VITALS

# Feature blocks used for the ablation ladder.
FEATURE_BLOCKS = {
    "demographics": CORE_DEMOGRAPHICS,
    "vitals": CORE_VITALS,
    "chemistry": CORE_CHEMISTRY,
    "haematology": CORE_HAEMATOLOGY,
}

ABLATION_LADDER = {
    "A_demographics": CORE_DEMOGRAPHICS,
    "B_plus_vitals": CORE_DEMOGRAPHICS + CORE_VITALS,
    "C_plus_chemistry": CORE_DEMOGRAPHICS + CORE_VITALS + CORE_CHEMISTRY,
    "D_full_core": CORE_DEMOGRAPHICS + CORE_VITALS + CORE_CHEMISTRY + CORE_HAEMATOLOGY,
}

# ------------------------------------------------ physiological plausible ranges
# Values outside these bounds are unit-entry errors, not biology. Verified against
# the observed distributions: e.g. PLT reaches 150000 (should be ~150 x10^9/L),
# ferritin 91118, LDL 10564, iron goes negative, BMD reaches 13259.
PLAUSIBLE_RANGE = {
    "Fasting Blood Sugar": (40, 500), "Blood Urea": (2, 200), "Creatinine": (0.2, 15),
    "Uric Acid": (0.5, 20), "Cholesterol": (50, 500), "Triglycerides": (10, 2000),
    "HDL cholestrol": (10, 150), "LDL cholestrol": (10, 350),
    "Calcium": (5, 15), "Phosphorus": (1, 10),
    "Total Bilirubin": (0.05, 20), "Direct bilirubin": (0.01, 15),
    "SGOT (AST)": (2, 1000), "SGPT (ALT)": (2, 1000),
    "Alkaline phosphatase": (20, 2000), "ESR": (1, 150),
    "TSH": (0.005, 150), "VT D3": (2, 150), "FER": (1, 2000),
    "IRON": (5, 500), "TIBC": (100, 700), "A1C": (3, 20),
    "HB": (5, 22), "HCT": (15, 70), "MCV": (50, 130), "MCH": (15, 45),
    "MCHC": (25, 40), "PLT": (10, 1000), "RDW.CV": (8, 30), "RDW.SD": (20, 100),
    "WBC.1": (1, 60), "RBC.1": (2, 9),
    "N": (5, 95), "L": (2, 90), "M": (0, 30), "EO": (0, 30), "BAS": (0, 10),
    "MPV": (5, 20), "PDW": (5, 30),
    COL_BMI: (12, 70), COL_SBP: (70, 260), COL_DBP: (35, 160),
}

# DEXA is exported as one column per WHO category per site; the measurement is
# filed into whichever column matches its category. Coalesce back to one value.
DEXA_SITES = ["L.Hip", "Lumbar spine", "Thoracic spine", "R.Hip"]
DEXA_CATEGORIES = {"طبیعی": "normal", "استئوپنی": "osteopenia", "استئوپوروز": "osteoporosis"}
DEXA_RANGE = {"BMD": (0.2, 2.0), "T.score": (-6, 6), "Z.score": (-6, 6)}

# Columns known to be corrupted in the source export (values belong to a
# different field). Excluded from all analyses; documented in the QC appendix.
CORRUPTED_COLUMNS = {
    "bph": "values are 'Left'/'Right' (kidney-stone laterality), not BPH grade",
}

MIN_EMPLOYER_N = 300  # minimum visits for an employer to form its own protocol fold
