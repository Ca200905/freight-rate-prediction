from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

CHART = "scorer_results/candidate_december.png"
OUT   = "Freight_Rate_Prediction_Report.pdf"

# Replace any special chars that latin-1 can not encode
def s(text):
    return (text
        .replace("\u2014", "-")   # em dash
        .replace("\u2013", "-")   # en dash
        .replace("\u2019", "'")   # right single quote
        .replace("\u2018", "'")   # left single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u00d7", "x")   # multiplication sign
        .replace("\u2192", "->")  # right arrow
    )

class PDF(FPDF):
    def header(self):
        self.set_fill_color(6, 74, 86)
        self.rect(0, 0, 210, 8, "F")
        self.ln(10)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, s(f"Freight Rate Prediction  |  Page {self.page_no()}"), align="C")

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(6, 74, 86)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, s(f"  {text}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(30, 30, 30)
        self.ln(2)

    def body(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, s(text))
        self.ln(1)

    def bullet(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 5.5, s(f"  *  {text}"))

    def kv(self, key, val, shade=False):
        if shade: self.set_fill_color(237, 245, 247)
        else:     self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(6, 74, 86)
        self.cell(68, 7, s(f"  {key}"), fill=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, s(f"  {val}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

    def table_header(self, cols, widths):
        self.set_fill_color(6, 74, 86)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        for col, w in zip(cols, widths):
            self.cell(w, 7, s(f" {col}"), fill=True)
        self.ln()
        self.set_text_color(40, 40, 40)

    def table_row(self, cells, widths, shade=False, bold_last=False):
        if shade: self.set_fill_color(237, 245, 247)
        else:     self.set_fill_color(255, 255, 255)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if bold_last and i == len(cells)-1:
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(6, 74, 86)
            else:
                self.set_font("Helvetica", "", 9)
                self.set_text_color(40, 40, 40)
            self.cell(w, 6.5, s(f" {cell}"), fill=True)
        self.ln()

pdf = PDF()
pdf.set_margins(15, 15, 15)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# == TITLE BLOCK ==
pdf.set_fill_color(6, 74, 86)
pdf.rect(0, 8, 210, 55, "F")
pdf.set_y(18)
pdf.set_font("Helvetica", "B", 24)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 12, "Freight Rate Prediction", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 13)
pdf.cell(0, 8, "ML Engineering Assessment  |  Spotter AI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(180, 220, 228)
pdf.cell(0, 6, "August 2026", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(30, 30, 30)

# Quick stats boxes
stats = [("Ensemble MAE", "$112.87"), ("MAPE", "5.20%"), ("Predictions", "12,000")]
x0 = 15
for label, val in stats:
    pdf.set_xy(x0, 72)
    pdf.set_fill_color(237, 245, 247)
    pdf.rect(x0, 72, 55, 20, "F")
    pdf.set_xy(x0, 75)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(6, 74, 86)
    pdf.cell(55, 7, val, align="C")
    pdf.set_xy(x0, 83)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(55, 5, label, align="C")
    x0 += 62
pdf.set_text_color(30, 30, 30)
pdf.ln(28)

# == SECTION 1 ==
pdf.section_title("1.  Data Exploration - Key Findings")
pdf.body("The training dataset has 48,000 freight loads (Jan-Oct 2025) across three equipment types. Key findings:")
for f in [
    "distance has a 0.91 Pearson correlation with posted_rate -- the single dominant driver.",
    "Average load: $2,374 (median $2,031, std $1,486). Range: $57 to $25,533.",
    "market_index and quote_signal are strong secondary predictors capturing live market conditions.",
    "Equipment type matters: Reefer > Dry Van > Flatbed for typical per-mile rates.",
    "Freight market peaked May-June 2025 (index ~1.30) then softened by August (~0.89).",
    "Target (posted_rate) is right-skewed -- log-transform produces a near-normal distribution.",
]:
    pdf.bullet(f)
pdf.ln(2)

pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(6, 74, 86)
pdf.cell(0, 6, "Dataset Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(40, 40, 40)
for key, val, shade in [
    ("Training rows", "48,000", True),
    ("Date range", "2025-01-01 to 2025-10-31", False),
    ("Equipment types", "Dry Van (56.7%) | Reefer (25.1%) | Flatbed (18.2%)", True),
    ("Mean posted_rate", "$2,374  (std $1,486)", False),
    ("Median posted_rate", "$2,031", True),
    ("Max posted_rate", "$25,533", False),
    ("distance vs rate correlation", "0.91 (strongest predictor)", True),
]:
    pdf.kv(key, val, shade)
pdf.ln(3)

# == SECTION 2 ==
pdf.section_title("2.  Data Quality Issues & Resolutions")
for title, detail in [
    ("weight -- 300 missing (0.6%)",
     "Filled with median per equipment type (not global median). Reefer and Dry Van trucks carry structurally different weights, so global imputation would introduce bias."),
    ("market_index -- 374 missing (0.8%)",
     "Filled using a 50-row rolling mean on time-sorted data, then global median as fallback. Rolling mean respects temporal structure -- the market index on a given day is best estimated from nearby days."),
    ("Extreme posted_rate outliers -- 240 rows above $6,511 (99.5th pct)",
     "Capped at the 99.5th percentile for training only. These are real loads but allowing their full weight distorts gradient descent for the other 99.5% of data. Predictions are capped at 99.9th percentile ($12,855) to prevent Ridge extrapolation from producing implausible rates."),
]:
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(6, 74, 86)
    pdf.set_x(pdf.l_margin + 4)
    pdf.cell(0, 6, s(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(60, 60, 60)
    pdf.set_x(pdf.l_margin + 8)
    pdf.multi_cell(0, 5, s(detail))
    pdf.ln(1)
pdf.set_text_color(40, 40, 40)

# == SECTION 3 ==
pdf.section_title("3.  Feature Engineering  (33 features from 13 raw columns)")
for group, items in [
    ("Time Features", [
        "day_of_week, month, quarter, day_of_year, week_of_year, is_weekend",
        "days_since_ref: days since 2025-01-01 -- captures long-term market trend",
        "Cyclical sin/cos of day-of-year, day-of-week, week-of-year -- ensures Dec 31 and Jan 1 are treated as adjacent",
    ]),
    ("Market / Numeric Features", [
        "rate_per_mile = market_index / distance -- market pressure normalised by haul length",
        "mi_x_qs = market_index x quote_signal -- interaction: when both are high, rates spike",
        "dist_x_weight = distance x weight / 1M -- combined load effort (tonne-miles proxy)",
        "log_distance, log_weight -- compress wide-ranging values for better model splits",
    ]),
    ("Geographic Features", [
        "haversine_km -- true great-circle (crow-flies) distance between pickup and delivery",
        "dist_ratio = road distance / haversine -- captures route tortuosity",
    ]),
    ("Categorical Features", [
        "pickup, delivery, equipment -- native categories (LightGBM) or ordinal codes (Ridge/XGB)",
        "route = pickup + delivery -- specific lane identifier (each lane has its own rate level)",
        "weight_bin -- light / medium / heavy / very_heavy (structural pricing tiers)",
    ]),
]:
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(6, 74, 86)
    pdf.cell(0, 6, s(group), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for item in items:
        pdf.bullet(item, size=9)
    pdf.ln(1)
pdf.set_text_color(40, 40, 40)
pdf.body("Target: log(1 + posted_rate_capped)\nLog-transform converts the skewed target to near-normal. At prediction time, expm1() reverses the transform.")

# == SECTION 4 ==
pdf.section_title("4.  Training & Validation Approach")
pdf.body(
    "Method: 5-Fold Rolling Time-Series Cross-Validation\n\n"
    "A random split would allow training on October data and validating on January -- leaking future information. "
    "Instead, each fold trains on all past months and validates on the next unseen month:"
)
pdf.table_header(["Fold", "Training Data", "Validation Month"], [18, 112, 50])
for i, (f, tr, v) in enumerate([
    ("1", "January - May 2025",       "June 2025"),
    ("2", "January - June 2025",      "July 2025"),
    ("3", "January - July 2025",      "August 2025"),
    ("4", "January - August 2025",    "September 2025"),
    ("5", "January - September 2025", "October 2025"),
]):
    pdf.table_row([f, tr, v], [18, 112, 50], shade=(i%2==0))
pdf.ln(4)

# == SECTION 5 ==
pdf.section_title("5.  Model Selection & Comparison")
pdf.body("Three model families were compared across all 5 folds:")
pdf.table_header(["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean MAE"], [45,22,22,22,22,22,25])
for i, row in enumerate([
    ("Ridge Regression", "$156", "$111", "$100", "$140", "$117", "$124.68"),
    ("LightGBM",         "$179", "$123", "$127", "$138", "$127", "$138.94"),
    ("XGBoost",          "$202", "$131", "$137", "$135", "$136", "$148.34"),
]):
    pdf.table_row(row, [45,22,22,22,22,22,25], shade=(i%2==0), bold_last=True)
pdf.ln(3)

pdf.body(
    "Surprising finding: Ridge Regression outperformed both gradient boosting models.\n\n"
    "After log-transforming the target, the pricing relationship becomes nearly linear:\n"
    "   log(rate)  ~  log(distance) + market_index + equipment_type + ...\n\n"
    "Ridge's L2 regularisation prevents overfitting on the 10-month training window.\n\n"
    "Final model: Weighted Ensemble (Ridge wins CV so gets highest weight)\n"
    "   Final prediction  =  0.40 x Ridge  +  0.35 x LightGBM  +  0.25 x XGBoost\n\n"
    "Combining all three models reduces variance and gives better predictions than any single model alone."
)

# == SECTION 6 ==
pdf.section_title("6.  Final Validation Metrics  (Hold-Out: October 2025)")
for key, val, note, shade in [
    ("MAE  - Mean Absolute Error",   "$112.87",  "Average dollar error per load",       True),
    ("MAPE - Mean Abs % Error",      "5.20%",    "Percentage error on average",         False),
    ("RMSE - Root Mean Sq Error",    "$651.21",  "Penalises large errors more heavily", True),
]:
    if shade: pdf.set_fill_color(237, 245, 247)
    else:     pdf.set_fill_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(6, 74, 86)
    pdf.cell(72, 7, s(f"  {key}"), fill=True)
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(30, 30, 30)
    pdf.cell(22, 7, s(val), fill=True)
    pdf.set_font("Helvetica", "", 9);  pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, s(f"  {note}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
pdf.ln(2)
pdf.set_text_color(40, 40, 40)
pdf.body("The model predicts freight rates within 5.2% on average. On a load averaging $2,374, that is roughly $123 error per prediction.")

pdf.ln(1)
pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(6, 74, 86)
pdf.cell(0, 6, "Top 10 Most Important Features (LightGBM)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(40, 40, 40)
pdf.table_header(["Rank", "Feature", "Why It Matters"], [15, 48, 117])
for i, (rank, feat, why) in enumerate([
    ("1",  "distance",       "Primary cost driver -- longer haul = higher rate (r=0.91)"),
    ("2",  "quote_signal",   "Live market quoting signal -- highly predictive of current rates"),
    ("3",  "dist_x_weight",  "Combined load effort metric (distance x weight)"),
    ("4",  "equipment",      "Reefer > Dry Van > Flatbed structural pricing difference"),
    ("5",  "market_index",   "Overall market rate pressure index"),
    ("6",  "pickup",         "Origin city lane-level rate effect"),
    ("7",  "day_of_year",    "Seasonal effects -- Q4 is peak freight season"),
    ("8",  "route",          "Specific origin-destination lane rates"),
    ("9",  "delivery",       "Destination city lane-level rate effect"),
    ("10", "mi_x_qs",        "Interaction: high market_index AND high quote_signal"),
]):
    pdf.table_row([rank, feat, why], [15, 48, 117], shade=(i%2==0))
pdf.ln(4)

# == SECTION 7 ==
pdf.section_title("7.  December 2025 Fixed-Route Predictions")
pdf.body(
    "Route: Lexington, KY -> Fort Wayne, IN  |  360 miles  |  Dry Van  |  32,000 lb\n\n"
    "December 2025 is outside the training window so market_index and quote_signal are unknown. "
    "These were extrapolated using a linear trend fitted to Jan-Oct 2025 daily averages, plus "
    "a day-of-week offset to capture weekday vs weekend market patterns.\n\n"
    "Predicted rate band: $849 - $861, with mild daily variation driven by weekday/weekend patterns."
)
if os.path.exists(CHART):
    pdf.image(CHART, x=15, w=180)
else:
    pdf.body("[Chart not found -- run score.py to generate scorer_results/candidate_december.png]")

# == SECTION 8 ==
pdf.section_title("8.  Code & Submission Files")
pdf.table_header(["File", "Description"], [80, 100])
for i, (f, d) in enumerate([
    ("train_and_predict.py",          "Full pipeline: clean, engineer, CV, ensemble, predict"),
    ("validation_predictions.csv",    "12,000 final predictions (load_id, predicted_rate)"),
    ("december-chart-inputs.csv",     "31-day December predictions (filled)"),
    ("scorer_results/candidate_december.png", "December predicted rate chart"),
    ("score.py",                      "Provided scorer and validator script"),
    ("requirements.txt",              "Python package dependencies"),
]):
    pdf.table_row([f, d], [80, 100], shade=(i%2==0))

pdf.ln(6)
pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(120, 120, 120)
pdf.cell(0, 5, "GitHub: https://github.com/Ca200905/freight-rate-prediction", align="C")

pdf.output(OUT)
print(f"PDF saved: {OUT}")
