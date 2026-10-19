


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import joblib
import pandas as pd
import csv

# ============================================================
# STUDENT PERFORMANCE INTELLIGENCE SYSTEM
# ============================================================
# Uses the existing AML project model:
#   outputs/final_model.pkl
#
# IMPORTANT:
# The trained Random Forest uses ONLY these five ML features:
#   gender
#   race/ethnicity
#   parental level of education
#   lunch
#   test preparation course
#
# Marks, attendance, behaviour, presentation, etc. are used by
# the presentation/evaluation layer and are NOT passed to the
# existing model. This keeps the demo scientifically correct.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "final_model.pkl"
MODEL_RESULTS_PATH = BASE_DIR / "outputs" / "model_comparison.csv"

# -----------------------------
# Actual project results
# -----------------------------
DEFAULT_RESULTS = [
    ["Baseline Random Forest", 35.00, 37.06, 35.00, 35.49, 57.06],
    ["Random Forest + Information Gain + SMOTE", 32.00, 34.94, 32.00, 32.43, 57.72],
    ["XGBoost + Information Gain + SMOTE", 33.00, 33.88, 33.00, 32.24, 58.06],
    ["Optimized Random Forest", 33.50, 34.72, 33.50, 32.63, 58.14],
    ["Optimized XGBoost", 33.50, 34.72, 33.50, 32.63, 58.38],
    ["Stacking Ensemble", 35.50, 35.78, 35.50, 34.48, 58.44],
]

CLASS_COUNTS = {
    "Average": 103,
    "Best": 198,
    "Better": 403,
    "Good": 296,
}

# -----------------------------
# Load model safely
# -----------------------------
model = None
preprocessor = None
target_encoder = None
scaler = None
label_encoder = None
feature_columns = []
selected_features = []
selected_indices = []
model_name = "Random Forest"
MODEL_STATUS = ""
FEATURE_ALIASES = {
    "Attendance (%)": "Attendance",
    "Attendance": "Attendance",
    "Study Hours / Day": "Study Hours",
    "Study Hours": "Study Hours",
    "Assignment Performance (%)": "Assignment Performance",
    "Assignment Performance": "Assignment Performance",
    "Class Participation (%)": "Class Participation",
    "Class Participation": "Class Participation",
    "Previous Semester GPA": "Previous Semester GPA",
}

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact.get("model")
    scaler = artifact.get("scaler")
    label_encoder = artifact.get("label_encoder")
    preprocessor = artifact.get("preprocessor")
    target_encoder = artifact.get("target_encoder")
    feature_columns = artifact.get("feature_columns", [])
    selected_features = artifact.get("selected_features", [])
    selected_indices = artifact.get("selected_indices", list(range(len(feature_columns))))
    model_name = artifact.get("model_name", "Random Forest")
    MODEL_STATUS = "CONNECTED • final_model.pkl"
except Exception as exc:
    MODEL_STATUS = f"MODEL NOT LOADED • {exc}"


# -----------------------------
# Utility functions
# -----------------------------
def pct(v):
    return f"{float(v):.1f}%"


def grade_for_mark(mark):
    mark = float(mark)
    if mark >= 90:
        return "A+"
    if mark >= 80:
        return "A"
    if mark >= 70:
        return "B+"
    if mark >= 60:
        return "B"
    if mark >= 50:
        return "C"
    if mark >= 40:
        return "D"
    return "F"


def rating_score(value):
    return {
        "Excellent": 100,
        "Very Good": 85,
        "Good": 70,
        "Average": 55,
        "Needs Improvement": 35,
    }.get(value, 0)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def build_model_input(df):
    normalized = df.copy()
    if FEATURE_ALIASES:
        renamed = {old: new for old, new in FEATURE_ALIASES.items() if old in normalized.columns}
        if renamed:
            normalized = normalized.rename(columns=renamed)

    if scaler is not None and feature_columns:
        required = [col for col in feature_columns if col in normalized.columns]
        if len(required) == len(feature_columns):
            scaled = scaler.transform(normalized[feature_columns])
            if selected_indices:
                return scaled[:, selected_indices], label_encoder
            return scaled, label_encoder

    if preprocessor is not None:
        return preprocessor.transform(normalized), target_encoder

    raise ValueError(
        "The model artifact is missing its preprocessing pipeline. "
        "Please retrain the model and regenerate outputs/final_model.pkl."
    )


def analytical_profile(avg, attendance, behaviour, presentation, participation, assignment):
    # This is an application-level analytical profile, not a model output.
    scores = [
        avg,
        attendance,
        rating_score(behaviour),
        rating_score(presentation),
        rating_score(participation),
        rating_score(assignment),
    ]
    overall = sum(scores) / len(scores)

    if avg >= 80 and attendance >= 85 and overall >= 82:
        label = "CONSISTENT HIGH PERFORMER"
    elif avg >= 80 and (attendance < 75 or rating_score(participation) < 55):
        label = "ACADEMICALLY STRONG • DEVELOPMENT OPPORTUNITY"
    elif avg < 60 and attendance >= 80 and rating_score(assignment) >= 70:
        label = "HIGH POTENTIAL • ACADEMIC SUPPORT NEEDED"
    elif attendance < 70 or rating_score(assignment) < 55:
        label = "AT-RISK • CONSISTENCY SIGNAL"
    elif avg >= 70 and overall >= 70:
        label = "STEADY DEVELOPING PERFORMER"
    else:
        label = "DEVELOPING PERFORMANCE PROFILE"

    return label, overall


def performance_signal(avg, attendance, behaviour, presentation, participation, assignment):
    signals = []

    if avg >= 80:
        signals.append(("STRONG", "Academic performance is currently strong."))
    elif avg < 60:
        signals.append(("ATTENTION", "Academic marks indicate an area for improvement."))
    else:
        signals.append(("STABLE", "Academic performance is in a moderate range."))

    if attendance >= 85:
        signals.append(("STRONG", "Attendance shows good consistency."))
    elif attendance < 70:
        signals.append(("ATTENTION", "Attendance may affect learning consistency."))
    else:
        signals.append(("STABLE", "Attendance is moderate."))

    if rating_score(behaviour) >= 85:
        signals.append(("STRONG", "Classroom behaviour is rated highly."))
    elif rating_score(behaviour) <= 55:
        signals.append(("ATTENTION", "Classroom behaviour needs attention."))

    if rating_score(presentation) >= 85:
        signals.append(("STRONG", "Presentation skills are rated highly."))
    elif rating_score(presentation) <= 55:
        signals.append(("DEVELOP", "Presentation skills have room for development."))

    if rating_score(participation) <= 55:
        signals.append(("DEVELOP", "Class participation can be strengthened."))

    if rating_score(assignment) >= 85:
        signals.append(("STRONG", "Assignment completion/performance is strong."))

    return signals[:7]


# ============================================================
# Main application
# ============================================================

root = tk.Tk()
root.title("Student Performance Intelligence System")
root.geometry("1380x850")
root.minsize(1180, 720)
root.configure(bg="#0E1117")

# -----------------------------
# Theme
# -----------------------------
BG = "#0E1117"
PANEL = "#171B24"
PANEL2 = "#1E2430"
TEXT = "#F4F7FB"
MUTED = "#9BA7B7"
ACCENT = "#E8B04B"
ACCENT2 = "#7FD1B9"
DANGER = "#E57373"
GOOD = "#7FD1B9"
WHITE = "#FFFFFF"

style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass

style.configure(
    "TNotebook",
    background=BG,
    borderwidth=0
)
style.configure(
    "TNotebook.Tab",
    background=PANEL,
    foreground=MUTED,
    padding=(18, 10),
    font=("Segoe UI", 10, "bold")
)
style.map(
    "TNotebook.Tab",
    background=[("selected", PANEL2)],
    foreground=[("selected", ACCENT)]
)
style.configure(
    "TCombobox",
    fieldbackground=PANEL2,
    background=PANEL2,
    foreground=TEXT,
    arrowcolor=ACCENT,
    bordercolor=PANEL2,
    padding=8,
    font=("Segoe UI", 10)
)
style.map(
    "TCombobox",
    fieldbackground=[("readonly", PANEL2)],
    foreground=[("readonly", TEXT)]
)
style.configure(
    "TButton",
    background=PANEL2,
    foreground=TEXT,
    padding=(14, 9),
    font=("Segoe UI", 10, "bold"),
    borderwidth=0
)
style.map(
    "TButton",
    background=[("active", "#2A3140")],
    foreground=[("active", WHITE)]
)

# -----------------------------
# Variables
# -----------------------------
name_var = tk.StringVar(value="Swar Sawalkar")
roll_var = tk.StringVar(value="23104B0023")

# gender_var = tk.StringVar()
# race_var = tk.StringVar()
# education_var = tk.StringVar()
# lunch_var = tk.StringVar()
# prep_var = tk.StringVar()

# math_var = tk.StringVar(value="78")
# reading_var = tk.StringVar(value="82")
# writing_var = tk.StringVar(value="75")
# attendance_var = tk.StringVar(value="92")

math_var = tk.StringVar(value="78")
reading_var = tk.StringVar(value="82")
writing_var = tk.StringVar(value="75")

# ML Predictive Features
attendance_var = tk.StringVar(value="92")
study_hours_var = tk.StringVar(value="4")
assignment_performance_var = tk.StringVar(value="85")
participation_var = tk.StringVar(value="85")
previous_gpa_var = tk.StringVar(value="8.5")

behaviour_var = tk.StringVar(value="Excellent")
presentation_var = tk.StringVar(value="Excellent")
analytical_participation_var = tk.StringVar(value="Very Good")
assignment_var = tk.StringVar(value="Excellent")
communication_var = tk.StringVar(value="Very Good")
discipline_var = tk.StringVar(value="Excellent")

prediction_var = tk.StringVar(value="—")
confidence_var = tk.StringVar(value="—")
profile_var = tk.StringVar(value="—")
profile_score_var = tk.StringVar(value="—")

prob_vars = {
    "Average": tk.StringVar(value="—"),
    "Best": tk.StringVar(value="—"),
    "Better": tk.StringVar(value="—"),
    "Good": tk.StringVar(value="—"),
}

last_prediction = {"label": None, "confidence": None}


# ============================================================
# Reusable UI helpers
# ============================================================

def card(parent, title=None):
    frame = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#252C38")
    if title:
        tk.Label(
            frame,
            text=title,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=18, pady=(15, 8))
    return frame


def label(parent, text, size=10, color=MUTED, bold=False):
    return tk.Label(
        parent,
        text=text,
        bg=parent.cget("bg"),
        fg=color,
        font=("Segoe UI", size, "bold" if bold else "normal")
    )


def entry(parent, variable, width=22):
    e = tk.Entry(
        parent,
        textvariable=variable,
        width=width,
        bg=PANEL2,
        fg=TEXT,
        insertbackground=TEXT,
        relief="flat",
        font=("Segoe UI", 10)
    )
    e.configure(highlightthickness=1, highlightbackground="#303846")
    return e


def combo(parent, variable, values, width=25):
    c = ttk.Combobox(
        parent,
        textvariable=variable,
        values=values,
        state="readonly",
        width=width
    )
    return c


def clear_children(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def set_status(text):
    status_var.set(text)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=25, pady=(18, 8))

tk.Label(
    header,
    text="STUDENT PERFORMANCE",
    bg=BG,
    fg=ACCENT,
    font=("Segoe UI", 12, "bold")
).pack(anchor="w")

tk.Label(
    header,
    text="INTELLIGENCE SYSTEM",
    bg=BG,
    fg=TEXT,
    font=("Segoe UI", 25, "bold")
).pack(anchor="w")

tk.Label(
    header,
    text="Advanced Machine Learning • Performance Profiling • What-If Analysis",
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 10)
).pack(anchor="w")

status_var = tk.StringVar(value=MODEL_STATUS)
tk.Label(
    header,
    textvariable=status_var,
    bg=BG,
    fg=ACCENT2 if model is not None else DANGER,
    font=("Segoe UI", 9, "bold")
).pack(anchor="e")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=22, pady=10)


# ============================================================
# TAB 1: DASHBOARD
# ============================================================

dashboard = tk.Frame(notebook, bg=BG)
notebook.add(dashboard, text="  DASHBOARD  ")

dash_top = tk.Frame(dashboard, bg=BG)
dash_top.pack(fill="x", padx=10, pady=10)

def metric_card(parent, title, value, subtitle):
    f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#252C38")
    f.pack(side="left", fill="both", expand=True, padx=6)
    tk.Label(f, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(
        anchor="w", padx=16, pady=(14, 4)
    )
    tk.Label(f, text=value, bg=PANEL, fg=TEXT, font=("Segoe UI", 20, "bold")).pack(
        anchor="w", padx=16
    )
    tk.Label(f, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(
        anchor="w", padx=16, pady=(2, 14)
    )
    return f

metric_card(dash_top, "DATASET", "1,000", "student records")
metric_card(dash_top, "CLASSES", "4", "Average • Good • Better • Best")
metric_card(dash_top, "ML FEATURES", "5", "predictive input features")
metric_card(dash_top, "SELECTED MODEL", "Random Forest", "highest weighted F1")

dash_mid = tk.Frame(dashboard, bg=BG)
dash_mid.pack(fill="both", expand=True, padx=10, pady=8)

overview_card = card(dash_mid, "PROJECT PIPELINE")
overview_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

pipeline = [
    ("01", "Student Profile", "Collect student and evaluation information"),
    ("02", "Preprocessing", "Encode the five ML input features"),
    ("03", "Feature Selection", "Mutual Information / Information Gain"),
    ("04", "Class Balancing", "SMOTE on training data"),
    ("05", "Model Comparison", "Random Forest • XGBoost • Ensemble"),
    ("06", "Prediction", "Performance class + probability"),
    ("07", "Intelligence Layer", "Profile • Signals • What-If"),
]

for num, title, desc in pipeline:
    row = tk.Frame(overview_card, bg=PANEL)
    row.pack(fill="x", padx=16, pady=5)
    tk.Label(
        row, text=num, bg=ACCENT, fg="#111111",
        font=("Segoe UI", 9, "bold"), width=4
    ).pack(side="left", padx=(0, 10))
    tx = tk.Frame(row, bg=PANEL)
    tx.pack(side="left", fill="x", expand=True)
    tk.Label(tx, text=title, bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w")
    tk.Label(tx, text=desc, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")

dist_card = card(dash_mid, "TARGET DISTRIBUTION")
dist_card.pack(side="left", fill="both", expand=True, padx=(6, 0))

max_count = max(CLASS_COUNTS.values())
for cls in ["Average", "Best", "Better", "Good"]:
    row = tk.Frame(dist_card, bg=PANEL)
    row.pack(fill="x", padx=18, pady=9)
    tk.Label(row, text=cls, bg=PANEL, fg=TEXT, width=10, anchor="w",
             font=("Segoe UI", 10, "bold")).pack(side="left")
    canvas = tk.Canvas(row, bg=PANEL, height=20, highlightthickness=0)
    canvas.pack(side="left", fill="x", expand=True, padx=8)
    canvas.update_idletasks()
    w = max(20, canvas.winfo_width())
    ratio = CLASS_COUNTS[cls] / max_count
    canvas.create_rectangle(0, 3, max(20, w * ratio), 17, fill=ACCENT, outline="")
    tk.Label(row, text=str(CLASS_COUNTS[cls]), bg=PANEL, fg=MUTED,
             width=6, anchor="e").pack(side="right")


# ============================================================
# TAB 2: STUDENT PROFILE
# ============================================================

profile_tab = tk.Frame(notebook, bg=BG)
notebook.add(profile_tab, text="  STUDENT PROFILE  ")

profile_canvas = tk.Canvas(profile_tab, bg=BG, highlightthickness=0)
profile_scroll = ttk.Scrollbar(profile_tab, orient="vertical", command=profile_canvas.yview)
profile_body = tk.Frame(profile_canvas, bg=BG)

profile_body.bind(
    "<Configure>",
    lambda e: profile_canvas.configure(scrollregion=profile_canvas.bbox("all"))
)
profile_canvas.create_window((0, 0), window=profile_body, anchor="nw")
profile_canvas.configure(yscrollcommand=profile_scroll.set)

profile_canvas.pack(side="left", fill="both", expand=True)
profile_scroll.pack(side="right", fill="y")


def form_row(parent, row, title, variable, values=None, width=25):
    # Use pack() consistently because the card itself is managed with pack().
    # Mixing grid() and pack() in the same parent causes TclError.
    row_frame = tk.Frame(parent, bg=PANEL)
    row_frame.pack(fill="x", padx=16, pady=6)

    tk.Label(
        row_frame,
        text=title,
        bg=PANEL,
        fg=MUTED,
        font=("Segoe UI", 9, "bold"),
        width=27,
        anchor="w"
    ).pack(side="left")

    if values is None:
        w = entry(row_frame, variable, width)
    else:
        w = combo(row_frame, variable, values, width)

    w.pack(side="left", fill="x", expand=True, padx=(10, 0))
    return w


student_card = card(profile_body, "STUDENT IDENTITY")
student_card.pack(fill="x", padx=12, pady=10)

form_row(student_card, 0, "Student Name", name_var)
form_row(student_card, 1, "Roll Number", roll_var)

# ml_card = card(profile_body, "PREDICTIVE PROFILE — USED BY THE TRAINED MODEL")
# ml_card.pack(fill="x", padx=12, pady=10)

# form_row(ml_card, 0, "Gender", gender_var, ["female", "male"])
# form_row(ml_card, 1, "Race / Ethnicity", race_var, ["group A", "group B", "group C", "group D", "group E"])
# form_row(
#     ml_card, 2, "Parental Education", education_var,
#     ["associate's degree", "bachelor's degree", "high school",
#      "master's degree", "some college", "some high school"]
# )
# form_row(ml_card, 3, "Lunch", lunch_var, ["standard", "free/reduced"])
# form_row(ml_card, 4, "Test Preparation", prep_var, ["completed", "none"])

ml_card = card(profile_body, "PREDICTIVE PROFILE — USED BY THE TRAINED MODEL")
ml_card.pack(fill="x", padx=12, pady=10)

form_row(ml_card, 0, "Attendance (%)", attendance_var)
form_row(ml_card, 1, "Study Hours / Day", study_hours_var)
form_row(ml_card, 2, "Assignment Performance (%)", assignment_performance_var)
form_row(ml_card, 3, "Class Participation (%)", participation_var)
form_row(ml_card, 4, "Previous Semester GPA", previous_gpa_var)

eval_card = card(profile_body, "STUDENT DEVELOPMENT PROFILE — ANALYTICAL LAYER")
eval_card.pack(fill="x", padx=12, pady=10)

form_row(eval_card, 0, "Attendance (%)", attendance_var)
form_row(eval_card, 1, "Behaviour", behaviour_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])
form_row(eval_card, 2, "Presentation", presentation_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])
form_row(eval_card, 3, "Participation", analytical_participation_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])
form_row(eval_card, 4, "Assignment Performance", assignment_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])
form_row(eval_card, 5, "Communication", communication_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])
form_row(eval_card, 6, "Discipline", discipline_var,
         ["Excellent", "Very Good", "Good", "Average", "Needs Improvement"])

marks_card = card(profile_body, "ACADEMIC RESULT")
marks_card.pack(fill="x", padx=12, pady=10)

form_row(marks_card, 0, "Mathematics", math_var)
form_row(marks_card, 1, "Reading", reading_var)
form_row(marks_card, 2, "Writing", writing_var)


tk.Label(
    profile_body,
    text="Note: The trained Random Forest uses Attendance, Study Hours, "
         "Assignment Performance, Class Participation and Previous Semester GPA "
         "as its predictive features.",
    bg=BG, fg=ACCENT, font=("Segoe UI", 9, "italic")
).pack(anchor="w", padx=18, pady=(0, 15))

# ============================================================
# TAB 3: PERFORMANCE INTELLIGENCE
# ============================================================

intel_tab = tk.Frame(notebook, bg=BG)
notebook.add(intel_tab, text="  PERFORMANCE INTELLIGENCE  ")

intel_top = tk.Frame(intel_tab, bg=BG)
intel_top.pack(fill="x", padx=12, pady=12)

# Academic result card
result_card = card(intel_top, "ACADEMIC RESULT")
result_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

marks_labels = {}
for i, (subj, var) in enumerate([
    ("Mathematics", math_var),
    ("Reading", reading_var),
    ("Writing", writing_var),
]):
    row = tk.Frame(result_card, bg=PANEL)
    row.pack(fill="x", padx=15, pady=5)
    tk.Label(row, text=subj, bg=PANEL, fg=MUTED, width=15, anchor="w").pack(side="left")
    mark_lbl = tk.Label(row, textvariable=var, bg=PANEL, fg=TEXT,
                        font=("Segoe UI", 11, "bold"), width=8)
    mark_lbl.pack(side="left")
    grade_var = tk.StringVar(value="—")
    marks_labels[subj] = grade_var
    tk.Label(row, textvariable=grade_var, bg=PANEL, fg=ACCENT,
             font=("Segoe UI", 10, "bold")).pack(side="left")

avg_var = tk.StringVar(value="—")
row = tk.Frame(result_card, bg=PANEL)
row.pack(fill="x", padx=15, pady=(12, 15))
tk.Label(row, text="Overall Average", bg=PANEL, fg=TEXT,
         font=("Segoe UI", 10, "bold"), width=15, anchor="w").pack(side="left")
tk.Label(row, textvariable=avg_var, bg=PANEL, fg=ACCENT,
         font=("Segoe UI", 17, "bold")).pack(side="left")

# Profile card
profile_result_card = card(intel_top, "PERFORMANCE FINGERPRINT")
profile_result_card.pack(side="left", fill="both", expand=True, padx=6)

fingerprint_canvas = tk.Canvas(profile_result_card, bg=PANEL, height=180, highlightthickness=0)
fingerprint_canvas.pack(fill="both", expand=True, padx=12, pady=5)

# ML result card
ml_result_card = card(intel_top, "ML PREDICTION")
ml_result_card.pack(side="left", fill="both", expand=True, padx=(6, 0))

tk.Label(ml_result_card, textvariable=prediction_var, bg=PANEL, fg=ACCENT,
         font=("Segoe UI", 24, "bold")).pack(pady=(20, 3))
tk.Label(ml_result_card, text="PREDICTED PERFORMANCE", bg=PANEL, fg=MUTED,
         font=("Segoe UI", 9, "bold")).pack()

tk.Label(ml_result_card, text="Confidence", bg=PANEL, fg=MUTED,
         font=("Segoe UI", 9)).pack(pady=(18, 0))
tk.Label(ml_result_card, textvariable=confidence_var, bg=PANEL, fg=TEXT,
         font=("Segoe UI", 17, "bold")).pack()

profile_line = tk.Label(ml_result_card, textvariable=profile_var, bg=PANEL,
                        fg=ACCENT2, font=("Segoe UI", 9, "bold"), wraplength=250)
profile_line.pack(padx=15, pady=15)

intel_bottom = tk.Frame(intel_tab, bg=BG)
intel_bottom.pack(fill="both", expand=True, padx=12, pady=5)

signal_card = card(intel_bottom, "PERFORMANCE SIGNALS")
signal_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

signal_text = tk.Text(
    signal_card, bg=PANEL, fg=TEXT, relief="flat",
    font=("Segoe UI", 10), height=10, wrap="word"
)
signal_text.pack(fill="both", expand=True, padx=15, pady=10)
signal_text.configure(state="disabled")

prob_card = card(intel_bottom, "MODEL PROBABILITY PROFILE")
prob_card.pack(side="left", fill="both", expand=True, padx=6)

for cls in ["Average", "Best", "Better", "Good"]:
    row = tk.Frame(prob_card, bg=PANEL)
    row.pack(fill="x", padx=15, pady=7)
    tk.Label(row, text=cls, bg=PANEL, fg=TEXT, width=10,
             anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")
    bar = tk.Canvas(row, bg=PANEL, height=18, highlightthickness=0)
    bar.pack(side="left", fill="x", expand=True, padx=8)
    tk.Label(row, textvariable=prob_vars[cls], bg=PANEL, fg=MUTED,
             width=8).pack(side="right")

actions_card = card(intel_bottom, "ACTIONS")
actions_card.pack(side="left", fill="both", expand=True, padx=(6, 0))

tk.Button(
    actions_card, text="RUN FULL ANALYSIS",
    command=lambda: run_full_analysis(),
    bg=ACCENT, fg="#111111", activebackground="#F2C76B",
    relief="flat", font=("Segoe UI", 11, "bold"),
    padx=15, pady=12
).pack(fill="x", padx=18, pady=(20, 8))

tk.Button(
    actions_card, text="OPEN WHAT-IF SIMULATOR",
    command=lambda: notebook.select(whatif_tab),
    bg=PANEL2, fg=TEXT, activebackground="#2A3140",
    relief="flat", font=("Segoe UI", 10, "bold"),
    padx=15, pady=10
).pack(fill="x", padx=18, pady=8)

tk.Button(
    actions_card, text="GENERATE REPORT",
    command=lambda: generate_report(),
    bg=PANEL2, fg=TEXT, activebackground="#2A3140",
    relief="flat", font=("Segoe UI", 10, "bold"),
    padx=15, pady=10
).pack(fill="x", padx=18, pady=8)


# ============================================================
# TAB 4: WHAT-IF SIMULATOR
# ============================================================

whatif_tab = tk.Frame(notebook, bg=BG)
notebook.add(whatif_tab, text="  WHAT-IF SIMULATOR  ")

tk.Label(
    whatif_tab,
    text="WHAT-IF PERFORMANCE SIMULATOR",
    bg=BG, fg=TEXT, font=("Segoe UI", 20, "bold")
).pack(anchor="w", padx=18, pady=(18, 2))

tk.Label(
    whatif_tab,
    text="Change one of the actual ML input features and rerun the trained model.",
    bg=BG, fg=MUTED, font=("Segoe UI", 10)
).pack(anchor="w", padx=18, pady=(0, 15))

wi_main = tk.Frame(whatif_tab, bg=BG)
wi_main.pack(fill="both", expand=True, padx=18)

wi_left = card(wi_main, "BASELINE")
wi_left.pack(side="left", fill="both", expand=True, padx=(0, 8))

wi_right = card(wi_main, "SIMULATED PROFILE")
wi_right.pack(side="left", fill="both", expand=True, padx=8)

wi_control = card(wi_main, "COMPARE")
wi_control.pack(side="left", fill="both", expand=True, padx=(8, 0))

# whatif_vars = {
#     "gender": tk.StringVar(),
#     "race/ethnicity": tk.StringVar(),
#     "parental level of education": tk.StringVar(),
#     "lunch": tk.StringVar(),
#     "test preparation course": tk.StringVar(),
# }

whatif_vars = {
    "Attendance (%)": tk.StringVar(),
    "Study Hours / Day": tk.StringVar(),
    "Assignment Performance (%)": tk.StringVar(),
    "Class Participation (%)": tk.StringVar(),
    "Previous Semester GPA": tk.StringVar(),
}

# field_values = {
#     "gender": ["female", "male"],
#     "race/ethnicity": ["group A", "group B", "group C", "group D", "group E"],
#     "parental level of education": [
#         "associate's degree", "bachelor's degree", "high school",
#         "master's degree", "some college", "some high school"
#     ],
#     "lunch": ["standard", "free/reduced"],
#     "test preparation course": ["completed", "none"],
# }

field_values = {
    "Attendance (%)": ["60", "70", "75", "80", "85", "90", "95", "100"],
    "Study Hours / Day": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "Assignment Performance (%)": ["50", "60", "70", "75", "80", "85", "90", "95", "100"],
    "Class Participation (%)": ["50", "60", "70", "75", "80", "85", "90", "95", "100"],
    "Previous Semester GPA": ["5.0", "5.5", "6.0", "6.5", "7.0", "7.5", "8.0", "8.5", "9.0", "9.5", "10.0"],
}

baseline_result_var = tk.StringVar(value="—")
baseline_conf_var = tk.StringVar(value="—")
sim_result_var = tk.StringVar(value="—")
sim_conf_var = tk.StringVar(value="—")
change_var = tk.StringVar(value="Select a profile change and click SIMULATE")

for i, key in enumerate(field_values):
    tk.Label(wi_left, text=key.title(), bg=PANEL, fg=MUTED,
             font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=15, pady=(8, 2))
    c = combo(wi_left, whatif_vars[key], field_values[key], 28)
    c.pack(fill="x", padx=15, pady=(0, 5))

tk.Label(wi_left, text="Baseline prediction", bg=PANEL, fg=MUTED).pack(pady=(15, 2))
tk.Label(wi_left, textvariable=baseline_result_var, bg=PANEL, fg=ACCENT,
         font=("Segoe UI", 20, "bold")).pack()
tk.Label(wi_left, textvariable=baseline_conf_var, bg=PANEL, fg=TEXT,
         font=("Segoe UI", 11, "bold")).pack(pady=(2, 10))

# Simulated controls
# sim_change_var = tk.StringVar(value="test preparation course")
# sim_value_var = tk.StringVar(value="completed")

sim_change_var = tk.StringVar(value="Attendance (%)")
sim_value_var = tk.StringVar(value="95")

tk.Label(wi_control, text="Change feature", bg=PANEL, fg=MUTED,
         font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=15, pady=(12, 3))
combo(
    wi_control, sim_change_var,
    list(field_values.keys()), 28
).pack(fill="x", padx=15)

# sim_value_combo = combo(
#     wi_control, sim_value_var,
#     field_values["test preparation course"], 28
# )
sim_value_combo = combo(
    wi_control, sim_value_var,
    field_values["Attendance (%)"], 28
)
sim_value_combo.pack(fill="x", padx=15, pady=10)

def update_sim_values(event=None):
    vals = field_values[sim_change_var.get()]
    sim_value_combo["values"] = vals
    sim_value_var.set(vals[0])

sim_change_var.trace_add("write", lambda *_: update_sim_values())

def simulate():
    if model is None:
        messagebox.showerror("Model Error", "The trained model could not be loaded.")
        return

    base = {
        "Attendance (%)": safe_float(attendance_var.get()),
        "Study Hours / Day": safe_float(study_hours_var.get()),
        "Assignment Performance (%)": safe_float(assignment_performance_var.get()),
        "Class Participation (%)": safe_float(participation_var.get()),
        "Previous Semester GPA": safe_float(previous_gpa_var.get()),
    }

    if any(v == "" for v in base.values()):
        messagebox.showwarning(
            "Missing ML Inputs",
            "Complete the five Predictive Profile fields in Student Profile first."
        )
        notebook.select(profile_tab)
        return

    for key in base:
        whatif_vars[key].set(base[key])

    base_df = pd.DataFrame([base])
    base_x, base_encoder = build_model_input(base_df)
    base_pred = model.predict(base_x)[0]
    base_label = str(base_encoder.inverse_transform([base_pred])[0])

    if hasattr(model, "predict_proba"):
        base_probs = model.predict_proba(base_x)[0]
        base_conf = max(base_probs) * 100
    else:
        base_conf = None

    changed = dict(base)
    changed[sim_change_var.get()] = sim_value_var.get()
    sim_df = pd.DataFrame([changed])
    sim_x, sim_encoder = build_model_input(sim_df)
    sim_pred = model.predict(sim_x)[0]
    sim_label = str(sim_encoder.inverse_transform([sim_pred])[0])

    if hasattr(model, "predict_proba"):
        sim_probs = model.predict_proba(sim_x)[0]
        sim_conf = max(sim_probs) * 100
    else:
        sim_conf = None

    baseline_result_var.set(base_label.upper())
    baseline_conf_var.set(f"Confidence: {base_conf:.2f}%" if base_conf is not None else "Confidence: N/A")
    sim_result_var.set(sim_label.upper())
    sim_conf_var.set(f"Confidence: {sim_conf:.2f}%" if sim_conf is not None else "Confidence: N/A")

    if base_label == sim_label:
        change_var.set(
            f"No class change: {base_label.upper()} → {sim_label.upper()} "
            f"after changing {sim_change_var.get()}."
        )
    else:
        change_var.set(
            f"Prediction changed: {base_label.upper()} → {sim_label.upper()} "
            f"after changing {sim_change_var.get()} from '{base[sim_change_var.get()]}' "
            f"to '{sim_value_var.get()}'."
        )

tk.Button(
    wi_control, text="SIMULATE CHANGE",
    command=simulate,
    bg=ACCENT, fg="#111111", activebackground="#F2C76B",
    relief="flat", font=("Segoe UI", 11, "bold"),
    padx=12, pady=12
).pack(fill="x", padx=15, pady=15)

tk.Label(
    wi_control, textvariable=change_var,
    bg=PANEL, fg=ACCENT2,
    font=("Segoe UI", 10, "bold"),
    wraplength=280, justify="left"
).pack(anchor="w", padx=15, pady=10)

tk.Label(
    wi_right, textvariable=sim_result_var,
    bg=PANEL, fg=ACCENT,
    font=("Segoe UI", 25, "bold")
).pack(pady=(35, 4))
tk.Label(
    wi_right, text="SIMULATED PREDICTION",
    bg=PANEL, fg=MUTED,
    font=("Segoe UI", 9, "bold")
).pack()
tk.Label(
    wi_right, textvariable=sim_conf_var,
    bg=PANEL, fg=TEXT,
    font=("Segoe UI", 12, "bold")
).pack(pady=12)

tk.Label(
    wi_right,
    text="This experiment changes a real model input and reruns the existing trained model.",
    bg=PANEL, fg=MUTED,
    font=("Segoe UI", 9), wraplength=280, justify="center"
).pack(padx=20, pady=25)


# ============================================================
# TAB 5: MODEL LAB
# ============================================================

model_tab = tk.Frame(notebook, bg=BG)
notebook.add(model_tab, text="  MODEL LAB  ")

tk.Label(
    model_tab,
    text="MODEL LAB",
    bg=BG, fg=TEXT, font=("Segoe UI", 20, "bold")
).pack(anchor="w", padx=18, pady=(18, 2))

tk.Label(
    model_tab,
    text="Measured results from the AML experiment",
    bg=BG, fg=MUTED, font=("Segoe UI", 10)
).pack(anchor="w", padx=18, pady=(0, 15))

lab_top = tk.Frame(model_tab, bg=BG)
lab_top.pack(fill="x", padx=18)

for title, value, sub in [
    ("Best Weighted F1", "35.49%", "Baseline Random Forest"),
    ("Highest Accuracy", "35.50%", "Stacking Ensemble"),
    ("Highest ROC-AUC", "58.44%", "Stacking Ensemble"),
]:
    metric_card(lab_top, title, value, sub)

table_frame = tk.Frame(model_tab, bg=PANEL)
table_frame.pack(fill="both", expand=True, padx=18, pady=15)

columns = ("Model", "Accuracy", "Precision", "Recall", "Weighted F1", "ROC-AUC")
tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=150 if col != "Model" else 360)

tree.pack(fill="both", expand=True, padx=12, pady=12)

results = DEFAULT_RESULTS

if MODEL_RESULTS_PATH.exists():
    try:
        loaded = pd.read_csv(MODEL_RESULTS_PATH)
        # Use the file only when expected columns exist.
        expected = {"Model", "Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "ROC-AUC"}
        if expected.issubset(set(loaded.columns)):
            results = [
                [
                    r["Model"],
                    float(r["Accuracy"]) * 100,
                    float(r["Weighted Precision"]) * 100,
                    float(r["Weighted Recall"]) * 100,
                    float(r["Weighted F1"]) * 100,
                    float(r["ROC-AUC"]) * 100,
                ]
                for _, r in loaded.iterrows()
            ]
    except Exception:
        pass

for r in results:
    tree.insert("", "end", values=(
        r[0], f"{r[1]:.2f}%", f"{r[2]:.2f}%", f"{r[3]:.2f}%",
        f"{r[4]:.2f}%", f"{r[5]:.2f}%"
    ))

note = tk.Frame(model_tab, bg=PANEL)
note.pack(fill="x", padx=18, pady=(0, 15))

tk.Label(
    note,
    text="MODEL SELECTION",
    bg=PANEL, fg=ACCENT,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", padx=15, pady=(12, 2))

tk.Label(
    note,
    text="Baseline Random Forest is the selected model because it has the highest Weighted F1 (35.49%). "
         "The Stacking Ensemble has the highest Accuracy (35.50%) and ROC-AUC (58.44%), "
         "but a lower Weighted F1 (34.48%).",
    bg=PANEL, fg=TEXT, font=("Segoe UI", 9),
    wraplength=1200, justify="left"
).pack(anchor="w", padx=15, pady=(0, 12))


# ============================================================
# TAB 6: REPORT
# ============================================================

report_tab = tk.Frame(notebook, bg=BG)
notebook.add(report_tab, text="  REPORT  ")

report_text = tk.Text(
    report_tab,
    bg=PANEL, fg=TEXT, insertbackground=TEXT,
    relief="flat", font=("Consolas", 10),
    wrap="word"
)
report_text.pack(fill="both", expand=True, padx=18, pady=18)


# ============================================================
# Fingerprint drawing
# ============================================================

def draw_fingerprint():
    fingerprint_canvas.delete("all")

    avg = (safe_float(math_var.get()) + safe_float(reading_var.get()) + safe_float(writing_var.get())) / 3
    vals = [
        avg,
        safe_float(attendance_var.get()),
        rating_score(presentation_var.get()),
        safe_float(participation_var.get()),
        safe_float(assignment_performance_var.get()),
        rating_score(behaviour_var.get()),
    ]
    labels = ["Academic", "Attendance", "Presentation", "Participation", "Assignment", "Behaviour"]

    w = max(400, fingerprint_canvas.winfo_width())
    center_x = w / 2
    center_y = 90
    radius = min(75, w / 5)

    # Hexagon-ish radial chart
    import math as _math
    points_outer = []
    points_inner = []

    for i in range(6):
        angle = -_math.pi / 2 + i * 2 * _math.pi / 6
        points_outer.extend([
            center_x + radius * _math.cos(angle),
            center_y + radius * _math.sin(angle)
        ])
        rr = radius * (vals[i] / 100)
        points_inner.extend([
            center_x + rr * _math.cos(angle),
            center_y + rr * _math.sin(angle)
        ])

    for scale in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for i in range(6):
            angle = -_math.pi / 2 + i * 2 * _math.pi / 6
            pts.extend([
                center_x + radius * scale * _math.cos(angle),
                center_y + radius * scale * _math.sin(angle)
            ])
        fingerprint_canvas.create_polygon(
            pts, outline="#303846", fill="", width=1
        )

    fingerprint_canvas.create_polygon(
        points_outer, outline="#4A5362", fill="", width=1
    )

    fingerprint_canvas.create_polygon(
        points_inner,
        outline=ACCENT,
        fill="#5B4520",
        stipple="gray50",
        width=2
    )

    for i in range(6):
        angle = -_math.pi / 2 + i * 2 * _math.pi / 6
        x1 = center_x + radius * _math.cos(angle)
        y1 = center_y + radius * _math.sin(angle)
        fingerprint_canvas.create_line(center_x, center_y, x1, y1, fill="#303846")

        lx = center_x + (radius + 28) * _math.cos(angle)
        ly = center_y + (radius + 28) * _math.sin(angle)
        anchor = "center"
        fingerprint_canvas.create_text(
            lx, ly,
            text=f"{labels[i]}\n{vals[i]:.0f}%",
            fill=TEXT, font=("Segoe UI", 8, "bold"),
            anchor=anchor
        )


# ============================================================
# Main analysis
# ============================================================

def run_full_analysis():
    try:
        math_mark = safe_float(math_var.get())
        reading_mark = safe_float(reading_var.get())
        writing_mark = safe_float(writing_var.get())
        attendance = safe_float(attendance_var.get())

        if not all(0 <= x <= 100 for x in [math_mark, reading_mark, writing_mark, attendance]):
            raise ValueError("Marks and attendance must be between 0 and 100.")

        avg = (math_mark + reading_mark + writing_mark) / 3
        avg_var.set(f"{avg:.2f}")
        marks_labels["Mathematics"].set(grade_for_mark(math_mark))
        marks_labels["Reading"].set(grade_for_mark(reading_mark))
        marks_labels["Writing"].set(grade_for_mark(writing_mark))

        profile, score = analytical_profile(
            avg,
            attendance,
            behaviour_var.get(),
            presentation_var.get(),
            analytical_participation_var.get(),
            assignment_var.get()
        )
        profile_var.set(profile)
        profile_score_var.set(f"{score:.1f}/100")

        signals = performance_signal(
            avg,
            attendance,
            behaviour_var.get(),
            presentation_var.get(),
            analytical_participation_var.get(),
            assignment_var.get()
        )

        signal_text.configure(state="normal")
        signal_text.delete("1.0", "end")
        for level, message in signals:
            prefix = "●"
            signal_text.insert("end", f"{prefix} {level}: {message}\n\n")
        signal_text.configure(state="disabled")

        draw_fingerprint()

        if model is None:
            prediction_var.set("MODEL ERROR")
            confidence_var.set("—")
            set_status("MODEL NOT LOADED")
            messagebox.showerror(
                "Model Error",
                f"Could not load:\n{MODEL_PATH}\n\n"
                "Make sure this GUI is inside the AML-Project folder and "
                "outputs/final_model.pkl exists."
            )
            return

        student = {
            "Attendance (%)": safe_float(attendance_var.get()),
            "Study Hours / Day": safe_float(study_hours_var.get()),
            "Assignment Performance (%)": safe_float(assignment_performance_var.get()),
            "Class Participation (%)": safe_float(participation_var.get()),
            "Previous Semester GPA": safe_float(previous_gpa_var.get()),
        }

        if any(v == "" for v in student.values()):
            prediction_var.set("ENTER ML INPUTS")
            confidence_var.set("—")
            set_status("Complete Predictive Profile fields")
            messagebox.showwarning(
                "ML Inputs Required",
                "Please complete Attendance, Study Hours, Assignment Performance, "
                "Class Participation and Previous Semester GPA."
            )
            notebook.select(profile_tab)
            return

        df = pd.DataFrame([student])
        X, encoder = build_model_input(df)

        pred = model.predict(X)[0]
        pred_label = str(encoder.inverse_transform([pred])[0])

        prediction_var.set(pred_label.upper())

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities) * 100
            confidence_var.set(f"{confidence:.2f}%")
            last_prediction["confidence"] = confidence

            classes = list(encoder.classes_)
            for i, cls in enumerate(classes):
                if i < len(probabilities):
                    prob_vars[str(cls)].set(f"{probabilities[i] * 100:.2f}%")
        else:
            confidence_var.set("N/A")
            last_prediction["confidence"] = None

        last_prediction["label"] = pred_label
        set_status("ANALYSIS COMPLETE • Random Forest prediction generated")

        update_report()

    except Exception as exc:
        messagebox.showerror("Analysis Error", str(exc))


def update_report():
    avg = avg_var.get()
    signals = performance_signal(
        safe_float(avg),
        safe_float(attendance_var.get()),
        behaviour_var.get(),
        presentation_var.get(),
        analytical_participation_var.get(),
        assignment_var.get()
    )

    report = f"""
============================================================
          STUDENT PERFORMANCE INTELLIGENCE REPORT
============================================================

STUDENT PROFILE
------------------------------------------------------------
Name                 : {name_var.get()}
Roll Number          : {roll_var.get()}

PREDICTIVE PROFILE
------------------------------------------------------------
Attendance           : {attendance_var.get()}%
Study Hours / Day    : {study_hours_var.get()}
Assignment Performance : {assignment_performance_var.get()}%
Class Participation  : {participation_var.get()}%
Previous Semester GPA: {previous_gpa_var.get()}

ACADEMIC RESULT
------------------------------------------------------------
Mathematics          : {math_var.get()}   Grade: {marks_labels["Mathematics"].get()}
Reading              : {reading_var.get()}   Grade: {marks_labels["Reading"].get()}
Writing              : {writing_var.get()}   Grade: {marks_labels["Writing"].get()}
Overall Average      : {avg}

STUDENT DEVELOPMENT
------------------------------------------------------------
Attendance           : {attendance_var.get()}%
Behaviour            : {behaviour_var.get()}
Presentation         : {presentation_var.get()}
Participation        : {participation_var.get()}
Assignment           : {assignment_var.get()}
Communication        : {communication_var.get()}
Discipline           : {discipline_var.get()}

ANALYTICAL PROFILE
------------------------------------------------------------
Profile              : {profile_var.get()}
Evaluation Score     : {profile_score_var.get()}

PERFORMANCE SIGNALS
------------------------------------------------------------
"""

    for level, msg in signals:
        report += f"{level:<12}: {msg}\n"

    report += f"""
MACHINE LEARNING PREDICTION
------------------------------------------------------------
Predicted Class      : {prediction_var.get()}
Confidence           : {confidence_var.get()}
Selected Model       : Baseline Random Forest
Selection Criterion  : Highest Weighted F1

MODEL PERFORMANCE
------------------------------------------------------------
Accuracy             : 35.00%
Weighted Precision   : 37.06%
Weighted Recall      : 35.00%
Weighted F1          : 35.49%
ROC-AUC              : 57.06%

NOTE
------------------------------------------------------------
The Random Forest prediction is generated from five academic
and engagement-related predictive features:
Attendance, Study Hours, Assignment Performance,
Class Participation and Previous Semester GPA.

Mathematics, Reading and Writing are used in the intelligence
layer for academic analysis and are not passed directly to
the Random Forest model.
============================================================
"""

    report_text.delete("1.0", "end")
    report_text.insert("1.0", report)


def generate_report():
    if prediction_var.get() == "—":
        run_full_analysis()
        if prediction_var.get() == "—":
            return

    update_report()

    file_path = filedialog.asksaveasfilename(
        title="Save Student Performance Report",
        defaultextension=".txt",
        filetypes=[("Text File", "*.txt")]
    )
    if not file_path:
        return

    try:
        Path(file_path).write_text(
            report_text.get("1.0", "end"),
            encoding="utf-8"
        )
        messagebox.showinfo("Report Saved", f"Report saved successfully:\n{file_path}")
    except Exception as exc:
        messagebox.showerror("Save Error", str(exc))


# Bind automatic updates
for var in [
    math_var,
    reading_var,
    writing_var,
    attendance_var,
    study_hours_var,
    assignment_performance_var,
    participation_var,
    previous_gpa_var,
    behaviour_var,
    presentation_var
]:
    var.trace_add("write", lambda *_: draw_fingerprint())
   

root.after(300, draw_fingerprint)
root.after(500, lambda: update_sim_values())


if __name__ == "__main__":
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
        root.destroy()
