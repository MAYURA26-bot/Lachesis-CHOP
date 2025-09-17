import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from recommendation import generate_recommendations 

# --------------------- Page ---------------------
st.set_page_config(page_title="Child Obesity Risk — Doctor-Style Interview", layout="centered")
st.title("🧒 Childhood Obesity : Doctor-Style Adaptive Interview")
# questions by topic (like a checkup). A surrogate Decision Tree picks the next topic. The RandomForest makes the final prediction.")

# --------------------- Load models ---------------------
@st.cache_resource
def load_artifacts():
    rf = joblib.load("obesity_model.pkl")       # RandomForest (final predictor)
    enc = joblib.load("encoders.pkl")           # {'nobeyesdad': {...}}
    bun = joblib.load("surrogate_dt.pkl")       # {'model','feature_names','class_names','fidelity'}
    dt = bun["model"]                            # DecisionTree (for flow)
    dt_feats = bun["feature_names"]
    class_names = bun.get("class_names")
    fidelity = bun.get("fidelity")
    rf_feats = list(getattr(rf, "feature_names_in_", dt_feats))
    return rf, enc, dt, dt_feats, class_names, fidelity, rf_feats

rf, encoders, dt, DT_FEATURES, CLASS_NAMES, FIDELITY, RF_FEATURES = load_artifacts()
TREE = dt.tree_

# --------------------- Label map ---------------------
inv_label = {v: k for k, v in encoders["nobeyesdad"].items()}

# --------------------- Doctor-style topics (clusters) ---------------------
TOPICS = {
    "Vitals": ["age", "height", "weight", "gender_Male"],
    "Diet":   ["favc", "fcvc", "ncp", "ch2o", "scc", "caec_Always", "caec_Frequently", "caec_Sometimes"],
    "Activity": ["faf", "mtrans_Bike", "mtrans_Motorbike", "mtrans_Public_Transportation", "mtrans_Walking"],
    "Screen Time": ["tue"],
    "Family": ["family_history_with_overweight"],
}

# quick reverse map: feature -> topic
FEAT2TOPIC = {f: t for t, feats in TOPICS.items() for f in feats}

# groups handled with one control
CAEC = ["caec_Always","caec_Frequently","caec_Sometimes"]
MTRANS = ["mtrans_Bike","mtrans_Motorbike","mtrans_Public_Transportation","mtrans_Walking"]

# --------------------- State ---------------------
if "answers" not in st.session_state: st.session_state.answers = {}
if "node" not in st.session_state:    st.session_state.node = 0
if "path" not in st.session_state:    st.session_state.path = []    # for explanation
if "phase" not in st.session_state:   st.session_state.phase = "interview"  # -> "complete" -> "done"
if "last_topic" not in st.session_state: st.session_state.last_topic = None  # UI continuity

def reset_all():
    st.session_state.answers = {}
    st.session_state.node = 0
    st.session_state.path = []
    st.session_state.phase = "interview"
    st.session_state.last_topic = None

with st.sidebar:
    st.button("🔄 Reset interview", on_click=reset_all)
    #if FIDELITY is not None:
    #    st.caption(f"Surrogate fidelity to RF: **{FIDELITY:.2%}**")

# --------------------- Helpers ---------------------
def is_leaf(n: int) -> bool:
    return TREE.children_left[n] == -1 and TREE.children_right[n] == -1

def auto_advance():
    """
    Keep advancing along the DT as long as we already have the feature needed at the current split.
    Append rule steps to path. Stop at first missing feature OR leaf.
    Returns the missing feature name (str) or None if leaf/no question needed.
    """
    while not is_leaf(st.session_state.node):
        fi = TREE.feature[st.session_state.node]
        thr = TREE.threshold[st.session_state.node]
        fname = DT_FEATURES[fi]
        if fname not in st.session_state.answers:
            return fname  # we need to ask about this feature
        val = float(st.session_state.answers[fname])
        nextn = TREE.children_left[st.session_state.node] if (val <= thr) else TREE.children_right[st.session_state.node]
        st.session_state.path.append({"feature": fname, "threshold": float(thr), "value": val})
        st.session_state.node = nextn
    return None  # leaf or fully answered path

def unresolved_in_topic(topic: str) -> list:
    return [f for f in TOPICS[topic] if f not in st.session_state.answers and f in DT_FEATURES]

def next_topic():
    """
    Decide which topic to ask next (doctor style):
      1) If DT needs a specific feature, take that feature’s topic.
      2) Else, pick the first topic with unresolved features.
    """
    need = auto_advance()
    if is_leaf(st.session_state.node):
        st.session_state.phase = "complete"
        return None

    if need:
        return FEAT2TOPIC.get(need, "Vitals")  # fall back to vitals

    # No specific need (rare) — just finish topics in a sensible order
    for t in ["Vitals", "Diet", "Activity", "Screen Time", "Family"]:
        if unresolved_in_topic(t):
            return t
    return None

def bmi_preview():
    # Not used by model; just for clinician-style feedback
    if all(k in st.session_state.answers for k in ("height","weight")):
        h = float(st.session_state.answers["height"])
        w = float(st.session_state.answers["weight"])
        if h > 0:
            return w / (h*h)
    return None

# --------------------- Render a topic (no callbacks inside form) ---------------------
def render_topic(topic: str) -> dict:
    """
    Show a few related questions together, like a doctor.
    Return dict of updates to apply after submit.
    Only renders unresolved features from that topic.
    """
    updates = {}
    st.subheader(f"🩺 {topic}")

    def yesno(label, default="No"):
        val = st.radio(label, ["No","Yes"], horizontal=True, index=(1 if default=="Yes" else 0))
        return 1.0 if val == "Yes" else 0.0
    def slider(label, lo, hi, step, default):
        return float(st.slider(label, min_value=lo, max_value=hi, step=step, value=default))
    def number(label, lo, hi, step, default):
        return float(st.number_input(label, min_value=lo, max_value=hi, step=step, value=default))

    feats = unresolved_in_topic(topic)

    # Vitals
    if topic == "Vitals":
        if "age" in feats:       updates["age"] = slider("Age (years)", 14, 18, 1, 16)
        if "height" in feats:    updates["height"] = number("Height (meters)", 1.0, 2.2, 0.01, 1.45)
        if "weight" in feats:    updates["weight"] = number("Weight (kg)", 10.0, 200.0, 0.5, 45.0)
        if "gender_Male" in feats: updates["gender_Male"] = yesno("Gender: Male? (Yes for Male, No for Female)")
        bmi = bmi_preview()
        if bmi:
            st.caption(f"Provisional BMI (for context only): **{bmi:.1f}**")

    # Diet
    if topic == "Diet":
        if "favc" in feats:      updates["favc"] = yesno("Do you often eat high-calorie foods (FAVC)?")
        if "fcvc" in feats:      updates["fcvc"] = slider("Vegetable intake (1 = rarely, 3 = daily)", 1, 3, 1, 2)
        if "ncp" in feats:       updates["ncp"] = slider("Main meals per day (NCP)", 1, 6, 1, 3)
        if "ch2o" in feats:      updates["ch2o"] = slider("Daily water (liters)", 1, 5, 1, 2)
        if "scc" in feats:       updates["scc"] = yesno("Do you monitor calorie intake (SCC)?")
        # CAEC as single select → one-hots
        if any(f in feats for f in CAEC):
            caec_choice = st.selectbox("Snacking between meals (CAEC)", ["Always","Frequently","Sometimes"], index=1)
            for k,opt in zip(CAEC, ["Always","Frequently","Sometimes"]):
                updates[k] = 1.0 if caec_choice == opt else 0.0

    # Activity
    if topic == "Activity":
        if "faf" in feats: updates["faf"] = slider("Physical activity frequency (0 none – 3 high)", 0, 3, 1, 1)
        # MTRANS as single select → one-hots
        if any(f in feats for f in MTRANS):
            mtrans_choice = st.selectbox("Primary transport mode (MTRANS)", ["Bike","Motorbike","Public Transportation","Walking"], index=2)
            mapping = {
                "Bike":"mtrans_Bike","Motorbike":"mtrans_Motorbike",
                "Public Transportation":"mtrans_Public_Transportation","Walking":"mtrans_Walking"
            }
            chosen = mapping[mtrans_choice]
            for k in MTRANS:
                updates[k] = 1.0 if k == chosen else 0.0

    # Screen Time
    if topic == "Screen Time":
        if "tue" in feats: updates["tue"] = slider("Tech use per day (0 low – 3 high)", 0, 3, 1, 2)

    # Family
    if topic == "Family":
        if "family_history_with_overweight" in feats:
            updates["family_history_with_overweight"] = yesno("Family history of overweight?")

    # Fallback generic (if any other feature slipped in)
    for f in feats:
        if f not in updates:
            updates[f] = number(f"Provide {f}", -1e6, 1e6, 0.1, 0.0)

    return updates

def remaining_features(order: list[str]) -> list[str]:
    return [f for f in order if f in DT_FEATURES and f not in st.session_state.answers]

# --------------------- Interview loop ---------------------
st.markdown("### 👨‍⚕️ Interview")

# Decide next topic (DT-driven), then render that topic as a single step.
topic = None
if st.session_state.phase == "interview":
    topic = next_topic()
    if topic is None and st.session_state.phase == "complete":
        pass  # moved to completion
    elif topic is None:
        # nothing to ask and not at leaf -> just finish orderly
        st.session_state.phase = "complete"

if st.session_state.phase == "interview" and topic:
    with st.form("topic_form"):
        updates = render_topic(topic)
        submitted = st.form_submit_button("Continue")
    if submitted:
        st.session_state.answers.update(updates)
        st.session_state.last_topic = topic
        st.rerun()

# --------------------- Leaf reached → Explain local path ---------------------


# --------------------- Completion: finish any remaining features ---------------------
UNIFIED_ORDER = RF_FEATURES if RF_FEATURES else DT_FEATURES

if st.session_state.phase == "complete":
    rem = remaining_features(UNIFIED_ORDER)
    if rem:
        # Pick the topic containing the first remaining feature
        t = FEAT2TOPIC.get(rem[0], "Vitals")
        with st.form("finish_form"):
            updates = render_topic(t)
            done = st.form_submit_button("Save & Next")
        if done:
            st.session_state.answers.update(updates)
            if not remaining_features(UNIFIED_ORDER):
                st.session_state.phase = "done"
            st.rerun()
    else:
        st.session_state.phase = "done"

# --------------------- Final RF prediction ---------------------
if st.session_state.phase == "done":
    row = {fn: 0.0 for fn in UNIFIED_ORDER}
    row.update(st.session_state.answers)
    X_row = pd.DataFrame([row])[UNIFIED_ORDER]
    try:
        pred = int(rf.predict(X_row)[0])
        st.success(f"🏷️ Final RandomForest prediction: **{inv_label.get(pred, str(pred))}**")

        # --- Recommendations UI (nice tabs) ---
        st.markdown("### 🧭 Recommendations")

        recs = generate_recommendations(pred)  # pass the integer prediction

        def _fmt(txt):
            if txt is None or txt == "":
                return "—"
            if isinstance(txt, (list, tuple)):
                return "\n".join([f"- {t}" for t in txt])
            return str(txt)

        if isinstance(recs, dict) and recs:
            tab_food, tab_ex, tab_other = st.tabs(["🍎 Food / Drink", "🏃 Activity", "🧭 Other"])

            with tab_food:
                st.markdown(_fmt(recs.get("Food/Drink")))
            with tab_ex:
                st.markdown(_fmt(recs.get("Exercise")))
            with tab_other:
                st.markdown(_fmt(recs.get("Other")))

            if "Note" in recs and recs["Note"]:
                st.caption(recs["Note"])
        else:
            st.info("No recommendations available for this case.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")