import pandas as pd
import streamlit as st

from src.analyzer import analyze

st.set_page_config(page_title="Roman Urdu Complaint Analyzer", page_icon="📨", layout="centered")

PRIORITY_COLOR = {"High": "#d64545", "Medium": "#e0a030", "Low": "#3a9d5d"}
SENTIMENT_COLOR = {"Negative": "#d64545", "Neutral": "#6b7280", "Positive": "#3a9d5d"}

EXAMPLES = [
    "sir LMS pe assignment submit ni horahi aur deadline kal hai",
    "fee challan mein galat amount aa raha hai please check kar lein",
    "attendance shortage ki wajah se kal ka paper nahi dene de rahe urgent help",
    "library ki timing kya hai",
    "campus ka wifi ab bohat fast hai shukriya",
]


def badge(label, color):
    return (f"<span style='background:{color};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:600;font-size:1.05rem'>{label}</span>")


st.title("📨 Roman Urdu Student Complaint Analyzer")
st.caption("Enter a complaint in Roman Urdu, English, or a mix of both. "
           "The system predicts its category, sentiment and priority.")

tab_single, tab_batch = st.tabs(["Single complaint", "Batch (CSV)"])

with tab_single:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES[0]
    st.write("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(f"#{i + 1}", help=ex, key=f"ex{i}"):
            st.session_state.text = ex
    text = st.text_area("Complaint", key="text", height=100)

    if st.button("Analyze", type="primary") and text.strip():
        r = analyze(text)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Category**")
        c1.markdown(badge(r["category"], "#2f6db5"), unsafe_allow_html=True)
        c1.caption(f"confidence {r['category_confidence']:.0%}")
        c2.markdown("**Sentiment**")
        c2.markdown(badge(r["sentiment"], SENTIMENT_COLOR[r["sentiment"]]), unsafe_allow_html=True)
        c2.caption(f"confidence {r['sentiment_confidence']:.0%}")
        c3.markdown("**Priority**")
        c3.markdown(badge(r["priority"]["label"], PRIORITY_COLOR[r["priority"]["label"]]), unsafe_allow_html=True)
        c3.caption(f"score {r['priority']['score']}")

        if r["priority"]["urgency_hits"]:
            st.info("Urgency indicators found: " + ", ".join(r["priority"]["urgency_hits"]))

        with st.expander("Category probabilities"):
            probs = pd.Series(r["category_probs"]).sort_values(ascending=False)
            st.bar_chart(probs)
        with st.expander("Text preprocessing"):
            st.write("**Normalized:**", r["normalized"])
            st.write("**After stopword removal:**", r["clean"])

with tab_batch:
    st.write("Upload a CSV with a column named **text**.")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        if "text" not in df.columns:
            st.error("The CSV needs a 'text' column.")
        else:
            rows = []
            for t in df["text"].astype(str):
                r = analyze(t)
                rows.append({"text": t, "category": r["category"], "sentiment": r["sentiment"],
                             "priority": r["priority"]["label"], "priority_score": r["priority"]["score"]})
            out = pd.DataFrame(rows)
            order = {"High": 0, "Medium": 1, "Low": 2}
            out = out.sort_values("priority", key=lambda s: s.map(order))
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode(), "analyzed_complaints.csv")
