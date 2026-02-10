import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🚀",
    layout="wide"
)

# ==================================================
# LOAD MODEL & DATA
# ==================================================
@st.cache_resource
def load_model():
    with open("tfidf_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("final_job_skills.csv")

tfidf = load_model()
df = load_data()

# ==================================================
# CLUSTERING (ROLE INFERENCE)
# ==================================================
@st.cache_resource
def cluster_jobs(text):
    vectors = tfidf.transform(text)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    return kmeans.fit_predict(vectors)

df["cluster"] = cluster_jobs(df["job_skills"])

CLUSTER_ROLE_MAP = {
    0: "Machine Learning / AI Engineer",
    1: "Data Analyst / Data Scientist",
    2: "Full Stack / Web Developer",
    3: "Cloud / DevOps Engineer",
    4: "General Technical Role"
}

def get_role(cluster):
    return CLUSTER_ROLE_MAP.get(cluster, "General Technical Role")

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def confidence_label(score):
    if score >= 0.55:
        return "🟢 Strong Fit"
    elif score >= 0.45:
        return "🟡 Moderate Fit"
    else:
        return "🔴 Exploratory"

def readiness_level(score):
    if score >= 0.55:
        return 0.85, "Ready"
    elif score >= 0.45:
        return 0.6, "Almost Ready"
    else:
        return 0.35, "Needs Upskilling"

def skill_sets(user_skills, job_skills):
    user = {s.strip().lower() for s in user_skills.split(",")}
    job = {s.strip().lower() for s in job_skills.split(",") if len(s.strip()) > 2}
    return user & job, job - user

def skill_impact(user_skills, job_skills, tfidf, user_vector):
    user_set = {s.strip().lower() for s in user_skills.split(",")}
    job_list = [s.strip().lower() for s in job_skills.split(",")]

    overlap = [s for s in job_list if s in user_set]
    if not overlap:
        return None

    features = tfidf.get_feature_names_out()
    vec = user_vector.toarray()[0]

    weights = {}
    for skill in overlap:
        if skill in features:
            idx = list(features).index(skill)
            weights[skill] = vec[idx]

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else None

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Controls")

top_n = st.sidebar.slider(
    "Number of career paths",
    3, 8, 5
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **AI Career Advisor**  
    ✔ NLP skill matching  
    ✔ Visual insights  
    ✔ Career guidance  
    """
)

# ==================================================
# HEADER
# ==================================================
st.title("🚀 AI Career Advisor")
st.caption("A visual AI dashboard to understand where your skills fit best")

user_skills = st.text_input(
    "Enter your skills (comma separated)",
    placeholder="python, machine learning, sql, cloud"
)

# ==================================================
# MAIN LOGIC
# ==================================================
if user_skills.strip():

    user_vector = tfidf.transform([user_skills])
    job_vectors = tfidf.transform(df["job_skills"])

    scores = cosine_similarity(user_vector, job_vectors).flatten()
    df["match_score"] = scores

    top_matches = (
        df.sort_values("match_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top = top_matches.iloc[0]
    role = get_role(top["cluster"])
    confidence = confidence_label(top["match_score"])
    readiness, readiness_text = readiness_level(top["match_score"])
    overlap, gap = skill_sets(user_skills, top["job_skills"])
    impact = skill_impact(user_skills, top["job_skills"], tfidf, user_vector)

    # ==================================================
    # HERO SNAPSHOT
    # ==================================================
    st.markdown("## 🎯 Career Snapshot")

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Role", role)
    col2.metric("Confidence", confidence)
    col3.progress(readiness, text=f"Readiness: {readiness_text}")

    st.divider()

    # ==================================================
    # MAIN VISUALS
    # ==================================================
    left, right = st.columns([1, 1])

    # -------- Skill Impact Pie (PRIMARY VISUAL)
    with left:
        st.subheader("🧩 Skill Impact")
        if impact:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                impact.values(),
                labels=impact.keys(),
                autopct="%1.0f%%",
                startangle=140
            )
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("Not enough overlapping skills to compute impact.")

    # -------- Skill Match Bars
    with right:
        st.subheader("📊 Skill Match Strength")
        if overlap:
            for skill in list(overlap)[:6]:
                st.write(skill.capitalize())
                st.progress(0.7)
        else:
            st.write("Low direct overlap detected.")

    st.divider()

    # ==================================================
    # GROWTH PATH
    # ==================================================
    st.subheader("🚀 Growth Path")

    if gap:
        st.write("Skills that will increase your readiness:")
        cols = st.columns(min(4, len(gap)))
        for i, skill in enumerate(list(gap)[:8]):
            cols[i % len(cols)].success(skill)
    else:
        st.success("You already meet most core skill requirements!")

    st.divider()

    # ==================================================
    # EXPLORE OTHER CAREERS (MINI CARDS)
    # ==================================================
    st.subheader("🔍 Explore Other Career Fits")

    cards = st.columns(len(top_matches))

    for i, row in top_matches.iterrows():
        with cards[i]:
            st.markdown(
                f"""
                **{get_role(row['cluster'])}**  
                {confidence_label(row['match_score'])}
                """
            )
            st.progress(min(row["match_score"], 1.0))

else:
    st.info("👆 Enter your skills to see a visual career dashboard")