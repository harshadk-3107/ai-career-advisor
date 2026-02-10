import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AI Career Advisor",
    layout="wide",
    page_icon="🚀"
)


CAREER_SKILLS = {
    "Data Scientist": {
        "python", "sql", "data science", "statistics", "machine learning",
        "pandas", "numpy", "data visualization"
    },
    "Machine Learning Engineer": {
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "model deployment", "mlops"
    },
    "Backend Developer": {
        "python", "java", "node", "api", "sql", "databases", "backend"
    },
    "Frontend Developer": {
        "html", "css", "javascript", "react", "ui", "frontend"
    },
    "DevOps Engineer": {
        "docker", "kubernetes", "aws", "ci/cd", "linux", "cloud"
    },
    "Business Analyst": {
        "sql", "excel", "business analysis", "dashboard",
        "data analytics", "power bi"
    }
}


st.sidebar.title("⚙️ Controls")
top_n = st.sidebar.slider("Number of career paths", 3, 6, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**AI Career Advisor**")
st.sidebar.markdown("✔ Skill-based reasoning")
st.sidebar.markdown("✔ Visual insights")
st.sidebar.markdown("✔ Role clarity")

st.markdown(
    """
    <h1>🚀 AI Career Advisor</h1>
    <p style="opacity:0.8">A visual dashboard to understand where your skills fit best</p>
    """,
    unsafe_allow_html=True
)


skills_input = st.text_input(
    "Enter your skills (comma separated)",
    placeholder="python, sql, data science"
)

if not skills_input.strip():
    st.stop()

user_skills = {s.strip().lower() for s in skills_input.split(",") if s.strip()}


results = []

for role, role_skills in CAREER_SKILLS.items():
    overlap = user_skills & role_skills
    score = len(overlap) / len(role_skills)
    results.append({
        "role": role,
        "score": score,
        "matched_skills": overlap
    })

results.sort(key=lambda x: x["score"], reverse=True)

top_results = results[:top_n]
top_role = top_results[0]


confidence_map = {
    0.7: ("High Fit", "🟢", 90),
    0.4: ("Moderate Fit", "🟡", 65),
    0.2: ("Exploratory", "🔴", 40)
}

for threshold, val in confidence_map.items():
    if top_role["score"] >= threshold:
        confidence_label, confidence_icon, readiness = val
        break


st.markdown("## 🎯 Career Snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Top Role", top_role["role"])

with col2:
    st.metric("Confidence", f"{confidence_icon} {confidence_label}")

with col3:
    st.progress(readiness / 100)
    st.caption(f"Readiness: {readiness}%")

st.markdown("---")


st.markdown("## 🧩 Skill Impact")

if top_role["matched_skills"]:
    labels = list(top_role["matched_skills"])
    sizes = [1] * len(labels)

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")

    st.pyplot(fig)
else:
    st.info("No overlapping skills found for the top role.")


st.markdown("## 📊 Match Strength")

for r in top_results:
    st.markdown(f"**{r['role']}**")
    st.progress(r["score"])
-
st.markdown("---")
st.markdown("## 🔍 Explore Other Career Fits")

cols = st.columns(len(top_results))

for col, r in zip(cols, top_results):
    with col:
        st.markdown(f"**{r['role']}**")
        st.progress(r["score"])
        st.caption(f"{int(r['score']*100)}% skill alignment")

