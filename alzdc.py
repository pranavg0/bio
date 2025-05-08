import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# — Page config —
st.set_page_config(page_title="Diet & Brain Health Explorer", layout="centered")

# — Title —
st.title("Diet and Brain Health Explorer")
st.write(
    "Complete the quiz to see how brain-healthy your diet is, "
    "then explore how diet scores relate to cognitive test results."
)

# -------------------------------------------------------------------
# 1. Personal Diet Quiz (8 questions)
# -------------------------------------------------------------------
st.header("1. Personal Diet Quiz")

score = 0
healthy = 0
unhealthy = 0

def ask(question, options, mapping, key):
    """Ask one question and update score & counts."""
    global score, healthy, unhealthy
    ans = st.selectbox(question, options, key=key)
    delta = mapping.get(ans, 0)
    score += delta
    if delta > 0:
        healthy += 1
    elif delta < 0:
        unhealthy += 1

# question definitions
questions = [
    ("Servings of leafy greens per week", ["None","1–3","4–6","7+"], {"1–3":1,"4–6":2,"7+":3}),
    ("Servings of berries per week",      ["None","1–2","3–4","5+"], {"1–2":1,"3–4":2,"5+":3}),
    ("Servings of fish per week",         ["None","1–2","3–4","5+"], {"1–2":1,"3–4":2,"5+":3}),
    ("Servings of nuts per week",         ["None","1–2","3–4","5+"], {"1–2":1,"3–4":2,"5+":3}),
    ("Use of olive oil daily",            ["Never","Sometimes","Daily"], {"Sometimes":1,"Daily":2}),
    ("Servings of whole grains per day",  ["<1","1–2","3–4","5+"], {"1–2":1,"3–4":2,"5+":3}),
    ("Servings of red meat per week",     ["5+","3–4","1–2","None"], {"3–4":-1,"5+":-2}),
    ("Sugary snacks or soda per week",    ["Daily","Several","Weekly","Rarely"], {"Several":-1,"Daily":-2})
]

# ask them all
for idx, (q, opts, mp) in enumerate(questions):
    ask(f"{idx+1}. {q}", opts, mp, key=f"q{idx}")

# compute percent healthy
total = healthy + unhealthy
percent_healthy = int(healthy/total*100) if total>0 else 0

# display quiz results
st.subheader("Quiz Results")
st.write(f"• Total score: **{score}**")
st.write(f"• Healthy choices: **{healthy}**  Unhealthy choices: **{unhealthy}**")
st.write(f"• Percent healthy: **{percent_healthy}%**")

# pie chart
fig1, ax1 = plt.subplots()
ax1.pie([healthy, unhealthy], labels=["Healthy","Unhealthy"], autopct="%1.1f%%", startangle=90)
ax1.axis("equal")
st.pyplot(fig1)

# -------------------------------------------------------------------
# 2. Sample Data Correlation & Deeper Analysis
# -------------------------------------------------------------------
st.header("2. Sample Data & Analysis")

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("brain_diet_data.csv")
    # recompute diet_score using same 5 variables
    df["diet_score"] = (
        df["leafy_greens"] + df["berries"] + df["fish"] +
        df["nuts"] + df["olive_oil"]
        - df["red_meat"] - df["sweets"] - df["fast_food"]
    )
    return df

try:
    df = load_and_prepare()
    st.write("Sample of imported data:")
    st.dataframe(df.head(), use_container_width=True)

    # histograms
    fig2, (ax2, ax3) = plt.subplots(1,2, figsize=(10,4))
    ax2.hist(df["diet_score"], bins=15, edgecolor="black")
    ax2.set_title("Diet Score Distribution")
    ax2.set_xlabel("Diet Score")
    ax2.set_ylabel("Count")
    ax3.hist(df["cognitive_score"], bins=15, edgecolor="black")
    ax3.set_title("Cognitive Score Distribution")
    ax3.set_xlabel("Cognitive Score")
    ax3.set_ylabel("Count")
    st.pyplot(fig2)

    # correlation matrix
    vars_corr = ["diet_score","cognitive_score","microbiome_diversity","amyloid_beta","tau_protein"]
    corr = df[vars_corr].corr()
    fig4, ax4 = plt.subplots(figsize=(5,4))
    cax = ax4.matshow(corr, cmap="coolwarm")
    fig4.colorbar(cax)
    ax4.set_xticks(range(len(vars_corr)))
    ax4.set_xticklabels(vars_corr, rotation=45, ha="right")
    ax4.set_yticks(range(len(vars_corr)))
    ax4.set_yticklabels(vars_corr)
    ax4.set_title("Correlation Matrix")
    st.pyplot(fig4)

    # regression
    x = df["diet_score"].values.reshape(-1,1)
    y = df["cognitive_score"].values
    slope, intercept, r_val, p_val, _ = linregress(df["diet_score"], df["cognitive_score"])
    st.write(f"Linear regression: cognitive_score ≈ {slope:.2f}*diet_score + {intercept:.2f}")
    st.write(f"Pearson r = {r_val:.2f}, p = {p_val:.3f}")

    # scatter plot
    fig5, ax5 = plt.subplots()
    ax5.scatter(df["diet_score"], df["cognitive_score"], alpha=0.6)
    ax5.plot(df["diet_score"], slope*df["diet_score"]+intercept, "r--")
    ax5.set_xlabel("Diet Score")
    ax5.set_ylabel("Cognitive Score")
    ax5.set_title("Diet Score vs Cognitive Score")
    st.pyplot(fig5)

except FileNotFoundError:
    st.error("Data file 'brain_diet_data.csv' not found. Please place it next to this script.")

# -------------------------------------------------------------------
# 3. References
# -------------------------------------------------------------------
st.header("References")
st.markdown(
    "- Alzheimer’s Association: https://www.alz.org\n"
    "- CDC – Diet and Alzheimer’s Disease: https://www.cdc.gov/alzheimers/diet\n"
    "- National Institute on Aging – MIND Diet: https://www.nia.nih.gov/health/mind-diet"
)
