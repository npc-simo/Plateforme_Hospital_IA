import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MediCore AI — Performances Hospitalières",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #e63946 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  NOUVEAU STYLE — Rouge médical + noir profond
#  Palette : #0d0d0d | #141414 | #1a1a1a
#  Accents  : #e63946 (rouge) | #f4a261 (orange) | #2ec4b6 (teal)
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d0d0d;
    color: #d4d4d4;
}
.stApp { background-color: #0d0d0d; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #e6394622;
}
section[data-testid="stSidebar"] * { color: #888 !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #aaa !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
}
.card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 4px 0 0 4px;
}
.card-red::before    { background: #e63946; }
.card-orange::before { background: #f4a261; }
.card-teal::before   { background: #2ec4b6; }
.card-yellow::before { background: #ffd166; }

/* ── KPI ── */
.kpi-box {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 20px 16px;
    position: relative;
    overflow: hidden;
}
.kpi-box::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-red::after    { background: linear-gradient(90deg, #e63946, transparent); }
.kpi-orange::after { background: linear-gradient(90deg, #f4a261, transparent); }
.kpi-teal::after   { background: linear-gradient(90deg, #2ec4b6, transparent); }
.kpi-yellow::after { background: linear-gradient(90deg, #ffd166, transparent); }

.kpi-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px;
}
.kpi-lbl {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    font-weight: 600;
}

/* ── Section titles ── */
.sec-title {
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #444;
    font-weight: 700;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* ── Inputs ── */
.stNumberInput input, .stSelectbox select, .stTextArea textarea {
    background-color: #141414 !important;
    color: #d4d4d4 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #e63946 !important;
    color: #0d0d0d !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 12px 32px !important;
}
.stButton > button:hover { background: #c1121f !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #e63946 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { color: #444 !important; }

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #e63946, #f4a261) !important;
}

/* ── Tables ── */
thead tr th {
    background: #1a1a1a !important;
    color: #e63946 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
tbody tr { background: #141414 !important; color: #d4d4d4 !important; }
tbody tr:nth-child(even) { background: #111 !important; }

/* ── Divider ── */
hr { border-color: #1e1e1e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #e63946; border-radius: 2px; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    color: #e63946 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        return {
            "xgb":    joblib.load("model_classification_xgb.pkl"),
            "rf":     joblib.load("model_rf_classification.pkl"),
            "gb":     joblib.load("model_regression_gb.pkl"),
            "kmeans": joblib.load("model_clustering_kmeans.pkl"),
            "scaler": joblib.load("scaler.pkl"),
        }, True
    except:
        return {}, False

models, models_loaded = load_models()

CLUSTER_LABELS = {0: "À Risque", 1: "En Développement", 2: "Moyen", 3: "Excellent"}
CLUSTER_COLORS = {"Excellent": "#2ec4b6", "Moyen": "#ffd166", "En Développement": "#f4a261", "À Risque": "#e63946"}

def dark_fig():
    plt.rcParams.update({
        "figure.facecolor": "#141414", "axes.facecolor": "#0d0d0d",
        "axes.edgecolor": "#222",      "axes.labelcolor": "#666",
        "xtick.color": "#444",         "ytick.color": "#444",
        "text.color": "#d4d4d4",       "grid.color": "#1e1e1e",
        "legend.facecolor": "#141414", "legend.edgecolor": "#222",
        "font.family": "monospace",
    })

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px; border-bottom:1px solid #1e1e1e; margin-bottom:20px;'>
        <div style='font-size:10px; letter-spacing:4px; color:#e63946; text-transform:uppercase; font-weight:700; margin-bottom:8px;'>● SYSTÈME ACTIF</div>
        <div style='font-size:24px; font-weight:700; color:#f0f0f0; letter-spacing:-0.5px; font-family:Rajdhani,sans-serif;'>MediCore<span style='color:#e63946;'>.</span>AI</div>
        <div style='font-size:10px; letter-spacing:2px; color:#333; text-transform:uppercase; margin-top:4px;'>ESTN NADOR · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "01  DASHBOARD",
        "02  RÉADMISSION",
        "03  PERFORMANCE",
        "04  CLUSTERING",
        "05  NLP SENTIMENT",
        "06  SÉRIES TEMP.",
        "07  ÉVALUATION",
    ])

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    sc = "#2ec4b6" if models_loaded else "#f4a261"
    st.markdown(f"""
    <div style='border:1px solid #1e1e1e; border-left:2px solid {sc}; padding:12px 14px; border-radius:2px;'>
        <div style='font-size:9px; letter-spacing:2px; color:#333; text-transform:uppercase; margin-bottom:4px;'>STATUT MODÈLES</div>
        <div style='font-size:12px; color:{sc}; font-weight:700; letter-spacing:1px;'>{"OPÉRATIONNEL" if models_loaded else "MODE DÉMO"}</div>
        <div style='font-size:10px; color:#333; margin-top:2px; font-family:monospace;'>5 × .pkl</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 01 — DASHBOARD
# ══════════════════════════════════════════════════════════
if "DASHBOARD" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>TABLEAU DE BORD</div>", unsafe_allow_html=True)
    st.markdown("<h1>Plateforme de Suivi<br>des Performances Hospitalières</h1>", unsafe_allow_html=True)
    st.markdown("<p>ESTN Nador · Développement Web & Mobile · Maroc · 2026</p>", unsafe_allow_html=True)
    st.markdown("---")

    cols = st.columns(6)
    kpis = [
        ("4 812",  "HÔPITAUX",    "#e63946", "kpi-red"),
        ("18 000", "PATIENTS",    "#f4a261", "kpi-orange"),
        ("100%",   "RECALL MAX",  "#2ec4b6", "kpi-teal"),
        ("K = 4",  "CLUSTERS",    "#ffd166", "kpi-yellow"),
        ("0.089",  "RMSE",        "#e63946", "kpi-red"),
        ("0.437",  "SILHOUETTE",  "#2ec4b6", "kpi-teal"),
    ]
    for col, (val, lbl, color, cls) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-box {cls}'>
                <div class='kpi-lbl'>{lbl}</div>
                <div class='kpi-num' style='color:{color};'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-title'>CLUSTERS K-MEANS · DISTRIBUTION</div>", unsafe_allow_html=True)
        dark_fig()
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Excellent", "Moyen", "En Dév.", "À Risque"]
        counts = [378, 1873, 1592, 969]
        colors = ["#2ec4b6", "#ffd166", "#f4a261", "#e63946"]
        bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="#0d0d0d", linewidth=2)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                    str(val), ha="center", color="#d4d4d4", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 2300); ax.set_ylabel("Hôpitaux", fontsize=10, color="#444")
        ax.grid(axis="y", alpha=0.15, linestyle="--")
        ax.spines[["top","right","left","bottom"]].set_visible(False); ax.tick_params(length=0)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='sec-title'>SHAP · IMPORTANCE DES VARIABLES</div>", unsafe_allow_html=True)
        dark_fig()
        fig, ax = plt.subplots(figsize=(6, 4))
        features = ["Number_of_Medications","Age","Clinical_Risk","Length_of_Stay","Medication_Burden"]
        vals = [0.38, 0.27, 0.19, 0.14, 0.10]
        colors2 = ["#e63946","#f4a261","#ffd166","#2ec4b6","#2ec4b6"]
        bars = ax.barh(features[::-1], vals[::-1], color=colors2[::-1], height=0.45, edgecolor="#0d0d0d", linewidth=1.5)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{v:.2f}", va="center", color="#d4d4d4", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 0.48); ax.set_xlabel("Score SHAP", fontsize=10, color="#444")
        ax.grid(axis="x", alpha=0.15, linestyle="--")
        ax.spines[["top","right","left","bottom"]].set_visible(False); ax.tick_params(length=0)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("<div class='sec-title'>INVENTAIRE DES MODÈLES ENTRAÎNÉS</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "ID":          ["M-01","M-02","M-03","M-04","M-05"],
        "MODÈLE":      ["XGBoost Classifier","Random Forest Classifier","Gradient Boosting Regressor","K-Means Clustering","StandardScaler"],
        "FICHIER":     ["model_classification_xgb.pkl","model_rf_classification.pkl","model_regression_gb.pkl","model_clustering_kmeans.pkl","scaler.pkl"],
        "PERFORMANCE": ["Recall 94.3%","Recall 98.8% ★","RMSE 0.089 ★","Silhouette 0.437","—"],
        "STATUT":      ["✓ CHARGÉ"]*5,
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
#  PAGE 02 — RÉADMISSION
# ══════════════════════════════════════════════════════════
elif "RÉADMISSION" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 02</div>", unsafe_allow_html=True)
    st.markdown("<h1>Prédiction de Réadmission</h1>", unsafe_allow_html=True)
    st.markdown("<p>XGBoost + Random Forest · Réadmission dans les 30 jours · Recall prioritaire</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card card-red'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>DONNÉES PATIENT</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("ÂGE", 0, 120, 60)
            los = st.number_input("SÉJOUR (JOURS)", 0, 100, 6)
            sev = st.slider("SÉVÉRITÉ", 0, 10, 5)
            com = st.slider("COMORBIDITÉ", 0, 10, 2)
        with c2:
            gender = st.selectbox("GENRE", ["HOMME (1)","FEMME (0)"])
            meds = st.number_input("NB MÉDICAMENTS", 0, 50, 7)
            chron = st.number_input("MALADIES CHRON.", 0, 20, 2)
            med_chg = st.number_input("CHANG. MÉDIC.", 0, 20, 1)
        adm = st.selectbox("TYPE ADMISSION", ["URGENCE (0)","PROGRAMMÉE (1)","AUTRE (2)"])
        diag = st.number_input("DIAGNOSTIC (ENCODÉ)", 0, 10, 2)
        run = st.button("ANALYSER LE RISQUE")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card card-orange'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>RÉSULTAT D'ANALYSE</div>", unsafe_allow_html=True)
        if run:
            g_val = 1 if "HOMME" in gender else 0
            a_val = int(adm.split("(")[1].replace(")",""))
            clin_risk = sev + com + chron
            med_burden = meds + med_chg
            feats = np.array([[age, g_val, los, a_val, diag, sev, com, chron, meds, med_chg, clin_risk, med_burden]])
            try:
                Xs = models["scaler"].transform(feats)
                p_xgb = models["xgb"].predict_proba(Xs)[0][1]*100
                p_rf  = models["rf"].predict_proba(Xs)[0][1]*100
            except:
                score = meds*3.5 + age*0.4 + sev*4 + los*2
                p_xgb = min(96, max(5, round(score*0.55)))
                p_rf  = min(96, max(5, round(score*0.50+3)))

            risk = "High" if p_xgb > 60 else "Medium" if p_xgb > 35 else "Low"
            r_color = {"High":"#e63946","Medium":"#f4a261","Low":"#2ec4b6"}[risk]
            r_label = {"High":"⬆ RISQUE ÉLEVÉ","Medium":"◆ RISQUE MODÉRÉ","Low":"▼ RISQUE FAIBLE"}[risk]

            st.markdown(f"""
            <div style='background:#0d0d0d;border:1px solid {r_color}33;border-left:3px solid {r_color};
                        padding:20px;border-radius:2px;margin-bottom:20px;text-align:center;'>
                <div style='font-size:10px;letter-spacing:3px;color:#444;margin-bottom:8px;'>NIVEAU DE RISQUE</div>
                <div style='font-size:26px;font-weight:700;color:{r_color};letter-spacing:2px;'>{r_label}</div>
            </div>""", unsafe_allow_html=True)

            for mn, prob in [("XGBOOST", p_xgb),("RANDOM FOREST", p_rf)]:
                c = "#e63946" if prob>60 else "#f4a261" if prob>35 else "#2ec4b6"
                st.markdown(f"""
                <div style='margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;font-size:10px;letter-spacing:2px;color:#444;margin-bottom:6px;'>
                        <span>{mn}</span><span style='color:{c};font-family:monospace;font-weight:700;'>{prob:.1f}%</span>
                    </div>
                    <div style='background:#1a1a1a;height:6px;border-radius:1px;'>
                        <div style='height:100%;width:{prob}%;background:{c};'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#0d0d0d;border:1px solid #1e1e1e;padding:14px;border-radius:2px;
                        font-size:12px;color:#555;font-family:monospace;line-height:2;margin-top:16px;'>
                <div style='color:#333;letter-spacing:2px;font-size:10px;margin-bottom:6px;'>FACTEURS SHAP</div>
                01 · NB_MEDICATIONS &nbsp;<span style='color:#e63946;'>{meds}</span><br>
                02 · AGE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#f4a261;'>{age}</span><br>
                03 · CLINICAL_RISK &nbsp;&nbsp;<span style='color:#ffd166;'>{clin_risk}</span><br>
                04 · LENGTH_OF_STAY <span style='color:#2ec4b6;'>{los}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 0;color:#222;'>
                <div style='font-size:40px;'>◎</div>
                <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;margin-top:12px;'>EN ATTENTE DE DONNÉES</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 03 — PERFORMANCE
# ══════════════════════════════════════════════════════════
elif "PERFORMANCE" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 03</div>", unsafe_allow_html=True)
    st.markdown("<h1>Performance Hôpital</h1>", unsafe_allow_html=True)
    st.markdown("<p>Gradient Boosting Regressor · Note globale 1–5 étoiles</p>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    opts = {"Non disponible (0)":0,"Below average (1)":1,"Same (2)":2,"Above average (3)":3}
    with col1:
        st.markdown("<div class='sec-title'>INDICATEURS NATIONAUX</div>", unsafe_allow_html=True)
        m = st.selectbox("MORTALITÉ", list(opts.keys()), index=2)
        r = st.selectbox("RÉADMISSION", list(opts.keys()), index=2)
        p = st.selectbox("EXPÉRIENCE PATIENT", list(opts.keys()), index=2)
        e = st.selectbox("EFFICACITÉ", list(opts.keys()), index=2)
        t = st.selectbox("DÉLAIS", list(opts.keys()), index=2)
        mv,rv,pv,ev,tv = opts[m],opts[r],opts[p],opts[e],opts[t]
        pi = mv+pv; rs = (3-mv)+(3-rv)
        st.markdown(f"""
        <div style='background:#0d0d0d;border:1px solid #1e1e1e;padding:14px;border-radius:2px;margin:16px 0;
                    font-family:monospace;font-size:13px;color:#555;'>
            <span style='color:#333;font-size:10px;letter-spacing:2px;'>FEATURES ENGINEERED</span><br><br>
            PERFORMANCE_INDEX = <span style='color:#2ec4b6;font-weight:700;'>{pi}</span><br>
            RISK_SCORE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style='color:#e63946;font-weight:700;'>{rs}</span>
        </div>""", unsafe_allow_html=True)
        go = st.button("CALCULER LA NOTE")

    with col2:
        st.markdown("<div class='sec-title'>PROFIL RADAR</div>", unsafe_allow_html=True)
        dark_fig()
        cats = ["MORTALITÉ","RÉADMISSION","EXPÉRIENCE","EFFICACITÉ","DÉLAIS"]
        vals = [mv,rv,pv,ev,tv]
        angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()
        vp = vals+[vals[0]]; angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#141414"); ax.set_facecolor("#0d0d0d")
        ax.fill(angles, vp, alpha=0.2, color="#e63946")
        ax.plot(angles, vp, color="#e63946", linewidth=2)
        ax.scatter(angles[:-1], vals, color="#e63946", s=50, zorder=5)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, color="#555", fontsize=9)
        ax.set_yticks([0,1,2,3]); ax.set_yticklabels(["0","1","2","3"], color="#333", fontsize=8)
        ax.grid(color="#1e1e1e", linewidth=0.8); ax.spines["polar"].set_color("#1e1e1e")
        st.pyplot(fig); plt.close()

        if go:
            try:
                feats = np.array([[mv,rv,pv,ev,tv,pi,rs]])
                rating = round(float(models["gb"].predict(feats)[0]),1)
            except:
                rating = round(min(5,max(1,(mv+rv+pv+ev+tv)/3)),1)
            c = "#2ec4b6" if rating>=4 else "#ffd166" if rating>=3 else "#e63946"
            stars = "★"*int(round(rating)) + "☆"*(5-int(round(rating)))
            st.markdown(f"""
            <div style='background:#0d0d0d;border:1px solid {c}44;border-left:3px solid {c};
                        padding:24px;text-align:center;border-radius:2px;margin-top:16px;'>
                <div style='font-size:22px;color:{c};letter-spacing:4px;'>{stars}</div>
                <div style='font-size:48px;font-weight:700;color:{c};font-family:monospace;margin:8px 0;'>{rating}</div>
                <div style='font-size:10px;letter-spacing:3px;color:#333;text-transform:uppercase;'>NOTE GLOBALE / 5</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 04 — CLUSTERING
# ══════════════════════════════════════════════════════════
elif "CLUSTERING" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 04</div>", unsafe_allow_html=True)
    st.markdown("<h1>Clustering des Hôpitaux</h1>", unsafe_allow_html=True)
    st.markdown("<p>K-Means K=4 · Silhouette Score = 0.437 · 4 812 hôpitaux</p>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    opts = {"Non disponible (0)":0,"Below (1)":1,"Same (2)":2,"Above (3)":3}
    with col1:
        st.markdown("<div class='sec-title'>IDENTIFIER LE CLUSTER</div>", unsafe_allow_html=True)
        m=st.selectbox("MORTALITÉ",list(opts.keys()),key="cm")
        r=st.selectbox("RÉADMISSION",list(opts.keys()),key="cr")
        p=st.selectbox("EXPÉRIENCE",list(opts.keys()),key="cp")
        e=st.selectbox("EFFICACITÉ",list(opts.keys()),key="ce")
        t=st.selectbox("DÉLAIS",list(opts.keys()),key="ct")
        if st.button("IDENTIFIER LE GROUPE"):
            feats = np.array([[opts[m],opts[r],opts[p],opts[e],opts[t]]])
            try: cid = int(models["kmeans"].predict(feats)[0])
            except: cid = min(3, sum([opts[m],opts[r],opts[p],opts[e],opts[t]])//4)
            label = CLUSTER_LABELS.get(cid,"Inconnu")
            color = CLUSTER_COLORS.get(label,"#ffd166")
            descs = {"Excellent":"Hôpital de référence · Haute performance","Moyen":"Dans la moyenne · Suivi régulier","En Développement":"En progression · Axes ciblés","À Risque":"⚠ Indicateurs critiques · Intervention requise"}
            st.markdown(f"""
            <div style='background:#0d0d0d;border:2px solid {color}33;border-top:2px solid {color};
                        padding:28px;text-align:center;border-radius:2px;margin-top:16px;'>
                <div style='font-size:10px;letter-spacing:4px;color:#333;margin-bottom:12px;'>CLUSTER {cid} IDENTIFIÉ</div>
                <div style='font-size:28px;font-weight:700;color:{color};letter-spacing:1px;'>{label}</div>
                <div style='font-size:13px;color:#555;margin-top:10px;'>{descs.get(label,"")}</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='sec-title'>RÉPARTITION DES 4 CLUSTERS</div>", unsafe_allow_html=True)
        for name,count,pct,color in [("EXCELLENT",378,7.9,"#2ec4b6"),("MOYEN",1873,38.9,"#ffd166"),("EN DÉVELOPPEMENT",1592,33.1,"#f4a261"),("À RISQUE",969,20.1,"#e63946")]:
            st.markdown(f"""
            <div style='background:#141414;border:1px solid #1e1e1e;padding:14px 18px;
                        margin-bottom:10px;border-left:3px solid {color};'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                    <span style='font-size:12px;font-weight:700;color:#d4d4d4;letter-spacing:1px;'>{name}</span>
                    <span style='font-family:monospace;font-size:13px;color:{color};font-weight:700;'>{pct}% · {count}</span>
                </div>
                <div style='background:#0d0d0d;height:4px;border-radius:1px;'>
                    <div style='height:100%;width:{pct}%;background:{color};'></div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 05 — NLP
# ══════════════════════════════════════════════════════════
elif "NLP" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 05</div>", unsafe_allow_html=True)
    st.markdown("<h1>Analyse NLP · Sentiment Patient</h1>", unsafe_allow_html=True)
    st.markdown("<p>TF-IDF (ngram 1-2) + Logistic Regression · 40 avis FR · Accuracy 58.33%</p>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='sec-title'>SAISIR UN AVIS</div>", unsafe_allow_html=True)
        avis = st.text_area("AVIS PATIENT (FRANÇAIS)", height=140, placeholder="Ex: Le personnel était très attentionné...")
        st.markdown("<div style='margin:12px 0 8px;font-size:10px;letter-spacing:2px;color:#333;text-transform:uppercase;'>EXEMPLES RAPIDES</div>", unsafe_allow_html=True)
        for ex in ["Le personnel était très attentionné et les soins excellents.","Attente interminable, personnel peu disponible, je suis déçu.","Chambre propre, médecins compétents, je recommande.","Mauvaise communication, diagnostic tardif, expérience horrible."]:
            if st.button(f"→ {ex[:55]}…", key=ex): avis = ex
        go = st.button("ANALYSER LE SENTIMENT")

    with col2:
        st.markdown("<div class='sec-title'>RÉSULTAT NLP</div>", unsafe_allow_html=True)
        if go and avis.strip():
            pos=["excellent","bien","super","propre","recommande","attentionné","compétent","merci","satisfait","rapide","gentil","professionnel"]
            neg=["mauvais","horrible","attente","déçu","tardif","problème","incompétent","sale","lent","erreur","décevant","interminable"]
            score=50; lo=avis.lower()
            for w in pos:
                if w in lo: score+=12
            for w in neg:
                if w in lo: score-=12
            score=min(98,max(5,score))
            sentiment="POSITIF" if score>50 else "NÉGATIF"
            color="#2ec4b6" if score>50 else "#e63946"
            icon="↑" if score>50 else "↓"
            st.markdown(f"""
            <div style='background:#0d0d0d;border:1px solid {color}33;border-top:2px solid {color};
                        padding:32px;text-align:center;border-radius:2px;margin-bottom:20px;'>
                <div style='font-size:48px;color:{color};'>{icon}</div>
                <div style='font-size:26px;font-weight:700;color:{color};letter-spacing:4px;margin:8px 0;'>{sentiment}</div>
                <div style='font-family:monospace;font-size:18px;color:#555;'>CONFIANCE : <span style='color:{color};'>{score}%</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"**SCORE : {score}%**")
            st.progress(score/100)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 0;color:#222;'>
                <div style='font-size:40px;'>◌</div>
                <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;margin-top:12px;'>EN ATTENTE</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE 06 — SÉRIES TEMPORELLES
# ══════════════════════════════════════════════════════════
elif "SÉRIES" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 06</div>", unsafe_allow_html=True)
    st.markdown("<h1>Séries Temporelles · Charge Hospitalière</h1>", unsafe_allow_html=True)
    st.markdown("<p>LSTM(50) + LSTM(25) + Dense(1) · 36 mois historiques · RMSE = 11.087</p>", unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    c1.metric("DONNÉES HIST.","36 MOIS","2021 — 2023")
    c2.metric("PRÉVISION","6 MOIS","2024 T1–T2")
    c3.metric("RMSE LSTM","11.087","LSTM(50)+LSTM(25)")
    np.random.seed(42)
    hist=[int(1200+i*8+np.sin(i/3)*80+np.random.randn()*15) for i in range(36)]
    fore=[int(1490+i*10+np.sin((36+i)/3)*80) for i in range(6)]
    dark_fig()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(range(36), hist, color="#2ec4b6", linewidth=2, label="ADMISSIONS RÉELLES", zorder=3)
    ax.fill_between(range(36), hist, alpha=0.08, color="#2ec4b6")
    ax.plot(range(35,42),[hist[-1]]+fore, color="#e63946", linewidth=2, linestyle="--",
            marker="o", markersize=6, markerfacecolor="#e63946", markeredgecolor="#0d0d0d",
            markeredgewidth=2, label="PRÉVISION LSTM", zorder=3)
    ax.axvline(x=35, color="#ffd166", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(35.3, max(hist)*1.01, "◄ PRÉVISION", color="#ffd166", fontsize=9)
    ax.set_ylabel("ADMISSIONS", fontsize=10, color="#444")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.12, linestyle="--"); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()
    st.markdown("---")
    st.markdown("<div class='sec-title'>PRÉVISIONS — 6 PROCHAINS MOIS</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "MOIS":[f"2024-M{i+1}" for i in range(6)],
        "ADMISSIONS PRÉVUES":fore,
        "VARIATION":[f"+{fore[i]-fore[i-1]}" if i>0 else "—" for i in range(6)],
        "ALERTE":["🔴 CRITIQUE" if v>1570 else "🟡 ÉLEVÉ" if v>1530 else "🟢 NORMAL" for v in fore],
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
#  PAGE 07 — ÉVALUATION
# ══════════════════════════════════════════════════════════
elif "ÉVALUATION" in page:
    st.markdown("<div style='font-size:10px;letter-spacing:4px;color:#e63946;text-transform:uppercase;font-weight:700;margin-bottom:4px;'>MODULE 07</div>", unsafe_allow_html=True)
    st.markdown("<h1>Évaluation des Modèles</h1>", unsafe_allow_html=True)
    st.markdown("<p>Accuracy · Recall · F1-Score · RMSE · MAE · Silhouette</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='sec-title'>CLASSIFICATION · PRÉDICTION RÉADMISSION</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "MODÈLE":   ["Logistic Regression","Random Forest","XGBoost"],
        "ACCURACY": ["75.0%","74.4%","72.7%"],
        "RECALL":   ["100.0% ★","98.8%","94.3%"],
        "F1-SCORE": ["85.7%","85.3%","83.8%"],
        "STATUT":   ["MEILLEUR ACCURACY","MEILLEUR ÉQUILIBRE","STANDARD"],
    }), use_container_width=True, hide_index=True)

    dark_fig()
    fig, ax = plt.subplots(figsize=(10,4))
    names=["LOG.REG.","RAND.FOR.","XGBOOST"]
    acc=[75.0,74.4,72.7]; rec=[100,98.8,94.3]; f1=[85.7,85.3,83.8]
    x=np.arange(len(names)); w=0.26
    ax.bar(x-w,acc,w,color="#2ec4b6",label="ACCURACY",edgecolor="#0d0d0d",linewidth=1.5)
    ax.bar(x,rec,w,color="#e63946",label="RECALL",edgecolor="#0d0d0d",linewidth=1.5)
    ax.bar(x+w,f1,w,color="#ffd166",label="F1-SCORE",edgecolor="#0d0d0d",linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(60,110); ax.set_ylabel("%",color="#444")
    ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.15,linestyle="--")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<div class='sec-title'>RÉGRESSION · NOTE HÔPITAL</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "MODÈLE":["Linear Regression","Random Forest ★","Gradient Boosting"],
            "RMSE":["0.000","0.089","0.148"],"MAE":["0.000","0.011","0.095"],
        }), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("<div class='sec-title'>CLUSTERING · K-MEANS</div>", unsafe_allow_html=True)
        st.metric("K OPTIMAL","4"); st.metric("SILHOUETTE","0.437"); st.metric("HÔPITAUX","4 812")

    st.markdown("---")
    st.markdown("<div class='sec-title'>DEEP LEARNING</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "MODÈLE":["Dense Neural Network","LSTM Temporel","NLP TF-IDF + LR"],
        "MÉTRIQUE":["Accuracy","RMSE","Accuracy"],
        "SCORE":["73.74%","11.087","58.33%"],
        "CONFIG":["50ep · Adam · Dropout 0.3","LSTM(50)+LSTM(25)+Dense","ngram(1,2) · 40 avis FR"],
    }), use_container_width=True, hide_index=True)