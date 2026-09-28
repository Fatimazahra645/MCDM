import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import weights as W

st.set_page_config(page_title="MCDM Rose", page_icon="🌸", layout="wide")

# ---------- Thème rose & fleurs ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#fff0f5 0%,#ffe0ec 55%,#ffd1e3 100%);}
h1,h2,h3 {color:#c2185b !important;}
.hero {background:linear-gradient(90deg,#f8bbd0,#f48fb1); border-radius:22px;
  padding:22px 30px; text-align:center; box-shadow:0 6px 18px rgba(233,30,99,.25);}
.hero h1 {color:#fff !important; margin:0; font-size:2.3rem;}
.hero p {color:#fff; margin:4px 0 0 0;}
.card {background:#fff; border-radius:18px; padding:16px 20px; margin:10px 0;
  border:2px solid #f8bbd0; box-shadow:0 4px 12px rgba(233,30,99,.12);}
.stButton>button {background:#e75480; color:#fff; border-radius:20px; border:0; padding:.5rem 1.4rem;}
.stButton>button:hover {background:#c2185b; color:#fff;}
[data-testid="stSidebar"] {background:#ffd6e7;}
.flower {position:fixed; top:-40px; font-size:26px; opacity:.55; z-index:0;
  animation:fall linear infinite; pointer-events:none;}
@keyframes fall {to {transform:translateY(115vh) rotate(360deg);}}
.footer {text-align:center; color:#c2185b; font-weight:600; margin-top:30px;}
</style>
""" + "".join(
    f'<div class="flower" style="left:{l}%;animation-duration:{d}s;animation-delay:{dl}s">{f}</div>'
    for l, d, dl, f in [(5, 14, 0, "🌸"), (18, 18, 3, "🌷"), (32, 16, 6, "🌺"), (47, 20, 1, "🌸"),
                        (61, 15, 8, "💮"), (74, 19, 4, "🌷"), (88, 17, 2, "🌸"), (95, 21, 9, "🌺")]
), unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🌸 Aide à la décision multicritère 🌸</h1>'
            '<p>Pondération des critères &amp; classement des alternatives</p></div>',
            unsafe_allow_html=True)

# ---------- Étape 1 : données ----------
st.sidebar.markdown("## 🌷 Navigation")
st.sidebar.info("1️ Données  →  2️ Pondération  →  3️ Classement")

st.header("1️_Matrice de décision")
src = st.radio("Source des données", ["Exemple", "Importer CSV/Excel", "Saisie manuelle"], horizontal=True)

if src == "Exemple":
    df = pd.DataFrame({"Prix (MAD)": [250, 200, 300, 275, 225],
                       "Qualité": [16, 16, 32, 32, 16],
                       "Autonomie (h)": [12, 8, 16, 8, 16],
                       "Poids (kg)": [1.8, 2.2, 1.5, 2.0, 1.7]},
                      index=["A1", "A2", "A3", "A4", "A5"])
elif src == "Importer CSV/Excel":
    f = st.file_uploader("Fichier (1re colonne = noms des alternatives)", type=["csv", "xlsx"])
    if f is None:
        st.stop()
    df = pd.read_csv(f, index_col=0) if f.name.endswith(".csv") else pd.read_excel(f, index_col=0)
else:
    nA = st.number_input("Nombre d'alternatives", 2, 30, 4)
    nC = st.number_input("Nombre de critères", 2, 10, 3)
    df = pd.DataFrame(np.ones((nA, nC)), index=[f"A{i+1}" for i in range(nA)],
                      columns=[f"C{j+1}" for j in range(nC)])

df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
df = df.apply(pd.to_numeric, errors="coerce").dropna()
if df.shape[0] < 2 or df.shape[1] < 2:
    st.warning("Il faut au moins 2 alternatives et 2 critères numériques.")
    st.stop()

st.markdown("**Type de chaque critère**")
cols = st.columns(len(df.columns))
types = [c.selectbox(name, ["Bénéfice ⬆", "Coût ⬇"], key=f"t_{name}")
         for c, name in zip(cols, df.columns)]
benefit = np.array([t.startswith("Bénéfice") for t in types])
X = df.values.astype(float)
n = X.shape[1]

st.session_state["df"] = df
st.session_state["benefit"] = benefit

# ---------- Étape 2 : pondération ----------
st.header("2️_Pondération des critères")
method = st.selectbox("Méthode de pondération", ["CRITIC", "Entropie", "AHP", "BWM"])
w, info = None, ""

if method == "CRITIC":
    w = W.critic(X, benefit)
elif method == "Entropie":
    w = W.entropy(X, benefit)
elif method == "AHP":
    st.caption("Échelle de Saaty (1 à 9). Remplissez le triangle supérieur : ligne i vs colonne j.")
    base = pd.DataFrame(np.ones((n, n)), index=df.columns, columns=df.columns)
    M = st.data_editor(base, use_container_width=True, key="ahp")
    w, cr = W.ahp(W.ahp_from_upper(M.values))
    info = f"Ratio de cohérence CR = **{cr:.3f}** " + ("✅ (acceptable)" if cr < 0.1 else "⚠️ (> 0,1 : revoir les comparaisons)")
elif method == "BWM":
    c1, c2 = st.columns(2)
    best = c1.selectbox("Meilleur critère", df.columns, index=0)
    worst = c2.selectbox("Pire critère", df.columns, index=n - 1)
    ib, iw = list(df.columns).index(best), list(df.columns).index(worst)
    if ib == iw:
        st.warning("Le meilleur et le pire critère doivent être différents.")
        st.stop()
    st.caption("Préférence du meilleur sur chaque critère, et de chaque critère sur le pire (1 à 9).")
    t = pd.DataFrame({"Meilleur→Autres": [1.0] * n, "Autres→Pire": [1.0] * n}, index=df.columns)
    t = st.data_editor(t, use_container_width=True, key="bwm")
    bo, ow = t["Meilleur→Autres"].values.copy(), t["Autres→Pire"].values.copy()
    bo[ib], ow[iw] = 1, 1
    w, xi = W.bwm(ib, iw, bo, ow)
    info = f"ξ* = **{xi:.4f}** (plus il est proche de 0, plus les jugements sont cohérents)"

if info:
    st.markdown(f'<div class="card">{info}</div>', unsafe_allow_html=True)

res = pd.DataFrame({"Critère": df.columns, "Poids": w}).round(4)
a, b = st.columns([1, 2])
a.dataframe(res, hide_index=True, use_container_width=True)
fig = px.bar(res, x="Critère", y="Poids", color="Poids", color_continuous_scale="Pinkyl", text="Poids")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.7)")
b.plotly_chart(fig, use_container_width=True)
st.session_state["weights"] = np.asarray(w, dtype=float)
st.session_state["weight_method"] = method

# ---------- Étape 3 : classement (partie 2) ----------
st.header("3️_Classement des alternatives")
try:
    import ranking
    ranking.render(df, benefit, st.session_state["weights"])
except ImportError:
    st.info("🌷 Classement indisponible (ranking.py manquant).")

st.markdown('<div class="footer">🌸 Réalisé par Fatima Zahra KAKA 🌸</div>', unsafe_allow_html=True)
