"""Partie 2 : classement (Pondération simple : WSM/WPM/WASPAS, et TOPSIS)."""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

EPS = 1e-12


def _ratio_norm(X, benefit):
    """Normalisation pour WSM/WPM/WASPAS : x/max (bénéfice), min/x (coût)."""
    X = np.asarray(X, dtype=float)
    mx, mn = X.max(axis=0), X.min(axis=0)
    safe = np.where(X == 0, EPS, X)
    return np.where(benefit, X / np.where(mx == 0, 1, mx), mn / safe)


def wsm(X, benefit, w):
    return (_ratio_norm(X, benefit) * w).sum(axis=1)


def wpm(X, benefit, w):
    R = np.clip(_ratio_norm(X, benefit), EPS, None)
    return np.prod(R ** w, axis=1)


def waspas(X, benefit, w, lam=0.5):
    return lam * wsm(X, benefit, w) + (1 - lam) * wpm(X, benefit, w)


def topsis(X, benefit, w):
    X = np.asarray(X, dtype=float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    V = X / np.where(norm == 0, 1, norm) * w
    ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
    anti = np.where(benefit, V.min(axis=0), V.max(axis=0))
    dp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((V - anti) ** 2).sum(axis=1))
    return dm / np.where(dp + dm == 0, 1, dp + dm), dp, dm


def render(df, benefit, weights):
    X = df.values.astype(float)
    w = np.asarray(weights, dtype=float)
    benefit = np.asarray(benefit, dtype=bool)

    family = st.radio("Méthode de classement", ["Pondération simple", "TOPSIS"], horizontal=True)
    out = pd.DataFrame(index=df.index)

    if family == "Pondération simple":
        sub = st.selectbox("Variante", ["WSM (somme pondérée)", "WPM (produit pondéré)", "WASPAS"])
        if sub.startswith("WSM"):
            out["Score"], name = wsm(X, benefit, w), "WSM"
        elif sub.startswith("WPM"):
            out["Score"], name = wpm(X, benefit, w), "WPM"
        else:
            lam = st.slider("λ (part de WSM)", 0.0, 1.0, 0.5, 0.05)
            out["WSM"], out["WPM"] = wsm(X, benefit, w), wpm(X, benefit, w)
            out["Score"], name = lam * out["WSM"] + (1 - lam) * out["WPM"], f"WASPAS (λ={lam})"
    else:
        c, dp, dm = topsis(X, benefit, w)
        out["D+"], out["D-"], out["Score"], name = dp, dm, c, "TOPSIS"

    out["Rang"] = out["Score"].rank(ascending=False, method="min").astype(int)
    out = out.sort_values("Rang")
    out.index.name = "Alternative"
    res = out.reset_index().round(4)
    cols = ["Rang", "Alternative"] + [c for c in res.columns if c not in ("Rang", "Alternative")]
    res = res[cols]

    best = res.iloc[0]
    st.markdown(f'<div class="card">🏆 Meilleure alternative ({name}) : <b>{best["Alternative"]}</b> '
                f'— score {best["Score"]:.4f}</div>', unsafe_allow_html=True)

    a, b = st.columns([1, 2])
    a.dataframe(res, hide_index=True, use_container_width=True)
    fig = px.bar(res, x="Alternative", y="Score", color="Score",
                 color_continuous_scale="Pinkyl", text="Score")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.7)")
    b.plotly_chart(fig, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button("🌸 Télécharger en CSV", res.to_csv(index=False).encode("utf-8-sig"),
                       "classement.csv", "text/csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        res.to_excel(xw, index=False, sheet_name="Classement")
        pd.DataFrame({"Critère": df.columns, "Poids": w,
                      "Type": np.where(benefit, "Bénéfice", "Coût")}).to_excel(
            xw, index=False, sheet_name="Poids")
    d2.download_button("🌷 Télécharger en Excel", buf.getvalue(), "classement.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
