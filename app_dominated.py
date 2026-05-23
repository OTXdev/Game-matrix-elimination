from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dominated_strategies import iterated_elimination

st.set_page_config(
    page_title="IESDS — Théorie des Jeux",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<div style="padding:2.5rem 0 2rem; border-bottom:1px solid #d4c9b8; margin-bottom:2rem;">
  <div class="page-title"> <h1> IESDS — <em>Domination Mixte</em> </h1> </div>
  <div class="page-subtitle"> <h4> Realisé par Asma Otsmane et Emna Saci | Eliminate Framework </h4> </div>
  <div class="page-subtitle">Élimination Itérative des Stratégies Strictement Dominées · Deux joueurs · Chacun maximise</div>


            
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-label">00 — Configuration</div>', unsafe_allow_html=True)

col_config1, col_config2 = st.columns([1, 1], gap="large")

PRESETS = {
    "Série TD— Exo 7 (3×3)": {
        "rows": ["A", "B", "C"], "cols": ["L", "M", "R"],
        "u1": "3,2,2\n5,1,2\n9,1,3",
        "u2": "5,0,2\n2,2,1\n0,5,2",
    },
    "Dilemme du prisonnier (2×2)": {
        "rows": ["Avoue", "N'avoue pas"],
        "cols": ["Avoue", "N'avoue pas"],
        "u1": "-5,0\n-10,-1",
        "u2": "-5,-10\n0,-1",
    },
    "Pierre Feuille Ciseaux (3×3)": {
        "rows": ["Pierre", "Feuille", "Ciseaux"],
        "cols": ["Pierre", "Feuille", "Ciseaux"],
        "u1": "0,1,-1\n-1,0,1\n1,-1,0",
        "u2": "0,-1,1\n1,0,-1\n-1,1,0",
    },
    "Coordination (2×2)": {
        "rows": ["A", "B"], "cols": ["A", "B"],
        "u1": "1,0\n0,1",
        "u2": "1,0\n0,1",
    },
}

with col_config1:
    preset = st.selectbox("Exemple prédéfini", [
        "Personnalisé",
        "Série TD — Exo 7 (3×3)",
        "Dilemme du prisonnier (2×2)",
        "Pierre Feuille Ciseaux (3×3)",
        "Coordination (2×2)",
    ])

    # Use preset-specific defaults so sliders reset when preset changes
    if preset in PRESETS:
        p = PRESETS[preset]
        n_rows_def = len(p["rows"])
        n_cols_def = len(p["cols"])
    else:
        n_rows_def = 3
        n_cols_def = 3

    # Include preset name in key so slider resets when preset changes
    n_rows = st.slider("Nombre de stratégies Joueur 1", 2, 6, n_rows_def,
                       key=f"nr_{preset}")
    n_cols = st.slider("Nombre de stratégies Joueur 2", 2, 6, n_cols_def,
                       key=f"nc_{preset}")

with col_config2:
    st.markdown("**Noms des stratégies**")

    if preset in PRESETS:
        row_defaults = PRESETS[preset]["rows"]
        col_defaults = PRESETS[preset]["cols"]
    else:
        row_defaults = [f"S{i+1}" for i in range(n_rows)]
        col_defaults = [f"T{j+1}" for j in range(n_cols)]

    row_labels = []
    for i in range(n_rows):
        default = row_defaults[i] if i < len(row_defaults) else f"S{i+1}"
        # key includes preset so widgets refresh when preset changes
        row_labels.append(st.text_input(f"Joueur 1 — Stratégie {i+1}", default,
                                        key=f"rl_{preset}_{i}"))

    col_labels = []
    for j in range(n_cols):
        default = col_defaults[j] if j < len(col_defaults) else f"T{j+1}"
        col_labels.append(st.text_input(f"Joueur 2 — Stratégie {j+1}", default,
                                        key=f"cl_{preset}_{j}"))

# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">01 — Saisie de la matrice</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card-info">
  <strong>Format :</strong> entrez séparément les gains de chaque joueur.<br>
  Ligne <em>i</em>, colonne <em>j</em> : u1[i,j] = gain joueur 1 ; u2[i,j] = gain joueur 2.
</div>
""", unsafe_allow_html=True)

col_u1, col_u2 = st.columns(2, gap="large")

with col_u1:
    st.markdown("**Gains Joueur 1 (u₁)**")
    default_u1 = PRESETS[preset]["u1"] if preset in PRESETS else \
        "\n".join(",".join(["0"] * n_cols) for _ in range(n_rows))
    # key includes preset so textarea resets when preset changes
    u1_text = st.text_area("u1 — une ligne par stratégie, valeurs séparées par virgule",
                            value=default_u1, height=160,
                            key=f"u1_area_{preset}",
                            label_visibility="collapsed")

with col_u2:
    st.markdown("**Gains Joueur 2 (u₂)**")
    default_u2 = PRESETS[preset]["u2"] if preset in PRESETS else \
        "\n".join(",".join(["0"] * n_cols) for _ in range(n_rows))
    u2_text = st.text_area("u2 — une ligne par stratégie, valeurs séparées par virgule",
                            value=default_u2, height=160,
                            key=f"u2_area_{preset}",
                            label_visibility="collapsed")

def parse_matrix(text, n_rows, n_cols):
    try:
        rows = [list(map(float, r.strip().split(",")))
                for r in text.strip().splitlines() if r.strip()]
        M = np.array(rows)
        if M.shape != (n_rows, n_cols):
            return None, f"Taille attendue {n_rows}×{n_cols}, reçue {M.shape}"
        return M, None
    except Exception as e:
        return None, str(e)

u1_mat, err1 = parse_matrix(u1_text, n_rows, n_cols)
u2_mat, err2 = parse_matrix(u2_text, n_rows, n_cols)

if err1 or err2:
    if err1:
        st.error(f"u1 : {err1}")
    if err2:
        st.error(f"u2 : {err2}")
    st.stop()

payoffs = np.stack([u1_mat, u2_mat], axis=2)

st.markdown('<div class="sec-label">02 — Matrice de jeu</div>', unsafe_allow_html=True)
df_preview = pd.DataFrame(
    [[f"({u1_mat[i,j]:.1f}, {u2_mat[i,j]:.1f})" for j in range(n_cols)]
     for i in range(n_rows)],
    index=row_labels, columns=col_labels
)
st.dataframe(df_preview, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
if st.button("▶  Lancer l'élimination itérative", type="primary", use_container_width=True):
    result = iterated_elimination(payoffs, row_labels, col_labels, verbose=False)
    st.session_state["result"] = result
    st.session_state["original_payoffs"] = payoffs
    st.session_state["original_row"] = row_labels
    st.session_state["original_col"] = col_labels

# ─────────────────────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.stop()

result   = st.session_state["result"]
log      = result["elimination_log"]
R        = result["payoffs"]
r_lab    = result["row_labels"]
c_lab    = result["col_labels"]

st.markdown("---")
st.markdown('<div class="sec-label">03 — Résultats</div>', unsafe_allow_html=True)

rows_elim = sum(1 for e in log if e["player"] == "Joueur 1")
cols_elim = sum(1 for e in log if e["player"] == "Joueur 2")
badge = (' <strong>Solution unique</strong>' if result["solved"]
         else ' <strong>Réduction partielle</strong>')

st.markdown(f"""
<div class="card" style="display:flex;align-items:center;
    justify-content:space-between;flex-wrap:wrap;gap:1rem;">
  <div>{badge}</div>
  <div style="display:flex;gap:2rem;font-family:'DM Mono',monospace;
      font-size:0.82rem;color:#5a5a5a;">
    <span><strong style="color:#1a1a1a;">{len(log)}</strong> élimination(s)</span>
    <span><strong style="color:#c0392b;">{rows_elim}</strong> ligne(s) supprimée(s)</span>
    <span><strong style="color:#2d4f7a;">{cols_elim}</strong> colonne(s) supprimée(s)</span>
    <span>Final : <strong>{R.shape[0]}×{R.shape[1]}</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_steps, tab_matrix, tab_theory = st.tabs(
    ["Étapes d'élimination", "Matrice réduite", "Théorie"])

with tab_steps:
    if not log:
        st.markdown("""
<div class="card-warn">
  <strong>Aucune stratégie dominée trouvée.</strong><br>
  <span style="font-size:0.85rem;color:#5a5a5a;">
  La matrice est déjà irréductible — aucune stratégie pure n'est strictement
  dominée par une stratégie mixte.
  </span>
</div>""", unsafe_allow_html=True)
    else:
        for idx, entry in enumerate(log):
            player   = entry["player"]
            elim     = entry["eliminated"]
            mixture  = entry["dominated_by"]
            badge_cl = "badge-j1" if player == "Joueur 1" else "badge-j2"
            mix_str  = " + ".join(
                f"<strong>{prob:.3f}</strong>·<em>{s}</em>"
                for s, prob in mixture.items()
            )
            st.markdown(f"""
<div class="step-card">
  <div class="step-num">#{idx+1}</div>
  <div style="flex:1;">
    <div style="margin-bottom:0.4rem;">
      <span class="{badge_cl}">{player}</span>
    </div>
    <div>Stratégie <strong style="color:#c0392b;">{elim}</strong> éliminée</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#5a5a5a;
        margin-top:0.3rem;">Dominée par : {mix_str}</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        orig_P   = st.session_state["original_payoffs"]
        orig_row = st.session_state["original_row"]
        orig_col = st.session_state["original_col"]
        elim_rows = [e["eliminated"] for e in log if e["player"] == "Joueur 1"]
        elim_cols = [e["eliminated"] for e in log if e["player"] == "Joueur 2"]

        fig, ax = plt.subplots(figsize=(max(4, len(orig_col)*1.4),
                                        max(3, len(orig_row)*1.0)))
        fig.patch.set_facecolor("#f5f2ed")
        ax.set_facecolor("#f5f2ed")
        n_r, n_c = len(orig_row), len(orig_col)
        for i in range(n_r):
            for j in range(n_c):
                is_er = orig_row[i] in elim_rows
                is_ec = orig_col[j] in elim_cols
                color = ("#fdecea" if is_er else
                         "#eaf1fb" if is_ec else "#ffffff")
                ax.add_patch(plt.Rectangle((j, n_r-i-1), 1, 1,
                             facecolor=color, edgecolor="#d4c9b8", linewidth=0.8))
                u1v = orig_P[i, j, 0]; u2v = orig_P[i, j, 1]
                ax.text(j + 0.5, n_r - i - 0.5,
                        f"({u1v:.1f}, {u2v:.1f})",
                        ha="center", va="center", fontsize=9,
                        color=("#c0392b" if is_er else
                               "#2d4f7a" if is_ec else "#1a1a1a"),
                        fontweight="bold" if (is_er or is_ec) else "normal")
        ax.set_xlim(0, n_c); ax.set_ylim(0, n_r)
        ax.set_xticks(np.arange(n_c) + 0.5)
        ax.set_xticklabels(orig_col, fontsize=9)
        ax.set_yticks(np.arange(n_r) + 0.5)
        ax.set_yticklabels(orig_row[::-1], fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d4c9b8")
        patches = [mpatches.Patch(color="#fdecea", label="Lignes éliminées (J1)"),
                   mpatches.Patch(color="#eaf1fb", label="Colonnes éliminées (J2)")]
        ax.legend(handles=patches, loc="upper right", fontsize=8,
                  framealpha=0.9, edgecolor="#d4c9b8")
        ax.set_title("Matrice originale — stratégies éliminées colorées",
                     fontsize=9, color="#9a8c7a", pad=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

with tab_matrix:
    col_mat, col_info = st.columns([3, 2], gap="large")
    with col_mat:
        if R.size == 0:
            st.warning("Matrice vide après réduction.")
        else:
            df_red = pd.DataFrame(
                [[f"({R[i,j,0]:.2f}, {R[i,j,1]:.2f})" for j in range(len(c_lab))]
                 for i in range(len(r_lab))],
                index=r_lab, columns=c_lab
            )
            st.dataframe(df_red, use_container_width=True)
    with col_info:
        if result["solved"]:
            st.markdown(f"""
<div class="card-success">
  <strong>Solution unique trouvée</strong><br><br>
  <span style="font-family:'DM Mono',monospace;font-size:0.85rem;">
  Joueur 1 : <strong>{r_lab[0]}</strong><br>
  Joueur 2 : <strong>{c_lab[0]}</strong><br>
  u1 = <strong>{R[0,0,0]:.4f}</strong><br>
  u2 = <strong>{R[0,0,1]:.4f}</strong>
  </span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="card-warn">
  <strong>Réduction partielle</strong><br><br>
  <span style="font-size:0.85rem;color:#5a5a5a;font-family:'DM Mono',monospace;">
  Lignes restantes : {', '.join(r_lab)}<br><br>
  Colonnes restantes : {', '.join(c_lab)}<br><br>
  → Calculer l'équilibre de Nash sur la matrice réduite.
  </span>
</div>""", unsafe_allow_html=True)
        orig_cells = st.session_state["original_payoffs"].shape[0] * \
                     st.session_state["original_payoffs"].shape[1]
        red_cells  = R.shape[0] * R.shape[1]
        pct = (1 - red_cells / orig_cells) * 100
        st.markdown(f"""
<div class="card" style="font-family:'DM Mono',monospace;font-size:0.8rem;">
  Original : {st.session_state["original_payoffs"].shape[0]}×{st.session_state["original_payoffs"].shape[1]} = {orig_cells} cases<br>
  Réduit   : {R.shape[0]}×{R.shape[1]} = {red_cells} cases<br>
  Réduction : <strong style="color:#c0392b;">{pct:.0f}%</strong>
</div>""", unsafe_allow_html=True)
    with st.expander("Export CSV"):
        df_exp_u1 = pd.DataFrame(R[:,:,0], index=r_lab, columns=c_lab)
        df_exp_u2 = pd.DataFrame(R[:,:,1], index=r_lab, columns=c_lab)
        st.markdown("**u1**"); st.dataframe(df_exp_u1)
        st.markdown("**u2**"); st.dataframe(df_exp_u2)
        st.download_button("Télécharger u1", df_exp_u1.to_csv().encode(),
                            "u1_reduit.csv", "text/csv")
        st.download_button("Télécharger u2", df_exp_u2.to_csv().encode(),
                            "u2_reduit.csv", "text/csv")

with tab_theory:
    st.markdown("""
<div class="card">
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#9a8c7a;
      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
    Pourquoi les deux joueurs maximisent ?
  </div>
  <p style="font-size:0.88rem;color:#3a3a3a;line-height:1.8;margin:0;">
  Le joueur 1 choisit sa stratégie pour maximiser <em>u1</em>,
  le joueur 2 choisit sa stratégie pour maximiser <em>u2</em>.
  Il n'y a <strong>aucune hypothèse de jeu à somme nulle</strong>.
  </p>
</div>
<div class="card">
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#9a8c7a;
      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
    Condition de domination stricte
  </div>
  <p style="font-size:0.88rem;color:#3a3a3a;line-height:1.8;margin:0;">
  Une stratégie pure <strong>i</strong> du joueur 1 est <em>strictement dominée</em>
  par une mixture <strong>p</strong> si :<br><br>
  <code style="background:#f0ebe3;padding:0.3rem 0.6rem;border-radius:2px;">
    Σₖ p[k] · u1[k,j] > u1[i,j]   ∀ j
  </code><br><br>
  Idem pour le joueur 2 sur ses colonnes avec <em>u2</em>.
  </p>
</div>
<div class="card">
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#9a8c7a;
      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
    Algorithme IESDS
  </div>
  <p style="font-size:0.88rem;color:#3a3a3a;line-height:1.8;margin:0;">
  <strong>1.</strong> Scanner les lignes (J1). Si une ligne est dominée → l'éliminer, recommencer.<br>
  <strong>2.</strong> Scanner les colonnes (J2). Si une colonne est dominée → l'éliminer, recommencer.<br>
  <strong>3.</strong> Alterner jusqu'à ce qu'il n'y ait plus d'élimination possible.<br>
  <strong>4.</strong> Si la matrice est 1×1 → solution unique. Sinon → calculer Nash sur la matrice réduite.<br><br>
  <strong>Propriété clé :</strong> IESDS ne supprime jamais un équilibre de Nash.
  </p>
</div>
""", unsafe_allow_html=True)
    if log:
        st.markdown("**Détail des mélanges dominants**")
        mix_rows = []
        for idx, e in enumerate(log):
            mix_str = " + ".join(f"{v:.3f}·{k}" for k, v in e["dominated_by"].items())
            mix_rows.append({"#": idx+1, "Joueur": e["player"],
                             "Éliminée": e["eliminated"],
                             "Dominée par": mix_str})
        st.dataframe(pd.DataFrame(mix_rows), use_container_width=True, hide_index=True)