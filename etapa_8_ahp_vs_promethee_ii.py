import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_ahp = pd.read_csv("tabela_final_com_metodo.csv")

if "Custo Total" in df_ahp.columns:
    df_ahp["Custo"] = df_ahp["Custo Total"]
    df_ahp["Equilibrio"] = df_ahp["Equilibrio"]
else:
    df_ahp["Custo"] = df_ahp["f1"]
    df_ahp["Equilibrio"] = df_ahp["f2"]

if "origem" not in df_ahp.columns and "metodo" in df_ahp.columns:
    df_ahp["origem"] = df_ahp["metodo"]

score_col_ahp = "Score"
if score_col_ahp not in df_ahp.columns:
    if "Score_Global" in df_ahp.columns:
        score_col_ahp = "Score_Global"
    else:
        raise ValueError("Não encontrei coluna de Score no AHP (nem 'Score' nem 'Score_Global').")

df_prom = pd.read_csv("tabela_promethee_II.csv")

df_prom["Custo"] = df_prom["Custo Total"]
df_prom["Equilibrio"] = df_prom["Equilibrio"]

best_ahp = df_ahp.loc[df_ahp[score_col_ahp].idxmax()]
best_prom = df_prom.loc[df_prom["phi"].idxmax()]

def make_key(df):
    return df["origem"].astype(str) + "_" + df["Custo"].astype(str) + "_" + df["Equilibrio"].astype(str)

df_ahp["key"] = make_key(df_ahp)
df_prom["key"] = make_key(df_prom)

set_ahp = set(df_ahp["key"])
set_prom = set(df_prom["key"])

common_keys = set_ahp.intersection(set_prom)

df_ahp["is_common"] = df_ahp["key"].isin(common_keys)
df_prom["is_common"] = df_prom["key"].isin(common_keys)

df_full = None
try:
    df_full = pd.read_csv("fronteira_unificada_com_criterios.csv")
except Exception:
    pass

plt.figure(figsize=(11, 8))

style_metodo = {
    'Pe': {'c': 'tab:blue',   'm': 'o', 'lbl': 'Pe (Epsilon)'},
    'Pw': {'c': 'tab:orange', 'm': '^', 'lbl': 'Pw (Soma Pond.)'}
}

if df_full is not None:
    for m in ['Pe', 'Pw']:
        subset = df_full[df_full['metodo'] == m]
        if not subset.empty:
            plt.scatter(
                subset['f1'], subset['f2'],
                c=style_metodo[m]['c'],
                marker=style_metodo[m]['m'],
                alpha=0.10, s=25,
                label=f"Fronteira completa {style_metodo[m]['lbl']}"
            )

for m in ['Pe', 'Pw']:
    subset = df_ahp[(df_ahp['origem'] == m) & (~df_ahp['is_common'])]
    if not subset.empty:
        plt.scatter(
            subset['Custo'], subset['Equilibrio'],
            c=style_metodo[m]['c'],
            marker=style_metodo[m]['m'],
            s=90, edgecolors='k', alpha=0.9,
            label=f"Analisadas {style_metodo[m]['lbl']} – AHP"
        )

for m in ['Pe', 'Pw']:
    subset = df_prom[(df_prom['origem'] == m) & (~df_prom['is_common'])]
    if not subset.empty:
        plt.scatter(
            subset['Custo'], subset['Equilibrio'],
            facecolors='none',
            edgecolors=style_metodo[m]['c'],
            marker=style_metodo[m]['m'],
            s=110, linewidths=1.5, alpha=0.9,
            label=f"Analisadas {style_metodo[m]['lbl']} – PROMETHEE II"
        )

df_common = df_ahp[df_ahp['is_common']].copy()
if not df_common.empty:
    plt.scatter(
        df_common['Custo'], df_common['Equilibrio'],
        c='magenta', marker='D', s=130,
        edgecolors='k', linewidths=1.2,
        label="Comuns AHP & PROMETHEE II"
    )

plt.scatter(
    best_ahp["Custo"], best_ahp["Equilibrio"],
    c='gold', marker='*', s=420,
    edgecolors='k', linewidths=1.5, zorder=10,
    label=f"Vencedora AHP (Score = {best_ahp[score_col_ahp]:.3f})"
)

plt.scatter(
    best_prom["Custo"], best_prom["Equilibrio"],
    c='red', marker='P', s=380,
    edgecolors='k', linewidths=1.5, zorder=11,
    label=f"Vencedora PROMETHEE II (φ = {best_prom['phi']:.3f})"
)

plt.xlabel("Custo Total ($f_1$)", fontsize=12)
plt.ylabel("Desequilíbrio ($f_2$)", fontsize=12)
plt.title("Comparação AHP × PROMETHEE II em $f_1$ × $f_2$", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig("grafico_comparativo_ahp_promethee_f1_f2.png", dpi=300)
plt.close()

print("Gráfico salvo: 'grafico_comparativo_ahp_promethee_f1_f2.png'")