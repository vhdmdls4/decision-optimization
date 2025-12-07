import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyDecision.algorithm import promethee_ii as promethee_II

df_alt = pd.read_csv("tabela_decisao_final_normalizada.csv")

expected_cols = [
    "Custo Total",
    "Equilibrio",
    "Robustez (Norm 0-100)",
    "Estabilidade (Norm 0-100)"
]
for col in expected_cols:
    if col not in df_alt.columns:
        raise ValueError(f"Coluna '{col}' não encontrada em tabela_decisao_final_normalizada.csv")

ct = df_alt["Custo Total"].values
ct_min, ct_max = ct.min(), ct.max()
den_ct = (ct_max - ct_min) if ct_max > ct_min else 1.0
n_f1 = (ct_max - ct) / den_ct

eq = df_alt["Equilibrio"].values
eq_min, eq_max = eq.min(), eq.max()
den_eq = (eq_max - eq_min) if eq_max > eq_min else 1.0
n_f2 = (eq_max - eq) / den_eq

n_f3 = df_alt["Robustez (Norm 0-100)"].values / 100.0
n_f4 = df_alt["Estabilidade (Norm 0-100)"].values / 100.0

dataset = np.column_stack([n_f1, n_f2, n_f3, n_f4])

weights = np.array([0.3891, 0.3170, 0.1724, 0.1215])

n_crit = dataset.shape[1]

Q = np.zeros(n_crit)
S = np.zeros(n_crit)
P = np.zeros(n_crit)

F = ['usual'] * n_crit

flow = promethee_II(dataset, weights, Q, S, P, F, sort=False, topn=0, graph=False, verbose=False)

phi = flow[:, 1]
df_alt["phi"] = phi

def calc_phi_plus_minus(matrix, weights):
    n, k = matrix.shape
    phi_plus = np.zeros(n)
    phi_minus = np.zeros(n)

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            pref_ab = 0.0
            for j in range(k):
                diff = matrix[a, j] - matrix[b, j]
                if diff > 0:
                    pref_ab += weights[j]
            phi_plus[a] += pref_ab
            phi_minus[b] += pref_ab

    if n > 1:
        phi_plus /= (n - 1)
        phi_minus /= (n - 1)

    return phi_plus, phi_minus

phi_plus, phi_minus = calc_phi_plus_minus(dataset, weights)
df_alt["phi_plus"] = phi_plus
df_alt["phi_minus"] = phi_minus

df_rank = df_alt.sort_values("phi", ascending=False).reset_index(drop=True)
df_rank.to_csv("tabela_promethee_II.csv", index=False)

print("\n=== TOP 5 (PROMETHEE II) ===")
print(
    df_rank[
        ["Custo Total", "Equilibrio", "phi_plus", "phi_minus", "phi"]
    ].head().to_string(index=False, float_format="%.4f")
)

best = df_rank.iloc[0]

plt.figure(figsize=(8, 6))
plt.scatter(df_alt["phi_plus"], df_alt["phi_minus"], s=80, alpha=0.8)

plt.scatter(
    best["phi_plus"], best["phi_minus"],
    c="gold", s=250, marker="*",
    edgecolors="k", linewidth=1.5,
    label=f"Melhor alternativa (φ = {best['phi']:.3f})"
)

plt.xlabel("Fluxo Positivo (Φ⁺)")
plt.ylabel("Fluxo Negativo (Φ⁻)")
plt.title("PROMETHEE I — Gráfico Φ⁺ × Φ⁻")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("promethee_phi_plus_minus.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df_rank["phi"].values, marker="o")
plt.title("PROMETHEE II — Fluxo Líquido (Ranking das Alternativas)")
plt.ylabel("Φ (fluxo líquido)")
plt.xlabel("Posição no ranking (0 = melhor)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("promethee_fluxo_liquido.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 7))

style = {
    'Pe': {'c': 'tab:blue',   'm': 'o', 'lbl': 'Pe (Epsilon)'},
    'Pw': {'c': 'tab:orange', 'm': '^', 'lbl': 'Pw (Soma Pond.)'}
}

try:
    df_full = pd.read_csv("fronteira_unificada_com_criterios.csv")
    
    for m in ['Pe', 'Pw']:
        subset = df_full[df_full['metodo'] == m]
        if not subset.empty:
            plt.scatter(
                subset['f1'], subset['f2'],
                c=style[m]['c'], marker=style[m]['m'],
                alpha=0.15, s=30,
                label=f"Todas {style[m]['lbl']}"
            )
except Exception:
    df_full = None

for m in ['Pe', 'Pw']:
    subset = df_alt[df_alt['origem'] == m]
    if not subset.empty:
        plt.scatter(
            subset['Custo Total'], subset['Equilibrio'],
            c=style[m]['c'], marker=style[m]['m'],
            s=80, edgecolors='k', alpha=1.0,
            label=f"Analisadas {style[m]['lbl']}"
        )

m_win = best['origem']
plt.scatter(
    best['Custo Total'], best['Equilibrio'],
    c='gold', marker='*', s=450,
    edgecolors='k', linewidth=1.5, zorder=10,
    label=f"Vencedora PROMETHEE II ({m_win})\nφ = {best['phi']:.3f}"
)

plt.xlabel('Custo Total ($f_1$)', fontsize=12)
plt.ylabel('Desequilíbrio ($f_2$)', fontsize=12)
plt.title('Seleção Final PROMETHEE II: Diferenciação Pe vs Pw', fontsize=14)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('grafico_decisao_promethee_diferenciado.png', dpi=300)
plt.close()

print("Gráfico salvo: 'grafico_decisao_promethee_diferenciado.png'")
