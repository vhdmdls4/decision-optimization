import pandas as pd
import numpy as np

ARQ_AHP = "pareto_global_4criterios_ahp.csv"
TOP_K = 20  # usar as 20 melhores do AHP
ARQ_OUT = "promethee_ii_resultados.csv"

# Pesos vindos do AHP
pesos = {
    "f1_norm": 0.5254,
    "f2_norm": 0.1104,
    "f3_norm": 0.3009,
    "f4_norm": 0.0634
}

criterios = list(pesos.keys())

def preferencia_linear(d):
    return max(0, d)

def main():
    df = pd.read_csv(ARQ_AHP)

    # Top-K segundo AHP
    df_top = df.nsmallest(TOP_K, "rank_ahp").copy()
    df_top.reset_index(drop=True, inplace=True)

    n = len(df_top)
    matriz_pref = np.zeros((n, n))

    # Calcular preferências π(a,b)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pi_ab = 0
            for c in criterios:
                d = df_top.loc[j, c] - df_top.loc[i, c]
                p = preferencia_linear(d)
                pi_ab += pesos[c] * p

            matriz_pref[i, j] = pi_ab

    # Fluxos phi+
    phi_pos = matriz_pref.sum(axis=1) / (n - 1)
    # Fluxos phi-
    phi_neg = matriz_pref.sum(axis=0) / (n - 1)
    # Fluxo líquido
    phi = phi_pos - phi_neg

    df_top["phi_plus"] = phi_pos
    df_top["phi_minus"] = phi_neg
    df_top["phi"] = phi

    df_top_sorted = df_top.sort_values("phi", ascending=False)

    df_top_sorted.to_csv(ARQ_OUT, index=False)
    print(f"Ranking PROMETHEE II salvo em: {ARQ_OUT}")

    print("\nTop soluções segundo PROMETHEE II:")
    print(df_top_sorted[["rank_ahp", "phi", "metodo", "run", "alt", "f1", "f2", "f3_robustez", "f4_risco"]].head(15))


if __name__ == "__main__":
    main()
