import pandas as pd
import matplotlib.pyplot as plt

ARQ_AHP = "pareto_global_4criterios_ahp.csv"

def main():
    # 1. Carrega dados já normalizados e rankeados
    df = pd.read_csv(ARQ_AHP)
    print(f"Arquivo '{ARQ_AHP}' carregado com {len(df)} linhas.")

    colunas_obrigatorias = [
        "rank_ahp", "score_ahp",
        "f1_norm", "f2_norm", "f3_norm", "f4_norm",
        "metodo"
    ]
    for c in colunas_obrigatorias:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória '{c}' não encontrada em {ARQ_AHP}.")

    # 2. Seleciona as Top-20 soluções segundo o AHP
    df_top = df.sort_values("rank_ahp").head(20).copy()
    df_top = df_top.reset_index(drop=True)

    print("\nTop-20 soluções carregadas para plot:")
    print(df_top[["rank_ahp", "score_ahp", "metodo"]])

    # 3. Define os critérios normalizados a serem plotados
    criterios = ["f1_norm", "f2_norm", "f3_norm", "f4_norm"]
    labels_criterios = [
        "f1 (Custo) norm.",
        "f2 (Desequilíbrio) norm.",
        "f3 (Robustez) norm. - menor é melhor",
        "f4 (Risco) norm. - menor é melhor"
    ]

    # 4. Prepara figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Posições dos eixos verticais
    xs = list(range(len(criterios)))

    # 5. Plota uma linha por solução
    for idx, row in df_top.iterrows():
        ys = [row[c] for c in criterios]

        # Destaque: método (Pw vs Pe)
        metodo = row["metodo"]
        if metodo == "Pw":
            cor = "tab:orange"
            alpha = 0.7
        else:  # Pe
            cor = "tab:blue"
            alpha = 0.7

        # Destaque extra para o rank 1
        if row["rank_ahp"] == 1:
            ax.plot(xs, ys, linewidth=3.0, color="black", label="Melhor (rank=1)")
            # Também re-plota em cima com cor do método, levemente deslocado
            ax.plot(xs, ys, linewidth=2.0, color=cor, alpha=0.9)
        else:
            ax.plot(xs, ys, linewidth=1.2, color=cor, alpha=alpha)

        # Opcional: anotação com o rank perto do último eixo
        ax.text(xs[-1] + 0.05, ys[-1], f"{int(row['rank_ahp'])}",
                fontsize=8, va="center")

    # 6. Desenha os eixos verticais
    for x in xs:
        ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_criterios, rotation=15, ha="right")

    # Como todos os critérios são normalizados [0,1] e queremos minimizar,
    # valores mais baixos são melhores.
    ax.set_ylabel("Valor normalizado (0 = melhor, 1 = pior)")
    ax.set_title("Top-20 soluções segundo AHP em coordenadas paralelas")

    # Legenda simples para métodos
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="tab:orange", lw=2, label="Método Pw"),
        Line2D([0], [0], color="tab:blue",   lw=2, label="Método Pe"),
        Line2D([0], [0], color="black",      lw=3, label="Melhor (rank=1)")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("top20_ahp_parallel.png", dpi=300)
    print("Figura salva como 'top20_ahp_parallel.png'.")
    plt.show()

if __name__ == "__main__":
    main()
