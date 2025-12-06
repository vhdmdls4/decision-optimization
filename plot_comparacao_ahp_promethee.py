import pandas as pd
import matplotlib.pyplot as plt
import os

ARQUIVO_PROM = "promethee_ii_resultados.csv"

# quantas soluções do topo de cada método considerar
TOP_N = 20  

def main():
    if not os.path.exists(ARQUIVO_PROM):
        raise FileNotFoundError(f"Arquivo {ARQUIVO_PROM} não encontrado.")

    df = pd.read_csv(ARQUIVO_PROM)

    # Conferir colunas existentes
    print("Colunas disponíveis em promethee_ii_resultados.csv:")
    print(df.columns.tolist())

    # Verificações mínimas
    required_cols = [
        "metodo", "run", "alt",
        "f1", "f2", "f3_robustez", "f4_risco",
        "score_ahp", "phi"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Coluna obrigatória '{c}' não encontrada no CSV.")

    # Cria um ID único por solução para cruzar AHP x PROMETHEE
    df["id"] = df.apply(
        lambda row: f"{row['metodo']}_r{int(row['run'])}_a{int(row['alt'])}",
        axis=1
    )

    # -----------------------------
    # 1) GRÁFICO AHP x PROMETHEE
    # -----------------------------
    # AHP: menor score_ahp é melhor
    # PROMETHEE: maior phi é melhor

    # Melhor solução segundo AHP
    idx_best_ahp = df["score_ahp"].idxmin()
    best_ahp = df.loc[idx_best_ahp]

    # Melhor solução segundo PROMETHEE II
    idx_best_prom = df["phi"].idxmax()
    best_prom = df.loc[idx_best_prom]

    print("\nMelhor solução segundo AHP:")
    print(best_ahp[["metodo", "run", "alt", "f1", "f2", "f3_robustez", "f4_risco", "score_ahp", "phi"]])

    print("\nMelhor solução segundo PROMETHEE II:")
    print(best_prom[["metodo", "run", "alt", "f1", "f2", "f3_robustez", "f4_risco", "score_ahp", "phi"]])

    plt.figure(figsize=(8, 6))

    # Scatter geral: cada ponto = solução, cor por método (Pw, Pe)
    for metodo in df["metodo"].unique():
        sub = df[df["metodo"] == metodo]
        plt.scatter(
            sub["score_ahp"],
            sub["phi"],
            alpha=0.7,
            label=f"Método {metodo}"
        )

    # Destaca melhor AHP
    plt.scatter(
        best_ahp["score_ahp"],
        best_ahp["phi"],
        marker="*",
        s=200,
        edgecolors="black",
        linewidths=1.5,
        label="Melhor AHP"
    )

    # Destaca melhor PROMETHEE
    plt.scatter(
        best_prom["score_ahp"],
        best_prom["phi"],
        marker="P",   # pentágono
        s=200,
        edgecolors="black",
        linewidths=1.5,
        label="Melhor PROMETHEE II"
    )

    plt.xlabel("Score AHP (menor é melhor)")
    plt.ylabel("Fluxo líquido φ (maior é melhor)")
    plt.title("Comparação AHP x PROMETHEE II (todas as soluções)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    nome_fig1 = "comparacao_ahp_promethee_scores.png"
    plt.savefig(nome_fig1, dpi=300)
    print(f"\nFigura 1 salva em: {nome_fig1}")

    # -------------------------------------------
    # 2) GRÁFICO COM SOLUÇÕES COMUNS (TOP-N)
    # -------------------------------------------

    # Top-N do AHP (menores scores)
    top_ahp = df.nsmallest(TOP_N, "score_ahp").copy()
    # Top-N do PROMETHEE II (maiores φ)
    top_prom = df.nlargest(TOP_N, "phi").copy()

    ids_ahp = set(top_ahp["id"])
    ids_prom = set(top_prom["id"])
    ids_comuns = sorted(ids_ahp.intersection(ids_prom))

    comuns = df[df["id"].isin(ids_comuns)].copy()

    print(f"\nTOP_N = {TOP_N}")
    print(f"Qtd top-N AHP: {len(top_ahp)}, top-N PROMETHEE: {len(top_prom)}")
    print(f"Soluções em comum entre top-N de AHP e PROMETHEE: {len(comuns)}")

    if comuns.empty:
        print("Não há soluções em comum entre os top-N de AHP e PROMETHEE. "
              "Ajuste TOP_N para capturar alguma interseção.")
    else:
        # Dentro do conjunto comum, identifica a melhor de cada método
        idx_best_ahp_comum = comuns["score_ahp"].idxmin()
        best_ahp_comum = comuns.loc[idx_best_ahp_comum]

        idx_best_prom_comum = comuns["phi"].idxmax()
        best_prom_comum = comuns.loc[idx_best_prom_comum]

        print("\nMelhor solução COMUM segundo AHP:")
        print(best_ahp_comum[["metodo", "run", "alt", "f1", "f2", "f3_robustez", "f4_risco", "score_ahp", "phi"]])

        print("\nMelhor solução COMUM segundo PROMETHEE II:")
        print(best_prom_comum[["metodo", "run", "alt", "f1", "f2", "f3_robustez", "f4_risco", "score_ahp", "phi"]])

        plt.figure(figsize=(8, 6))

        # Scatter das soluções comuns no plano f1 x f2
        for metodo in comuns["metodo"].unique():
            sub = comuns[comuns["metodo"] == metodo]
            plt.scatter(
                sub["f1"],
                sub["f2"],
                alpha=0.7,
                label=f"Método {metodo}"
            )

        # Destaca melhor AHP (comum)
        plt.scatter(
            best_ahp_comum["f1"],
            best_ahp_comum["f2"],
            marker="*",
            s=200,
            edgecolors="black",
            linewidths=1.5,
            label="Melhor AHP (comum)"
        )

        # Destaca melhor PROMETHEE (comum)
        plt.scatter(
            best_prom_comum["f1"],
            best_prom_comum["f2"],
            marker="P",
            s=200,
            edgecolors="black",
            linewidths=1.5,
            label="Melhor PROMETHEE II (comum)"
        )

        plt.xlabel("f1(x) – Custo total")
        plt.ylabel("f2(x) – Desequilíbrio de carga")
        plt.title(f"Soluções comuns entre top-{TOP_N} de AHP e PROMETHEE II")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout()

        nome_fig2 = f"comparacao_ahp_promethee_comuns_top{TOP_N}.png"
        plt.savefig(nome_fig2, dpi=300)
        print(f"Figura 2 salva em: {nome_fig2}")


if __name__ == "__main__":
    main()
