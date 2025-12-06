import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO_PROM = "promethee_ii_resultados.csv"
PASTA_FIGURAS = "figuras_mo"


def carregar_dados(arquivo: str) -> pd.DataFrame:
    df = pd.read_csv(arquivo)
    print(f"Arquivo '{arquivo}' carregado com {len(df)} linhas.")
    print("Colunas disponíveis:", list(df.columns))

    # Ordena por fluxo líquido φ (decrescente)
    df = df.sort_values("phi", ascending=False).reset_index(drop=True)
    df["rank_phi"] = np.arange(1, len(df) + 1)

    # Rótulo amigável para cada alternativa (metodo-run.alt)
    df["label_alt"] = df.apply(
        lambda row: f"{row['metodo']}-r{int(row['run'])}a{int(row['alt'])}", axis=1
    )

    return df


def garantir_pasta_figuras():
    if PASTA_FIGURAS and not os.path.exists(PASTA_FIGURAS):
        os.makedirs(PASTA_FIGURAS, exist_ok=True)


def plot_barras_phi(df: pd.DataFrame):
    """
    Gráfico de barras do fluxo líquido φ para todas as alternativas,
    ordenadas da melhor (maior φ) para a pior.
    """
    garantir_pasta_figuras()

    x = np.arange(len(df))
    phi_vals = df["phi"].values
    labels = df["label_alt"].values

    plt.figure(figsize=(10, 6))
    plt.bar(x, phi_vals)

    # Marca a melhor solução em destaque
    plt.bar(0, phi_vals[0], color="tab:red")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Alternativas (método-run.alt)")
    plt.ylabel("Fluxo líquido φ")
    plt.title("Ranking global PROMETHEE II (fluxo líquido φ)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    nome_arquivo = os.path.join(PASTA_FIGURAS, "promethee_phi_barras.png")
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()
    print(f"Figura salva em: {nome_arquivo}")


def plot_scatter_f1_f2_color_phi(df: pd.DataFrame):
    """
    Scatter f1 × f2 com cor representando o valor de φ.
    Destaca a melhor solução (rank_phi = 1) com círculo.
    """
    garantir_pasta_figuras()

    f1 = df["f1"].values
    f2 = df["f2"].values
    phi_vals = df["phi"].values
    ranks = df["rank_phi"].values
    metodos = df["metodo"].values

    plt.figure(figsize=(9, 6))

    scatter = plt.scatter(f1, f2, c=phi_vals, cmap="viridis", s=60, alpha=0.9)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Fluxo líquido φ (PROMETHEE II)")

    # Destaca a melhor solução (rank_phi = 1)
    best = df.iloc[0]
    plt.scatter(
        best["f1"],
        best["f2"],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        label="Melhor (rank φ = 1)",
    )

    # Anotações opcionais com o rank_phi em cada ponto
    for i, (xi, yi, rk) in enumerate(zip(f1, f2, ranks)):
        plt.text(xi + 0.5, yi + 0.2, str(int(rk)), fontsize=8)

    # Legendas por método (Pw / Pe)
    # Para isso, plota "pontos fantasmas" apenas para a legenda
    for metodo, cor in [("Pw", "tab:orange"), ("Pe", "tab:blue")]:
        plt.scatter([], [], c=cor, label=f"Método {metodo}")

    plt.xlabel("f1(x) – Custo total")
    plt.ylabel("f2(x) – Desequilíbrio de carga")
    plt.title("Soluções PROMETHEE II no espaço (f1, f2)\ncor = fluxo líquido φ")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")

    plt.tight_layout()
    nome_arquivo = os.path.join(PASTA_FIGURAS, "promethee_scatter_f1f2_phi.png")
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()
    print(f"Figura salva em: {nome_arquivo}")


def plot_scatter_top10(df: pd.DataFrame, top_k: int = 10):
    """
    Scatter f1 × f2 usando apenas as Top-k soluções segundo φ,
    para focar na região "de interesse" da fronteira.
    """
    garantir_pasta_figuras()

    df_top = df.nsmallest(top_k, "rank_phi")  # menores ranks = melhores

    f1 = df_top["f1"].values
    f2 = df_top["f2"].values
    phi_vals = df_top["phi"].values
    ranks = df_top["rank_phi"].values
    metodos = df_top["metodo"].values

    plt.figure(figsize=(9, 6))

    # Cores por método
    cores = []
    for m in metodos:
        if m == "Pw":
            cores.append("tab:orange")
        else:
            cores.append("tab:blue")

    plt.scatter(f1, f2, c=cores, s=80, alpha=0.9)

    # Destaca o melhor (rank_phi = 1)
    best = df_top[df_top["rank_phi"] == 1].iloc[0]
    plt.scatter(
        best["f1"],
        best["f2"],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2.2,
        label="Melhor PROMETHEE II",
    )

    # Anota rank junto ao ponto
    for xi, yi, rk in zip(f1, f2, ranks):
        plt.text(xi + 0.5, yi + 0.2, str(int(rk)), fontsize=8)

    # Legenda de método
    plt.scatter([], [], c="tab:orange", label="Método Pw")
    plt.scatter([], [], c="tab:blue", label="Método Pe")

    plt.xlabel("f1(x) – Custo total")
    plt.ylabel("f2(x) – Desequilíbrio de carga")
    plt.title(f"Top-{top_k} soluções segundo PROMETHEE II no espaço (f1, f2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")

    plt.tight_layout()
    nome_arquivo = os.path.join(
        PASTA_FIGURAS, f"promethee_scatter_f1f2_top{top_k}.png"
    )
    plt.savefig(nome_arquivo, dpi=300)
    plt.close()
    print(f"Figura salva em: {nome_arquivo}")


def main():
    df = carregar_dados(ARQUIVO_PROM)

    # 1) Barras de φ (ranking PROMETHEE II)
    plot_barras_phi(df)

    # 2) Scatter f1 × f2 colorido por φ
    plot_scatter_f1_f2_color_phi(df)

    # 3) Scatter f1 × f2 apenas com Top-10 PROMETHEE II
    plot_scatter_top10(df, top_k=10)


if __name__ == "__main__":
    main()
