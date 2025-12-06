import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ARQ = "promethee_ii_resultados.csv"
PASTA = "figuras_mo"

def normalizar(df, col):
    x = df[col].values.astype(float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def plot_parallel(df):

    if not os.path.exists(PASTA):
        os.makedirs(PASTA)

    # Normaliza todos os critérios no sentido “quanto menor melhor”
    df_plot = df.copy()
    df_plot["f1_norm"] = normalizar(df_plot, "f1")
    df_plot["f2_norm"] = normalizar(df_plot, "f2")
    df_plot["f3_norm"] = 1 - normalizar(df_plot, "f3_robustez")  # robustez é MAX → inverte
    df_plot["f4_norm"] = normalizar(df_plot, "f4_risco")         # risco é MIN

    # Selecionar somente as top-20 alternativas
    df_top = df_plot.nsmallest(20, "rank_phi")

    criterios = ["f1_norm", "f2_norm", "f3_norm", "f4_norm"]
    x = range(len(criterios))

    plt.figure(figsize=(12, 7))

    for _, row in df_top.iterrows():
        metodo = row["metodo"]
        cor = "tab:orange" if metodo == "Pw" else "tab:blue"
        plt.plot(x, row[criterios], color=cor, alpha=0.7)

    # Destacar a melhor alternativa
    best = df_top.iloc[0]
    plt.plot(
        x,
        best[criterios].values,
        color="black",
        linewidth=3,
        label="Melhor (rank φ = 1)"
    )

    plt.xticks(x, ["f1 (Custo)", "f2 (Desequilibrio)", "f3 (Robustez)", "f4 (Risco)"])
    plt.ylabel("Normalizado (0 = melhor)")
    plt.title("Top-20 soluções segundo PROMETHEE II — Coordenadas Paralelas")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend()
    plt.tight_layout()

    nome = os.path.join(PASTA, "promethee_parallel.png")
    plt.savefig(nome, dpi=300)
    plt.close()
    print("Figura salva:", nome)


def main():
    df = pd.read_csv(ARQ)
    df = df.sort_values("phi", ascending=False).reset_index(drop=True)
    df["rank_phi"] = np.arange(1, len(df)+1)
    plot_parallel(df)


if __name__ == "__main__":
    main()
