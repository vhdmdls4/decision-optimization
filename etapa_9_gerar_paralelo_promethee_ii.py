import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. PREPARAR DADOS: TOP 8 PROMETHEE II
# ==========================================
def preparar_dados_top8_promethee(path_csv="tabela_promethee_II.csv"):
    # Carrega a tabela resultante do PROMETHEE II
    df = pd.read_csv(path_csv)

    # Conferência de colunas necessárias
    required_cols = [
        "Custo Total",
        "Equilibrio",
        "Robustez (Norm 0-100)",
        "Estabilidade (Norm 0-100)",
        "phi",
        "origem",        # deve conter 'Pe' ou 'Pw'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada em {path_csv}")

    # Renomeia para nomes internos de trabalho
    df["f1"] = df["Custo Total"]
    df["f2"] = df["Equilibrio"]
    df["f3"] = df["Robustez (Norm 0-100)"]
    df["f4"] = df["Estabilidade (Norm 0-100)"]

    # Ranges globais para normalização
    ranges = {}
    for col in ["f1", "f2", "f3", "f4"]:
        vmin = df[col].min()
        vmax = df[col].max()
        ranges[col] = (vmin, vmax)

    def norm_min(series, col):
        vmin, vmax = ranges[col]
        if vmax == vmin:
            return np.ones_like(series, dtype=float)
        return (vmax - series) / (vmax - vmin)

    def norm_max(series, col):
        vmin, vmax = ranges[col]
        if vmax == vmin:
            return np.ones_like(series, dtype=float)
        return (series - vmin) / (vmax - vmin)

    # f1, f2 -> minimizar  |  f3, f4 -> maximizar (já são benefício)
    df["n_f1"] = norm_min(df["f1"], "f1")
    df["n_f2"] = norm_min(df["f2"], "f2")
    df["n_f3"] = norm_max(df["f3"], "f3")
    df["n_f4"] = norm_max(df["f4"], "f4")

    # Seleciona Top 8 por fluxo líquido φ (PROMETHEE II)
    df_top = df.sort_values("phi", ascending=False).head(8).reset_index(drop=True)

    return df_top


# ==========================================
# 2. PLOTAGEM EM COORDENADAS PARALELAS
# ==========================================
def plot_paralelo_promethee(df_top):
    fig, ax = plt.subplots(figsize=(12, 7))

    cols_plot = ["n_f1", "n_f2", "n_f3", "n_f4"]
    labels_eixo = [
        "Custo ($f_1$)\n(Minimizar)",
        "Equilíbrio ($f_2$)\n(Minimizar)",
        "Robustez ($f_3$)\n(Maximizar)",
        "Estabilidade ($f_4$)\n(Maximizar)",
    ]
    x_coords = list(range(len(cols_plot)))

    # Plota de trás pra frente para a melhor (índice 0) ficar por cima
    for idx in reversed(df_top.index):
        row = df_top.loc[idx]
        y_values = [row[c] for c in cols_plot]

        # Cor por método
        if row["origem"] == "Pw":
            color = "tab:orange"
            alpha = 0.6
        else:  # 'Pe'
            color = "tab:blue"
            alpha = 0.6

        # Melhor solução PROMETHEE II (phi máximo)
        if idx == 0:
            # Contorno preto grosso
            ax.plot(
                x_coords,
                y_values,
                color="black",
                linewidth=4,
                alpha=0.85,
                zorder=10,
            )
            # Linha colorida por cima
            ax.plot(
                x_coords,
                y_values,
                color=color,
                linewidth=2.5,
                zorder=11,
            )
            ax.text(
                x_coords[-1] + 0.05,
                y_values[-1],
                f"Melhor PROMETHEE II\nφ = {row['phi']:.3f} e Score Global AHP = {row['Score_Global']:.3f}",
                va="center",
                fontsize=9,
            )
        else:
            ax.plot(
                x_coords,
                y_values,
                color=color,
                linewidth=1.3,
                alpha=alpha,
                zorder=2,
            )

    # Eixos verticais
    for x in x_coords:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1, alpha=0.3)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels_eixo, fontsize=11, fontweight="bold")

    ax.set_ylabel("Utilidade Normalizada (1.0 = Melhor, 0.0 = Pior)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Coordenadas Paralelas: Top 8 Soluções", fontsize=14, pad=15)

    # Legenda
    legend_elements = [
        Line2D([0], [0], color="tab:orange", lw=2, label="Soluções Pw"),
        Line2D([0], [0], color="tab:blue", lw=2, label="Soluções Pe"),
        Line2D([0], [0], color="black", lw=3, label="Melhor Solução AHP (Score Global) e PROMETHEE II (φ Máximo)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig("grafico_paralelo_top8.png", dpi=300)
    print("Gráfico salvo: grafico_paralelo_top8.png")


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    df_top8 = preparar_dados_top8_promethee()
    print("Top 8 Soluções para conferência:")
    print(df_top8[["Custo Total", "Equilibrio", "Robustez (Norm 0-100)",
                   "Estabilidade (Norm 0-100)", "phi", "origem"]])
    plot_paralelo_promethee(df_top8)
