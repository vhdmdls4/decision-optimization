import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def preparar_dados_promethee(path_csv="tabela_promethee_II.csv"):
    df = pd.read_csv(path_csv)

    required_cols = [
        "Custo Total",
        "Equilibrio",
        "Robustez (Norm 0-100)",
        "Estabilidade (Norm 0-100)",
        "phi"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada em {path_csv}")

    df["f1"] = df["Custo Total"]
    df["f2"] = df["Equilibrio"]
    df["f3"] = df["Robustez (Norm 0-100)"]
    df["f4"] = df["Estabilidade (Norm 0-100)"]

    df_top5 = df.sort_values("phi", ascending=False).head(5).reset_index(drop=True)
    return df_top5


def plot_radar_promethee(df_top5):
    categories = ['Custo (f1)', 'Equilíbrio (f2)', 'Robustez (f3)', 'Estabilidade (f4)']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ranges_local = {}
    for col in ['f1', 'f2', 'f3', 'f4']:
        ranges_local[col] = (df_top5[col].min(), df_top5[col].max())

    def scale_local(val, col, mode):
        mn, mx = ranges_local[col]
        if mx == mn:
            return 1.0

        if mode == 'min':
            norm = (mx - val) / (mx - mn)
        else:
            norm = (val - mn) / (mx - mn)

        return 0.2 + 0.8 * norm

    df_plot = df_top5.copy()
    df_plot['p_f1'] = df_plot['f1'].apply(lambda x: scale_local(x, 'f1', 'min'))
    df_plot['p_f2'] = df_plot['f2'].apply(lambda x: scale_local(x, 'f2', 'min'))
    df_plot['p_f3'] = df_plot['f3'].apply(lambda x: scale_local(x, 'f3', 'max'))
    df_plot['p_f4'] = df_plot['f4'].apply(lambda x: scale_local(x, 'f4', 'max'))

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], categories, color='black', size=11, weight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [], color="grey", size=7)
    plt.ylim(0, 1.05)

    cores = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    estilos = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    for rank, (idx, row) in enumerate(df_plot.iterrows(), start=1):
        values = [row['p_f1'], row['p_f2'], row['p_f3'], row['p_f4']]
        values += values[:1]

        origem = row.get('origem', '')
        origem_txt = f" ({origem})" if origem != '' else ""
        label = f"#{rank}{origem_txt}\nφ = {row['phi']:.3f}"

        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle=estilos[rank-1],
            color=cores[rank-1],
            label=label
        )
        ax.fill(angles, values, color=cores[rank-1], alpha=0.05)

    plt.title("PROMETHEE II – Radar Top 5 Soluções\n(Escala Local para Evidenciar Diferenças)", size=14, y=1.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig("grafico_radar_promethee_top5.png", dpi=300)
    print("Gráfico salvo: grafico_radar_promethee_top5.png")


if __name__ == "__main__":
    df_top5 = preparar_dados_promethee()
    print("Top 5 PROMETHEE II para conferência:")
    print(df_top5[[
        "Custo Total",
        "Equilibrio",
        "Robustez (Norm 0-100)",
        "Estabilidade (Norm 0-100)",
        "phi",
        "origem"
    ]])
    plot_radar_promethee(df_top5)
