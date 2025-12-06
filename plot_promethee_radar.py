import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARQ = "promethee_ii_resultados.csv"
PASTA = "figuras_mo"

def normalizar_min_max(serie):
    # mantém como Series, preservando o índice
    serie = serie.astype(float)
    return (serie - serie.min()) / (serie.max() - serie.min() + 1e-9)

def preparar_dados_radar(df_top5):
    """
    Retorna um dicionário:
      nome_alt -> valores_normalizados (já no sentido 1 = melhor, 0 = pior)
    Critérios (na ordem):
      f1 (Custo)       -> menor é melhor
      f2 (Desequilíbrio) -> menor é melhor
      f3 (Robustez)    -> maior é melhor
      f4 (Risco)       -> menor é melhor
    """
    # Normaliza cada critério em [0,1]
    f1_norm = normalizar_min_max(df_top5["f1"])
    f2_norm = normalizar_min_max(df_top5["f2"])
    f3_norm = normalizar_min_max(df_top5["f3_robustez"])
    f4_norm = normalizar_min_max(df_top5["f4_risco"])

    # Converte tudo para escala "1 = melhor, 0 = pior"
    f1_bom = 1 - f1_norm      # minimização
    f2_bom = 1 - f2_norm      # minimização
    f3_bom = f3_norm          # maximização
    f4_bom = 1 - f4_norm      # minimização

    criterios = ["Custo (f1)", "Desequilíbrio (f2)", "Robustez (f3)", "Risco (f4)"]

    dados = {}
    for idx, row in df_top5.iterrows():
        label = f"{row['metodo']}-run{int(row['run'])}-alt{int(row['alt'])}"
        valores = [
            f1_bom.loc[idx],
            f2_bom.loc[idx],
            f3_bom.loc[idx],
            f4_bom.loc[idx],
        ]
        dados[label] = valores

    return criterios, dados

def plot_radar_promethee(df):
    if not os.path.exists(PASTA):
        os.makedirs(PASTA)

    # Ordena por phi (decrescente) e pega top-5
    df_ord = df.sort_values("phi", ascending=False).reset_index(drop=True)
    df_top5 = df_ord.head(5)

    criterios, dados = preparar_dados_radar(df_top5)

    n_criterios = len(criterios)
    angles = np.linspace(0, 2 * np.pi, n_criterios, endpoint=False).tolist()
    angles += angles[:1]  # fecha o ciclo

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Grade e limites
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)

    # Rótulos dos eixos angulares
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criterios)

    # Linhas de referência radiais
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Desenha cada solução
    for label, valores in dados.items():
        valores_ciclo = valores + valores[:1]
        ax.plot(angles, valores_ciclo, linewidth=1.5, alpha=0.8, label=label)
        ax.fill(angles, valores_ciclo, alpha=0.10)

    # Destaca a melhor (primeira do df_top5)
    best_label = list(dados.keys())[0]
    best_vals = dados[best_label] + dados[best_label][:1]
    ax.plot(angles, best_vals, linewidth=3, label=f"Melhor: {best_label}", color="black")
    ax.fill(angles, best_vals, alpha=0.15, color="black")

    plt.title("Top-5 soluções segundo PROMETHEE II (escala 1 = melhor)", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)
    plt.tight_layout()

    nome_arq = os.path.join(PASTA, "promethee_radar_top5.png")
    plt.savefig(nome_arq, dpi=300)
    plt.close()
    print("Figura de radar salva em:", nome_arq)

def main():
    df = pd.read_csv(ARQ)
    plot_radar_promethee(df)

if __name__ == "__main__":
    main()
