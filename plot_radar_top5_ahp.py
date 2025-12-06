import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ARQ_AHP = "pareto_global_4criterios_ahp.csv"
N_TOP = 5  # quantidade de soluções que você quer no radar

def main():
    # 1. Carregar arquivo com ranking AHP
    df = pd.read_csv(ARQ_AHP)
    print(f"Arquivo '{ARQ_AHP}' carregado com {len(df)} linhas.")

    # 2. Garantir que as colunas necessárias existem
    colunas_necessarias = [
        "rank_ahp",
        "f1_norm", "f2_norm", "f3_norm", "f4_norm",
        "metodo", "run", "alt"
    ]
    for c in colunas_necessarias:
        if c not in df.columns:
            raise ValueError(f"Coluna '{c}' não encontrada em {ARQ_AHP}.")

    # 3. Pegar as N_TOP melhores soluções (rank_ahp pequeno = melhor)
    df_top = df.nsmallest(N_TOP, "rank_ahp").copy()
    df_top = df_top.sort_values("rank_ahp")  # só para garantir ordem

    print("\nTop soluções selecionadas para o radar:")
    print(df_top[["rank_ahp", "f1", "f2", "f3_robustez", "f4_risco", "metodo", "run", "alt"]])

    # 4. Critérios (na ordem dos eixos do radar)
    criterios = [
        "f1_norm",        # custo
        "f2_norm",        # desequilíbrio
        "f3_norm",        # robustez (já invertida na normalização)
        "f4_norm"         # risco
    ]
    labels_criterios = [
        "f1 (Custo)",
        "f2 (Desequilíbrio)",
        "f3 (Robustez)",
        "f4 (Risco)"
    ]

    n_criterios = len(criterios)

    # 5. Converter para escala "quanto maior, melhor"
    #    valor_bom = 1 - valor_normalizado (onde 0 era melhor e 1 pior)
    valores_bons = 1.0 - df_top[criterios].values

    # 6. Ângulos para o radar (em radianos)
    angles = np.linspace(0, 2 * np.pi, n_criterios, endpoint=False)
    # fechar o polígono: repetir primeiro ângulo no final
    angles = np.concatenate((angles, [angles[0]]))

    # 7. Figura
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # 8. Plotar cada solução
    for idx, (_, linha) in enumerate(df_top.iterrows()):
        vals = valores_bons[idx, :]
        # fechar polígono: repetir primeiro valor no final
        vals = np.concatenate((vals, [vals[0]]))

        rank = int(linha["rank_ahp"])
        metodo = linha["metodo"]
        run = int(linha["run"])
        alt = int(linha["alt"])

        # label descrevendo solução
        label_sol = f"rank {rank} - {metodo} (run {run}, alt {alt})"

        ax.plot(angles, vals, linewidth=1.5, marker='o', label=label_sol)
        ax.fill(angles, vals, alpha=0.1)

    # 9. Configuração dos eixos
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_criterios)

    # eixo radial de 0 a 1 (0 = pior, 1 = melhor depois da transformação)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

    ax.set_title("Top-5 soluções segundo AHP (escala 0–1, maior = melhor)", pad=20)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

    plt.tight_layout()
    plt.savefig("top5_ahp_radar.png", dpi=300)
    print("Gráfico salvo como 'top5_ahp_radar.png'.")
    plt.show()


if __name__ == "__main__":
    main()
