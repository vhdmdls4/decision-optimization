import pandas as pd
import matplotlib.pyplot as plt
import os

ARQ_AHP = "pareto_global_4criterios_ahp.csv"

def main():
    # ==========================
    # 1. Carregar e filtrar top-20
    # ==========================
    df = pd.read_csv(ARQ_AHP)
    if 'rank_ahp' not in df.columns:
        raise ValueError("A coluna 'rank_ahp' não foi encontrada. "
                         "Certifique-se de ter rodado o script gerar_pareto_global_ahp.py antes.")

    df_top20 = df.sort_values('rank_ahp').head(20).copy()

    print("Top-20 carregado. Linhas:")
    print(df_top20[['rank_ahp', 'score_ahp', 'f1', 'f2', 'f3_robustez', 'f4_risco', 'metodo']])

    # Cria pasta de figuras se quiser organizar
    pasta_fig = "figuras_mo"
    os.makedirs(pasta_fig, exist_ok=True)

    # ==========================
    # 2. Gráfico f1 x f2 (custo x desequilíbrio)
    # ==========================
    plt.figure(figsize=(8, 6))

    # Agrupa por método para colorir diferente Pw vs Pe
    for metodo, sub in df_top20.groupby('metodo'):
        plt.scatter(
            sub['f1'],
            sub['f2'],
            label=f"Método {metodo}",
            alpha=0.8
        )

        # Opcional: anotar o rank ao lado de cada ponto
        for _, row in sub.iterrows():
            plt.annotate(
                int(row['rank_ahp']),
                (row['f1'], row['f2']),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    # Destacar a melhor solução (rank_ahp = 1)
    best = df_top20.nsmallest(1, 'rank_ahp').iloc[0]
    plt.scatter(
        best['f1'],
        best['f2'],
        s=150,
        facecolors='none',
        edgecolors='black',
        linewidths=2,
        label=f"Melhor (rank={int(best['rank_ahp'])})"
    )

    plt.xlabel("f1(x) – Custo total")
    plt.ylabel("f2(x) – Desequilíbrio de carga")
    plt.title("Top-20 soluções segundo AHP no espaço (f1, f2)")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    nome_fig1 = os.path.join(pasta_fig, "ahp_top20_f1_f2.png")
    plt.savefig(nome_fig1, dpi=300)
    print(f"Figura salva em: {nome_fig1}")

    # ==========================
    # 3. Gráfico do score AHP por rank
    # ==========================
    plt.figure(figsize=(8, 4))

    df_top20_ord = df_top20.sort_values('rank_ahp')
    plt.plot(
        df_top20_ord['rank_ahp'],
        df_top20_ord['score_ahp'],
        marker='o'
    )

    plt.xlabel("Rank AHP")
    plt.ylabel("Score AHP (menor = melhor)")
    plt.title("Score AHP das 20 melhores soluções")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(df_top20_ord['rank_ahp'])

    nome_fig2 = os.path.join(pasta_fig, "ahp_top20_score.png")
    plt.tight_layout()
    plt.savefig(nome_fig2, dpi=300)
    print(f"Figura salva em: {nome_fig2}")

    # Se quiser visualizar na hora:
    # plt.show()

if __name__ == "__main__":
    main()
