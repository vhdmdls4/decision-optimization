import pandas as pd
import numpy as np

ARQUIVO_ENTRADA = "pareto_global_4criterios.csv"
ARQUIVO_SAIDA   = "pareto_global_4criterios_ahp.csv"

def main():
    # -------------------------------------------------------------------------
    # 1. Lê o arquivo com as soluções de Pareto (4 critérios)
    # -------------------------------------------------------------------------
    df = pd.read_csv(ARQUIVO_ENTRADA)
    print(f"Arquivo '{ARQUIVO_ENTRADA}' carregado com {len(df)} linhas.")
    print("Colunas disponíveis:", list(df.columns))

    # -------------------------------------------------------------------------
    # 2. Mapeamento de critérios -> colunas do CSV + sentido (min/max)
    #    - f1: custo total        (minimizar)
    #    - f2: desequilíbrio      (minimizar)
    #    - f3: robustez           (maximizar)
    #    - f4: risco              (minimizar)
    # -------------------------------------------------------------------------
    CRIT_INFO = {
        "f1": {"col": "f1",           "sense": "min"},
        "f2": {"col": "f2",           "sense": "min"},
        "f3": {"col": "f3_robustez",  "sense": "max"},
        "f4": {"col": "f4_risco",     "sense": "min"},
    }

    # Verifica se as colunas existem
    for crit, info in CRIT_INFO.items():
        col = info["col"]
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada em {ARQUIVO_ENTRADA}.")

    # -------------------------------------------------------------------------
    # 3. Pesos AHP já calculados (matriz A_novo)
    #    Ordem: [f1 (Custo), f2 (Equilíbrio), f3 (Robustez), f4 (Risco)]
    # -------------------------------------------------------------------------
    pesos_ahp = {
        "f1": 0.52541396,
        "f2": 0.11036581,
        "f3": 0.30086488,
        "f4": 0.06335535,
    }

    print("\nPesos AHP utilizados:")
    for crit, w in pesos_ahp.items():
        print(f"  {crit}: {w:.4f} ({w*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. Normalização min–máx com sentido (min/max)
    #
    #    - Se sense == "min":  crit_norm = (x - min) / (max - min)
    #                          (0 = melhor, 1 = pior)
    #    - Se sense == "max":  crit_norm = (max - x) / (max - min)
    #                          (0 = melhor, 1 = pior)
    #
    #    Se max == min, o critério não varia; definimos tudo como 0.
    # -------------------------------------------------------------------------
    for crit, info in CRIT_INFO.items():
        col = info["col"]
        sense = info["sense"]

        valores = df[col].astype(float).values
        vmin = np.min(valores)
        vmax = np.max(valores)

        denom = vmax - vmin
        if denom <= 0:
            # Critério constante: todo mundo recebe 0 (todos igualmente bons)
            df[f"{crit}_norm"] = 0.0
            print(f"Atenção: critério {crit} ({col}) tem vmax == vmin. Normalizado como 0.")
            continue

        if sense == "min":
            # minimizar
            df[f"{crit}_norm"] = (valores - vmin) / denom
        elif sense == "max":
            # maximizar (invertendo para seguir lógica de minimização do score)
            df[f"{crit}_norm"] = (vmax - valores) / denom
        else:
            raise ValueError(f"Sense inválido para {crit}: {sense}")

        print(f"{crit} ({col}): min={vmin:.4f}, max={vmax:.4f} -> coluna '{crit}_norm' criada.")

    # -------------------------------------------------------------------------
    # 5. Calcula o score agregado AHP
    #    Score = w1*f1_norm + w2*f2_norm + w3*f3_norm + w4*f4_norm
    #    (quanto MENOR o score, melhor a solução)
    # -------------------------------------------------------------------------
    df["score_ahp"] = (
        pesos_ahp["f1"] * df["f1_norm"] +
        pesos_ahp["f2"] * df["f2_norm"] +
        pesos_ahp["f3"] * df["f3_norm"] +
        pesos_ahp["f4"] * df["f4_norm"]
    )

    # -------------------------------------------------------------------------
    # 6. Ordena da melhor (menor score_ahp) para a pior
    # -------------------------------------------------------------------------
    df_ordenado = df.sort_values(by="score_ahp", ascending=True).reset_index(drop=True)

    # Cria uma coluna de ranking explícita (1 = melhor)
    df_ordenado["rank_ahp"] = df_ordenado.index + 1

    # -------------------------------------------------------------------------
    # 7. Salva em CSV e mostra Top 10 no console
    # -------------------------------------------------------------------------
    df_ordenado.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    print(f"\nArquivo com ranking AHP salvo em: {ARQUIVO_SAIDA}")

    print("\nTop 20 soluções segundo o AHP:")
    colunas_para_mostrar = [
        "rank_ahp",
        "score_ahp",
        "f1", "f2", "f3_robustez", "f4_risco",
        "metodo", "run", "alt", "w1", "w2", "epsilon",
    ]
    colunas_existentes = [c for c in colunas_para_mostrar if c in df_ordenado.columns]

    print(df_ordenado[colunas_existentes].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
