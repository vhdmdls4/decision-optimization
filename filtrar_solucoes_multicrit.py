import pandas as pd
import numpy as np

# 1. Carregar fronteiras enriquecidas
ARQ_PW = "fronteiras_pw_enriquecido.csv"
ARQ_PE = "fronteiras_pe_enriquecido.csv"

def filtrar_factivel(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Mantém apenas soluções com penalidade de capacidade ~ 0."""
    if "pen_cap" not in df.columns:
        raise ValueError("Coluna 'pen_cap' não encontrada no DataFrame.")
    df_feas = df[df["pen_cap"] <= tol].copy()
    df_feas.reset_index(drop=True, inplace=True)
    return df_feas

def pareto_nao_dominadas_4crit(
    df: pd.DataFrame,
    cols_min = ("f1", "f2", "f4_risco"),
    cols_max = ("f3_robustez",)
) -> pd.DataFrame:
    """
    Filtra as soluções não-dominadas considerando:
      - cols_min: critérios a minimizar
      - cols_max: critérios a maximizar

    Domínio: solução j domina i se:
      j é <= i em todos os critérios escalarizados para min
      e < em pelo menos um deles.
    """
    for col in list(cols_min) + list(cols_max):
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")

    # Copia e transforma critérios de maximização em "custos" negativos
    dados = df.copy()
    for col in cols_max:
        dados[col] = -dados[col]

    crit_cols = list(cols_min) + list(cols_max)
    valores = dados[crit_cols].to_numpy()

    n = len(valores)
    dominado = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominado[i]:
            continue
        for j in range(n):
            if i == j or dominado[j]:
                continue

            # j domina i?
            # j <= i em todos os critérios E < em pelo menos um
            if np.all(valores[j] <= valores[i]) and np.any(valores[j] < valores[i]):
                dominado[i] = True
                break

    nao_dom = df[~dominado].copy()
    nao_dom.reset_index(drop=True, inplace=True)
    return nao_dom

def main():
    # --- Carrega CSVs ---
    df_pw = pd.read_csv(ARQ_PW)
    df_pe = pd.read_csv(ARQ_PE)

    print(f"[PW] Linhas originais: {len(df_pw)}")
    print(f"[PE] Linhas originais: {len(df_pe)}")

    # --- 1) Filtra apenas soluções factíveis ---
    df_pw_feas = filtrar_factivel(df_pw)
    df_pe_feas = filtrar_factivel(df_pe)

    print(f"[PW] Soluções factíveis: {len(df_pw_feas)}")
    print(f"[PE] Soluções factíveis: {len(df_pe_feas)}")

    # --- 2) Frente de Pareto em 4 critérios, separada por método ---
    pareto_pw = pareto_nao_dominadas_4crit(df_pw_feas)
    pareto_pe = pareto_nao_dominadas_4crit(df_pe_feas)

    print(f"[PW] Soluções não-dominadas (4 crit.): {len(pareto_pw)}")
    print(f"[PE] Soluções não-dominadas (4 crit.): {len(pareto_pe)}")

    pareto_pw.to_csv("pareto_pw_4criterios.csv", index=False)
    pareto_pe.to_csv("pareto_pe_4criterios.csv", index=False)
    print("Arquivos salvos: 'pareto_pw_4criterios.csv' e 'pareto_pe_4criterios.csv'.")

    # --- 3) (Opcional) Frente global combinando métodos ---
    pareto_pw_aux = pareto_pw.copy()
    pareto_pw_aux["metodo"] = "Pw"

    pareto_pe_aux = pareto_pe.copy()
    pareto_pe_aux["metodo"] = "Pe"

    df_all = pd.concat([pareto_pw_aux, pareto_pe_aux], ignore_index=True)

    pareto_all = pareto_nao_dominadas_4crit(df_all)
    print(f"[GLOBAL] Soluções não-dominadas (Pw+Pe): {len(pareto_all)}")

    pareto_all.to_csv("pareto_global_4criterios.csv", index=False)
    print("Arquivo salvo: 'pareto_global_4criterios.csv'.")

if __name__ == "__main__":
    main()
