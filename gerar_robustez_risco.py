import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------
# 1) Carregar dados da instância
# ---------------------------------------------------------

def load_instance(prefix="data_5x50"):
    """Carrega matrizes a, b, c e escalares m, n a partir dos CSVs."""
    base = "."
    a = np.loadtxt(os.path.join(base, f"{prefix}_a.csv"), delimiter=",")
    b = np.loadtxt(os.path.join(base, f"{prefix}_b.csv"), delimiter=",")
    c = np.loadtxt(os.path.join(base, f"{prefix}_c.csv"), delimiter=",")
    m = int(np.loadtxt(os.path.join(base, f"{prefix}_m.csv"), delimiter=","))
    n = int(np.loadtxt(os.path.join(base, f"{prefix}_n.csv"), delimiter=","))

    params = {"a": a, "b": b, "c": c, "m": m, "n": n}
    print(f"Instância carregada: m={m} agentes, n={n} tarefas.")
    return params

# ---------------------------------------------------------
# 2) Decodificar a solução a partir da coluna 'solucao'
# ---------------------------------------------------------

def decode_solution(sol_str: str, n: int) -> np.ndarray:
    """
    Converte a string '0,1,4,3,...' em um vetor numpy de tamanho n.
    """
    tokens = str(sol_str).replace("[", "").replace("]", "").split(",")
    vals = [int(t) for t in tokens if t.strip() != ""]
    if len(vals) != n:
        raise ValueError(
            f"Tamanho da solução ({len(vals)}) != n ({n}). "
            f"Valor bruto: {str(sol_str)[:80]}..."
        )
    return np.array(vals, dtype=int)

# ---------------------------------------------------------
# 3) Cálculo de f3 (Robustez) e f4 (Risco)
# ---------------------------------------------------------

def compute_f3_robustness(x: np.ndarray, params: dict) -> float:
    """
    f3(x): Robustez = folga média relativa de capacidade.

    Para cada agente i:
        carga_i  = sum_{j: x_j = i} a[i, j]
        slack_i  = max(0, b[i] - carga_i)
        slack_rel_i = slack_i / b[i]

    f3(x) = média(slack_rel_i)
    Quanto MAIOR f3, mais robusta a solução (mais folga).
    """
    a = params["a"]
    b = params["b"]
    m = params["m"]
    n = params["n"]

    loads = np.zeros(m)
    for j in range(n):
        i = int(x[j])
        loads[i] += a[i, j]

    slack = np.maximum(0.0, b - loads)
    slack_rel = slack / b

    return float(slack_rel.mean())


def compute_f4_risk(
    x: np.ndarray,
    params: dict,
    num_scenarios: int = 200,
    rho: float = 0.10,
    rng: np.random.Generator | None = None,
) -> float:
    """
    f4(x): Risco = violação média (normalizada) sob cenários de incerteza.

    Para cada cenário s:
        a_s[i,j] = a[i,j] * (1 + eps_{i,j}),   eps ~ U[-rho, rho]
        carga_i^s = sum_{j: x_j = i} a_s[i, j]
        viol_i^s  = max(0, carga_i^s - b[i])

        risco_s = sum_i (viol_i^s / b[i])

    f4(x) = média_s (risco_s)
    Quanto MAIOR f4, maior o risco (pior).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    a = params["a"]
    b = params["b"]
    m = params["m"]
    n = params["n"]

    riscos = []

    for _ in range(num_scenarios):
        eps = rng.uniform(-rho, rho, size=a.shape)
        a_pert = a * (1.0 + eps)

        loads = np.zeros(m)
        for j in range(n):
            i = int(x[j])
            loads[i] += a_pert[i, j]

        viol = np.maximum(0.0, loads - b)
        risco_s = float((viol / b).sum())
        riscos.append(risco_s)

    return float(np.mean(riscos))

# ---------------------------------------------------------
# 4) Enriquecer um DataFrame de fronteira com f3 e f4
# ---------------------------------------------------------

def enrich_frontier(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas:
      - run, alt, f1, f2, pen_cap, w1, w2, solucao  (no caso de Pw)
      - ou run, alt, f1, f2, pen_cap, epsilon, solucao  (no caso de Pe)

    Lê a coluna 'solucao', reconstrói x, e adiciona:
      - f3_robustez
      - f4_risco
    """
    n = params["n"]

    if "solucao" not in df.columns:
        raise ValueError("Coluna 'solucao' não encontrada no CSV de fronteira.")

    f3_vals = []
    f4_vals = []

    for idx, row in df.iterrows():
        x = decode_solution(row["solucao"], n)
        f3 = compute_f3_robustness(x, params)
        f4 = compute_f4_risk(x, params)
        f3_vals.append(f3)
        f4_vals.append(f4)

    df_enr = df.copy()
    df_enr["f3_robustez"] = f3_vals
    df_enr["f4_risco"] = f4_vals

    return df_enr

# ---------------------------------------------------------
# 5) Função principal
# ---------------------------------------------------------

def main():
    params = load_instance(prefix="data_5x50")

    # Carrega fronteiras geradas pelo seu código multiobjetivo
    df_pw = pd.read_csv("fronteiras_pw.csv")
    df_pe = pd.read_csv("fronteiras_pe.csv")

    print("Enriquecendo fronteira Pw com f3 e f4...")
    df_pw_enr = enrich_frontier(df_pw, params)
    df_pw_enr.to_csv("fronteiras_pw_enriquecido.csv", index=False)
    print("Arquivo salvo: fronteiras_pw_enriquecido.csv")

    print("Enriquecendo fronteira Pe com f3 e f4...")
    df_pe_enr = enrich_frontier(df_pe, params)
    df_pe_enr.to_csv("fronteiras_pe_enriquecido.csv", index=False)
    print("Arquivo salvo: fronteiras_pe_enriquecido.csv")

if __name__ == "__main__":
    main()
