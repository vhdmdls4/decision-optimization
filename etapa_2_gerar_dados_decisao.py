import numpy as np
import pandas as pd
import os

def load_params(prefix="data_5x50"):
    try:
        a = np.loadtxt(f"{prefix}_a.csv", delimiter=",")
        b = np.loadtxt(f"{prefix}_b.csv", delimiter=",")
    except Exception as e:
        print(f"Erro ao carregar arquivos principais (a, b): {e}")
        return None

    try:
        m = int(np.loadtxt(f"{prefix}_m.csv", delimiter=","))
    except:
        m = len(b)
        print(f"Aviso: {prefix}_m.csv não encontrado. Inferido m={m}")

    try:
        n = int(np.loadtxt(f"{prefix}_n.csv", delimiter=","))
    except:
        n = a.shape[1]
        print(f"Aviso: {prefix}_n.csv não encontrado. Inferido n={n}")

    params = {
        "a": a,
        "b": b,
        "m": m,
        "n": n,
    }
    return params

def parse_solution_string(s: str, n: int) -> np.ndarray:
    clean_s = str(s).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
    valores = [int(x) for x in clean_s.split(",") if x.strip() != ""]
    if len(valores) != n:
        print(f"[AVISO] Solução com tamanho {len(valores)} (esperado {n}). Valor bruto: {s}")
        pass
    return np.array(valores, dtype=int)

def calcular_f3_robustez_capacidade(x: np.ndarray, params: dict) -> float:
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

def calcular_f4_estabilidade(solucao: np.ndarray, params: dict) -> float:
    m = params["m"]
    a = params["a"]

    cargas = np.zeros(m)
    for t, agente in enumerate(solucao):
        cargas[agente] += a[agente, t]
        
    total_carga = np.sum(cargas)
    
    if total_carga > 0:
        p = cargas / total_carga
    else:
        p = np.zeros(m)

    p_pos = p[p > 0]
    entropia = -np.sum(p_pos * np.log(p_pos))
    
    return float(entropia)

def adicionar_criterios(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if df.empty:
        return df
        
    n = params["n"]
    df = df.copy()
    df["sol_array"] = df["solucao"].apply(lambda s: parse_solution_string(s, n))
    df["f3_robustez"] = df["sol_array"].apply(lambda sol: calcular_f3_robustez_capacidade(sol, params))
    df["f4_estabilidade"] = df["sol_array"].apply(lambda sol: calcular_f4_estabilidade(sol, params))
    df = df.drop(columns=["sol_array"])

    return df

if __name__ == "__main__":
    print("Carregando parâmetros...")
    params = load_params("data_5x50")

    if params:
        arquivos_processados = []
        dfs_unificacao = []
        
        for tipo, arquivo in [("Pe", "fronteiras_pe.csv"), ("Pw", "fronteiras_pw.csv")]:
            if os.path.exists(arquivo):
                print(f"Processando {arquivo}...")
                df = pd.read_csv(arquivo)
                df_ext = adicionar_criterios(df, params)
                nome_saida = f"fronteiras_{tipo.lower()}_com_criterios.csv"
                df_ext.to_csv(nome_saida, index=False)
                arquivos_processados.append(nome_saida)
                df_ext["metodo"] = tipo
                dfs_unificacao.append(df_ext)
            else:
                print(f"Aviso: {arquivo} não encontrado.")

        if dfs_unificacao:
            df_unificada = pd.concat(dfs_unificacao, ignore_index=True)
            cols = list(df_unificada.columns)
            head_cols = ['run', 'alt', 'f1', 'f2', 'f3_robustez', 'f4_estabilidade']
            tail_cols = [c for c in cols if c not in head_cols]
            df_unificada = df_unificada[head_cols + tail_cols]
            df_unificada.to_csv("fronteira_unificada_com_criterios.csv", index=False)
            arquivos_processados.append("fronteira_unificada_com_criterios.csv")

        print("\n=== Concluído com Sucesso ===")
        print("Arquivos gerados:")
        for arq in arquivos_processados:
            print(f" - {arq}")