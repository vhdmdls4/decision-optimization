import numpy as np
import pandas as pd
import os

# ======================
# 1. Carregar parâmetros
# ======================

def load_params(prefix="data_5x50"):
    # Tenta carregar as matrizes principais
    try:
        a = np.loadtxt(f"{prefix}_a.csv", delimiter=",")   # recurso a[i,j]
        b = np.loadtxt(f"{prefix}_b.csv", delimiter=",")   # capacidade b[i]
    except Exception as e:
        print(f"Erro ao carregar arquivos principais (a, b): {e}")
        return None

    # Tenta carregar m e n, ou infere se não existirem
    try:
        m = int(np.loadtxt(f"{prefix}_m.csv", delimiter=","))
    except:
        m = len(b) # Infere pelo tamanho de b
        print(f"Aviso: {prefix}_m.csv não encontrado. Inferido m={m}")

    try:
        n = int(np.loadtxt(f"{prefix}_n.csv", delimiter=","))
    except:
        n = a.shape[1] # Infere pelas colunas de a
        print(f"Aviso: {prefix}_n.csv não encontrado. Inferido n={n}")

    params = {
        "a": a,
        "b": b,
        "m": m,
        "n": n,
    }
    return params

# =========================
# 2. Funções de avaliação
# =========================

def parse_solution_string(s: str, n: int) -> np.ndarray:
    """
    Converte a string '0,1,2,3,...' do CSV em um vetor numpy.
    Robusto contra aspas e colchetes.
    """
    # Limpeza extra para garantir que formatos como "[0, 1]" ou "'0', '1'" funcionem
    clean_s = str(s).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
    
    valores = [int(x) for x in clean_s.split(",") if x.strip() != ""]
    
    if len(valores) != n:
        print(f"[AVISO] Solução com tamanho {len(valores)} (esperado {n}). Valor bruto: {s}")
        pass
        
    return np.array(valores, dtype=int)

def calcular_f3_robustez_capacidade(x: np.ndarray, params: dict) -> float:
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

def calcular_f4_estabilidade(solucao: np.ndarray, params: dict) -> float:
    """
    f4(x) = Entropia da Distribuição de CARGA (Recursos) por agente.
    
    Mudança: Em vez de contar tarefas (discreto), usamos a soma de recursos (contínuo).
    Isso gera variabilidade nos valores de f4.
    
    Objetivo: Minimizar (Menor entropia = Carga concentrada? Depende da interpretação).
    Para manter coerência com 'Estabilidade Estrutural' (ordem), manteremos Minimizar.
    """
    m = params["m"]
    a = params["a"]

    # Calcula a carga total de cada agente (Recursos)
    cargas = np.zeros(m)
    for t, agente in enumerate(solucao):
        cargas[agente] += a[agente, t]
        
    total_carga = np.sum(cargas)
    
    # Probabilidades baseadas na fração da carga total
    # Se total_carga for 0 (impossível, mas seguro), evita erro
    if total_carga > 0:
        p = cargas / total_carga
    else:
        p = np.zeros(m)

    # Entropia de Shannon
    p_pos = p[p > 0]  # evita log(0)
    entropia = -np.sum(p_pos * np.log(p_pos))
    
    return float(entropia)

def adicionar_criterios(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Recebe DataFrame, processa f3 e f4.
    """
    if df.empty:
        return df
        
    n = params["n"]
    df = df.copy()

    # 1. Converter string -> vetor numpy
    # Aplica o parser linha a linha
    df["sol_array"] = df["solucao"].apply(lambda s: parse_solution_string(s, n))

    # 2. Calcular f3 e f4
    df["f3_robustez"] = df["sol_array"].apply(lambda sol: calcular_f3_robustez_capacidade(sol, params))
    df["f4_estabilidade"] = df["sol_array"].apply(lambda sol: calcular_f4_estabilidade(sol, params))

    # Remove a coluna temporária de array para salvar no CSV limpo
    df = df.drop(columns=["sol_array"])

    return df

# =========================
# 3. Main: gerar novos CSVs
# =========================

if __name__ == "__main__":
    print("Carregando parâmetros...")
    params = load_params("data_5x50")

    if params:
        # Carrega fronteiras originais
        # Verifica se arquivos existem para evitar crash
        arquivos_processados = []
        dfs_unificacao = []
        
        for tipo, arquivo in [("Pe", "fronteiras_pe.csv"), ("Pw", "fronteiras_pw.csv")]:
            if os.path.exists(arquivo):
                print(f"Processando {arquivo}...")
                df = pd.read_csv(arquivo)
                
                # Adiciona critérios
                df_ext = adicionar_criterios(df, params)
                
                # Salva individual
                nome_saida = f"fronteiras_{tipo.lower()}_com_criterios.csv"
                df_ext.to_csv(nome_saida, index=False)
                arquivos_processados.append(nome_saida)
                
                # Prepara para unificação
                df_ext["metodo"] = tipo # Marca a origem
                dfs_unificacao.append(df_ext)
            else:
                print(f"Aviso: {arquivo} não encontrado.")

        # Gera unificado se houver dados
        if dfs_unificacao:
            df_unificada = pd.concat(dfs_unificacao, ignore_index=True)
            
            # Ordenar colunas para ficar bonito (f1, f2, f3, f4 juntos)
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