# parte 2 ( algoritmo de solucao )

import numpy as np
import random
import time

PENALTY_COEFFICIENT = 10000.0

def heuristica_construtiva_com_aleatoriedade(params, peso_custo=0.5):
    n = params['n']
    m = params['m']
    
    solucao = np.empty(n, dtype=int)
    
    carga_atual = np.zeros(m)
    capacidade_max = params['b']
    
    max_custo_global = np.max(params['c']) if np.max(params['c']) > 0 else 1.0

    tarefas = list(range(n))
    random.shuffle(tarefas)
    
    for tarefa in tarefas:
        scores = []
        
        for agente in range(m):
            custo_bruto = params['c'][agente][tarefa]
            recurso_necessario = params['a'][agente][tarefa]

            score_custo = custo_bruto / max_custo_global
            
            nova_carga = carga_atual[agente] + recurso_necessario
            score_carga = nova_carga / capacidade_max[agente]

            penalidade = 0
            if nova_carga > capacidade_max[agente]:
                penalidade = 1000.0

            score_final = (peso_custo * score_custo) + \
                          ((1 - peso_custo) * score_carga) + \
                          penalidade

            score_final += random.uniform(0, 0.001)
            
            scores.append(score_final)
        
        melhor_agente = np.argmin(scores)
        
        solucao[tarefa] = melhor_agente
        carga_atual[melhor_agente] += params['a'][melhor_agente][tarefa]
        
    return solucao

def greedy_initial_solution(params: dict, f_id: int) -> np.ndarray:

    n = params['n']
    m = params['m']
    a = params['a']
    b = params['b']
    c = params['c']
    
    solucao = np.zeros(n, dtype=int)
    carga_agentes = np.zeros(m) 
    
    if f_id == 1:
        for j in range(n):
            melhor_agente = -1

            agentes_ordenados_custo = np.argsort(c[:, j])
            
            for agente_i in agentes_ordenados_custo:
                if carga_agentes[agente_i] + a[agente_i, j] <= b[agente_i]:
                    melhor_agente = agente_i
                    break
            
            if melhor_agente == -1:
                melhor_agente = np.argmin(carga_agentes)

            solucao[j] = melhor_agente
            carga_agentes[melhor_agente] += a[melhor_agente, j]
            
    else:
        for j in range(n):
            melhor_agente = -1
            
            agentes_ordenados_carga = np.argsort(carga_agentes)
            
            for agente_i in agentes_ordenados_carga:
                if carga_agentes[agente_i] + a[agente_i, j] <= b[agente_i]:
                    melhor_agente = agente_i
                    break
            
            if melhor_agente == -1:
                melhor_agente = agentes_ordenados_carga[0]

            solucao[j] = melhor_agente
            carga_agentes[melhor_agente] += a[melhor_agente, j]

    return solucao

def calculate_objective_and_penalty(solucao: np.ndarray, f_id: int, params: dict):
    n = params['n']
    m = params['m']
    a = params['a']
    b = params['b']
    c = params['c']
    
    carga_agentes = np.zeros(m)
    custo_total = 0.0
    
    for j in range(n):
        agente_i = solucao[j]
        carga_agentes[agente_i] += a[agente_i, j]
        custo_total += c[agente_i, j]
    
    f1 = custo_total
    
    f2 = np.max(carga_agentes) - np.min(carga_agentes)
    
    violacao = np.maximum(0, carga_agentes - b)
    penalidade_total = PENALTY_COEFFICIENT * np.sum(violacao)
    
    if f_id == 1:
        return f1, penalidade_total
    else:
        return f2, penalidade_total

def get_fitness(solucao: np.ndarray, f_id: int, params: dict) -> float:
    f_obj, penalidade = calculate_objective_and_penalty(solucao, f_id, params)
    return f_obj + penalidade

def shake_k_shift(solucao: np.ndarray, k: int, params: dict) -> np.ndarray:
    n = params['n']
    m = params['m']
    nova_solucao = solucao.copy()
    
    tarefas_para_mover = random.sample(range(n), k)
    
    for j in tarefas_para_mover:
        agente_atual = nova_solucao[j]
        
        novo_agente = random.randint(0, m - 1)
        while novo_agente == agente_atual:
            novo_agente = random.randint(0, m - 1)
            
        nova_solucao[j] = novo_agente
        
    return nova_solucao

def explore_neighborhood_shift(solucao: np.ndarray, f_id: int, params: dict, best_fitness: float) -> (np.ndarray, float, bool):
    n = params['n']
    m = params['m']
    
    for j in range(n):
        agente_atual = solucao[j]
        
        for novo_agente in range(m):
            if novo_agente == agente_atual:
                continue
                
            vizinho = solucao.copy()
            vizinho[j] = novo_agente
            
            fitness_vizinho = get_fitness(vizinho, f_id, params)
            
            if fitness_vizinho < best_fitness:
                return vizinho, fitness_vizinho, True
    
    return solucao, best_fitness, False

def explore_neighborhood_switch(solucao: np.ndarray, f_id: int, params: dict, best_fitness: float) -> (np.ndarray, float, bool):
    n = params['n']
    
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            vizinho = solucao.copy()
            vizinho[j1] = solucao[j2]
            vizinho[j2] = solucao[j1]

            fitness_vizinho = get_fitness(vizinho, f_id, params)
            
            if fitness_vizinho < best_fitness:
                return vizinho, fitness_vizinho, True
                
    return solucao, best_fitness, False

def local_search_vnd(solucao: np.ndarray, f_id: int, params: dict) -> np.ndarray:
    l_max = 2
    l = 1
    
    current_sol = solucao.copy()
    best_fitness = get_fitness(current_sol, f_id, params)
    
    while l <= l_max:
        melhorou = False
        
        if l == 1:
            nova_sol, novo_fitness, melhorou = explore_neighborhood_shift(current_sol, f_id, params, best_fitness)
        elif l == 2:
            nova_sol, novo_fitness, melhorou = explore_neighborhood_switch(current_sol, f_id, params, best_fitness)
            
        if melhorou:
            current_sol = nova_sol
            best_fitness = novo_fitness
            l = 1
        else:
            l += 1
            
    return current_sol

def gvns(f_id: int, params: dict, k_max: int, max_iter_sem_melhora: int) -> (np.ndarray, list):
    best_sol = greedy_initial_solution(params, f_id=f_id)
    # best_sol = heuristica_construtiva_com_aleatoriedade(params, peso_custo=0.5)
    
    f_obj, penalidade = calculate_objective_and_penalty(best_sol, f_id, params)
    best_fitness = f_obj + penalidade
    
    print(f"Solucao Inicial (f_id={f_id}): Fitness = {best_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f})")
    
    iter_sem_melhora = 0
    iter_atual = 0
    
    historico_convergencia = [(iter_atual, best_fitness, f_obj, penalidade)]
    
    while iter_sem_melhora < max_iter_sem_melhora:
        k = 1
        
        while k <= k_max:
            
            sol_prime = shake_k_shift(best_sol, k, params)
            
            sol_second = local_search_vnd(sol_prime, f_id, params)
            
            f_obj_second, p_second = calculate_objective_and_penalty(sol_second, f_id, params)
            fitness_second = f_obj_second + p_second
            
            if fitness_second < best_fitness:
                best_sol = sol_second
                best_fitness = fitness_second
                f_obj = f_obj_second
                penalidade = p_second
                
                k = 1
                iter_sem_melhora = 0
                
                print(f"Iter {iter_atual}: Novo Melhor! Fitness = {best_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f}) (k={k})")
                
                historico_convergencia.append((iter_atual, best_fitness, f_obj, penalidade))
            else:
                k += 1
                
        iter_sem_melhora += 1
        iter_atual += 1
        
        historico_convergencia.append((iter_atual, best_fitness, f_obj, penalidade))

    print(f"Fim do GVNS. Melhor Fitness: {best_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f})")
    return best_sol, historico_convergencia

def load_data(prefix='data_5x50'):

    try:
        a = np.loadtxt(f'{prefix}_a.csv', delimiter=',')
        b = np.loadtxt(f'{prefix}_b.csv', delimiter=',')
        c = np.loadtxt(f'{prefix}_c.csv', delimiter=',')
        m_scalar = np.loadtxt(f'{prefix}_m.csv', delimiter=',')
        n_scalar = np.loadtxt(f'{prefix}_n.csv', delimiter=',')

        params = {
            'a': a,
            'b': b,
            'c': c,
            'm': int(m_scalar),
            'n': int(n_scalar)
        }
        
        print(f"Dados carregados: {params['m']} agentes, {params['n']} tarefas.")
        return params
        
    except IOError as e:
        print(f"Erro ao carregar arquivos de dados com prefixo '{prefix}'.")
        print(f"Verifique se os arquivos .csv estão no mesmo diretório.")
        print(f"Erro: {e}")
        return None

if __name__ == "__main__":
    
    params = load_data('data_5x50')
    
    if params is None:
        exit(1)
    
    K_MAX = 5
    MAX_ITER_SEM_MELHORA = 100
    N_RUNS = 5
    
    # f1: Custo Total
    resultados_f1 = []
    
    print("\n------ GVNS para f1 (Custo Total) ------")
    for r in range(N_RUNS):
        print(f"\n>>> Execução {r+1} / {N_RUNS} (f1)")
        start_time = time.time()
        
        melhor_solucao_f1, convergencia_f1 = gvns(
            f_id=1, 
            params=params, 
            k_max=K_MAX, 
            max_iter_sem_melhora=MAX_ITER_SEM_MELHORA
        )
        
        tempo = time.time() - start_time
        f1_final, p1_final = calculate_objective_and_penalty(melhor_solucao_f1, 1, params)
        f2_de_f1, _ = calculate_objective_and_penalty(melhor_solucao_f1, 2, params)
        
        print(f"Tempo de execucao (f1, run {r+1}): {tempo:.2f} s")
        print(f"  -> Custo (f1)={f1_final:.2f}, equilibrio (f2)={f2_de_f1:.2f}, Penalidade (p(x))={p1_final:.2f}")
        
        resultados_f1.append(f1_final)
    
    resultados_f1 = np.array(resultados_f1)
    print("\nResumo f1 (sobre 5 execuções):")
    print(f"  min = {resultados_f1.min():.2f}")
    print(f"  max = {resultados_f1.max():.2f}")
    print(f"  std = {resultados_f1.std():.2f}")
    
    # f2: Equilíbrio de Carga
    resultados_f2 = []
    
    print("\n------ GVNS para f2 (Equilíbrio) ------")
    for r in range(N_RUNS):
        print(f"\n>>> Execução {r+1} / {N_RUNS} (f2)")
        start_time = time.time()
        
        melhor_solucao_f2, convergencia_f2 = gvns(
            f_id=2, 
            params=params, 
            k_max=K_MAX, 
            max_iter_sem_melhora=MAX_ITER_SEM_MELHORA
        )
        
        tempo = time.time() - start_time
        f1_de_f2, _ = calculate_objective_and_penalty(melhor_solucao_f2, 1, params)
        f2_final, p2_final = calculate_objective_and_penalty(melhor_solucao_f2, 2, params)
        
        print(f"Tempo de execucao (f2, run {r+1}): {tempo:.2f} s")
        print(f"  -> Custo (f1)={f1_de_f2:.2f}, equilibrio (f2)={f2_final:.2f}, Penalidade={p2_final:.2f}")
        
        resultados_f2.append(f2_final)
    
    resultados_f2 = np.array(resultados_f2)
    print("\nResumo f2 (sobre 5 execuções):")
    print(f"  min = {resultados_f2.min():.2f}")
    print(f"  max = {resultados_f2.max():.2f}")
    print(f"  std = {resultados_f2.std():.2f}")