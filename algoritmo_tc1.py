# parte 2 ( algoritmo de solucao )

import numpy as np
import random
from typing import Iterator, Callable, Tuple
import matplotlib.pyplot as plt

PENALIDADE_COEFICIENTE = 10000.0

class ProbData:
    def __init__(self, params: dict):
        self.m = params['m']              # nº de agentes
        self.n = params['n']              # nº de tarefas
        self.recurso = params['a']        # a[i,j] = recurso do agente i p/ tarefa j
        self.custo = params['c']          # c[i,j] = custo de atribuir j a i
        self.capacidade = params['b']     # b[i] = capacidade do agente i

# Estruturas de Vizinhança (n1, n2, n3, ...)
def gerar_vizinhos_shift_completo(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
    # Gera vizinhos de n1 - shift.
    n_tarefas = probdata.n
    m_agentes = probdata.m
    
    for j in range(n_tarefas):
        agente_atual = solucao[j]
        
        for novo_agente in range(m_agentes):
            if novo_agente != agente_atual:
                yield aplicar_shift(solucao, j, novo_agente)

def aplicar_shift(solucao: np.ndarray, tarefa: int, novo_agente: int) -> np.ndarray:
    # Shift move tarefa para novo_agente
    vizinho = solucao.copy()
    vizinho[tarefa] = novo_agente
    return vizinho

def aplicar_swap(solucao: np.ndarray, t1: int, t2: int) -> np.ndarray:
    # Swap troca os agentes das tarefas t1 e t2
    vizinho = solucao.copy()
    vizinho[t1], vizinho[t2] = vizinho[t2], vizinho[t1]
    return vizinho

def gerar_vizinhos_smart_shift(x: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
    # n1 inteligente com verificacao: move de agentes sobrecarregados para menos carregados (f2)
    cargas = np.zeros(probdata.m)
    for t, a in enumerate(x):
        cargas[int(a)] += probdata.recurso[int(a)][t]
    # ordena agentes p/ carga decrescente
    agentes_criticos = np.argsort(cargas)[::-1]
    
    for a_origem in agentes_criticos:
        tarefas_agente = [t for t, ag in enumerate(x) if ag == a_origem] # identifica tarefas desse agente
        if not tarefas_agente: continue
        
        carga_origem = cargas[a_origem]
        
        for t in tarefas_agente:
            for a_dest in range(probdata.m):
                if a_dest == a_origem: continue

                consumo = probdata.recurso[a_dest][t]
                if (cargas[a_dest] + consumo) < carga_origem: # só gera vizinho se o destino mais tranquilo que origem
                    yield aplicar_shift(x, t, a_dest)

def gerar_vizinhos_swap(solucao: np.ndarray, **kwargs) -> Iterator[np.ndarray]:
    # troca tarefas entre agentes de forma aleatoria
    n_tarefas = len(solucao)
    indices = list(range(n_tarefas))
    random.shuffle(indices) # estocástico
    
    for i in range(len(indices)):
        j1 = indices[i]
        for k in range(i + 1, len(indices)):
            j2 = indices[k]
            
            # nao troca se tiver com mesmo agente
            if solucao[j1] != solucao[j2]:
                yield aplicar_swap(solucao, j1, j2)

def first_improvement(solucao_atual: np.ndarray, 
                      func_geradora: Callable, 
                      func_obj: Callable, 
                      **kwargs) -> Tuple[np.ndarray, float, bool]:
    custo_atual = func_obj(solucao_atual)
    
    for vizinho in func_geradora(solucao_atual, **kwargs):
        custo_vizinho = func_obj(vizinho)
        
        if custo_vizinho < custo_atual:
            return vizinho, custo_vizinho, True
            
    return solucao_atual, custo_atual, False

def best_improvement(solucao_atual: np.ndarray, 
                     func_geradora: Callable, 
                     func_obj: Callable, 
                     **kwargs) -> Tuple[np.ndarray, float, bool]:
    
    custo_atual = func_obj(solucao_atual)
    melhor_vizinho = solucao_atual.copy()
    melhor_custo = custo_atual
    encontrou_melhoria = False
    
    for vizinho in func_geradora(solucao_atual, **kwargs):
        custo_vizinho = func_obj(vizinho)
        
        if custo_vizinho < melhor_custo:
            melhor_custo = custo_vizinho
            melhor_vizinho = vizinho
            encontrou_melhoria = True
            
    return melhor_vizinho, melhor_custo, encontrou_melhoria

def shake(x: np.ndarray, k: int, probdata: object) -> np.ndarray:
    y = x.copy()
    n_tarefas = len(x)
    m_agentes = probdata.m
    
    indices_tarefas = np.random.permutation(n_tarefas)
    
    if k == 1:
        # N1: shift de uma task
        t = indices_tarefas[0]
        agente_atual = y[t]
        possiveis = [a for a in range(m_agentes) if a != agente_atual]
        if possiveis:
            novo_agente = np.random.choice(possiveis)
            # mesma lógica do VND
            y = aplicar_shift(y, t, novo_agente)
            
    elif k == 2:
        # N2: Swap de duas tasks
        t1, t2 = indices_tarefas[0], indices_tarefas[1]
        if y[t1] != y[t2]:
            y = aplicar_swap(y, t1, t2)
        else:
            possiveis = [a for a in range(m_agentes) if a != y[t1]]
            if possiveis:
                y = aplicar_shift(y, t1, np.random.choice(possiveis))

    elif k >= 3: # Ruin & Recreate (baseado no LNS do handbook)
        qtd_ruin = min(k, int(n_tarefas * 0.2)) 
        tarefas_ruin = indices_tarefas[:qtd_ruin]
        
        for t in tarefas_ruin:
            melhor_agente = -1
            menor_custo_local = float('inf')
            
            for a in range(m_agentes):
                custo = probdata.custo[a][t]
                if custo < menor_custo_local:
                    menor_custo_local = custo
                    melhor_agente = a
            
            y[t] = melhor_agente

    return y

def vnd_hibrido(solucao_inicial: np.ndarray, 
                probdata: object, 
                func_obj: Callable,
                f_id: int) -> np.ndarray:
    # vnd hibrido com shift completo pra custo e inteligente pra equilibro
    x = solucao_inicial.copy()
    k = 1
    
    while k <= 2:
        melhorou = False
        novo_x = None
        
        if k == 1:
            # n1 shift
            if f_id == 1:
                novo_x, _, melhorou = first_improvement(
                    x, gerar_vizinhos_shift_completo, func_obj, probdata=probdata
                )
            else:
                novo_x, _, melhorou = first_improvement(
                    x, gerar_vizinhos_smart_shift, func_obj, probdata=probdata
                )
        elif k == 2:
            # n2 swap
            novo_x, _, melhorou = first_improvement(
                x, gerar_vizinhos_swap, func_obj, probdata=probdata
            )
            
        if melhorou:
            x = novo_x
            k = 1 
        else:
            k += 1

    x_refinado, _, melhorou_bi = best_improvement(
        x, gerar_vizinhos_smart_shift, func_obj, probdata=probdata
    )
    
    if melhorou_bi:
        x = x_refinado
    return x

def heuristica_construtiva_com_aleatoriedade(probdata, peso_custo=0.5):
    # heuristica construtiva com aleatoriedade e ponderacao custo vs equilibrio peso custo fixo em 0.5 inicialmente, mas testado variando
    m = probdata.m
    n = probdata.n

    solucao = np.empty(n, dtype=int)
    
    carga_atual = np.zeros(m)
    capacidade_max = probdata.capacidade
    
    max_custo_global = np.max(probdata.custo) if np.max(probdata.custo) > 0 else 1.0
    
    tarefas = list(range(n))
    random.shuffle(tarefas) # aleatoriedade na ordem de atribuição
    
    for tarefa in tarefas:
        scores = []
        
        for agente in range(m):
            custo_bruto = probdata.custo[agente][tarefa]
            recurso_necessario = probdata.recurso[agente][tarefa]

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
        carga_atual[melhor_agente] += probdata.recurso[melhor_agente][tarefa]
        
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
        agente_i = int(solucao[j])
        carga_agentes[agente_i] += a[agente_i, j]
        custo_total += c[agente_i, j]
    
    f1 = custo_total
    
    f2 = np.max(carga_agentes) - np.min(carga_agentes)
    
    violacao = np.maximum(0, carga_agentes - b)
    penalidade_total = PENALIDADE_COEFICIENTE * np.sum(violacao)
    
    if f_id == 1:
        return f1, penalidade_total
    else:
        return f2, penalidade_total


def get_fitness(solucao: np.ndarray, f_id: int, params: dict) -> float:
    # Fitness penalizado: F(x) = f(x) + P(x)
    f_obj, penalidade = calculate_objective_and_penalty(solucao, f_id, params)
    return f_obj + penalidade

def gvns(f_id: int, params: dict, k_max: int, max_iter_sem_melhora: int):    
    probdata = ProbData(params)
    
    func_fitness = lambda sol: get_fitness(sol, f_id, params)
    
    # peso_custo = 0.8 if f_id == 1 else 0.2
    best_sol = heuristica_construtiva_com_aleatoriedade(probdata, peso_custo=0.5)
    # best_sol = greedy_initial_solution(params, f_id=f_id)
    
    f_obj, penalidade = calculate_objective_and_penalty(best_sol, f_id, params)
    best_fitness = f_obj + penalidade
    
    print(f"Solucao Inicial (f_id={f_id}): Fitness = {best_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f})")
    
    iter_sem_melhora = 0
    iter_atual = 0
    
    historico_convergencia = [(iter_atual, best_fitness, f_obj, penalidade)]
    
    while iter_sem_melhora < max_iter_sem_melhora:
        k = 1
        
        while k <= k_max:
            solucao_shake = shake(best_sol, k, probdata)
            
            solucao_vnd = vnd_hibrido(solucao_shake, probdata, func_fitness, f_id)
            
            f_obj_second, p_second = calculate_objective_and_penalty(solucao_vnd, f_id, params)
            fitness_second = f_obj_second + p_second

            if fitness_second < best_fitness:
                best_sol = solucao_vnd
                best_fitness = fitness_second
                f_obj = f_obj_second
                penalidade = p_second

                print(f"Iter {iter_atual}: Novo Melhor! Fitness = {best_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f}) (k={k})")

                k = 1
                iter_sem_melhora = 0
                
                historico_convergencia.append((iter_atual, best_fitness, f_obj, penalidade))
            else:
                k += 1
                
        iter_sem_melhora += 1
        iter_atual += 1
        
        historico_convergencia.append((iter_atual, best_fitness, f_obj, penalidade))

    print(f"Fim do GVNS f_id={f_id}. Melhor Fitness: {best_fitness:.4f} (fn objetivo={f_obj:.2f}, p(x)={penalidade:.2f})")
    return best_sol, historico_convergencia

# metodos para entrega #2
def estimar_intervalos_normalizacao(params: dict, k_max: int, max_iter_sem_melhora: int):
    print("\n[Normalização] Estimando extremos de f1 e f2 com GVNS mono-objetivo...")
    
    sol_f1, _ = gvns(
        f_id=1,
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora
    )
    f1_f1, _ = calculate_objective_and_penalty(sol_f1, 1, params)
    f2_f1, _ = calculate_objective_and_penalty(sol_f1, 2, params)
    
    sol_f2, _ = gvns(
        f_id=2,
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora
    )
    f1_f2, _ = calculate_objective_and_penalty(sol_f2, 1, params)
    f2_f2, _ = calculate_objective_and_penalty(sol_f2, 2, params)
    
    f1_min = min(f1_f1, f1_f2)
    f1_max = max(f1_f1, f1_f2)
    f2_min = min(f2_f1, f2_f2)
    f2_max = max(f2_f1, f2_f2)
    
    print(f"[Normalização] f1 ∈ [{f1_min:.2f}, {f1_max:.2f}]")
    print(f"[Normalização] f2 ∈ [{f2_min:.2f}, {f2_max:.2f}]")
    
    norm_data = {
        'f1_min': f1_min,
        'f1_max': f1_max,
        'f2_min': f2_min,
        'f2_max': f2_max,
        'x_f1': sol_f1,
        'x_f2': sol_f2
    }
    return norm_data

def get_fitness_pw(solucao: np.ndarray,
                   params: dict,
                   w1: float,
                   w2: float,
                   norm_data: dict) -> float:
    f1, f2, pen_cap = calcular_valores_biojetivos(solucao, params)
    
    # trata divisao por zero
    denom_f1 = (norm_data['f1_max'] - norm_data['f1_min'])
    denom_f2 = (norm_data['f2_max'] - norm_data['f2_min'])
    if denom_f1 <= 0.0:
        denom_f1 = 1.0
    if denom_f2 <= 0.0:
        denom_f2 = 1.0
    
    f1_norm = (f1 - norm_data['f1_min']) / denom_f1
    f2_norm = (f2 - norm_data['f2_min']) / denom_f2
    
    F_pw = w1 * f1_norm + w2 * f2_norm + pen_cap
    return F_pw

def calcular_valores_biojetivos(solucao: np.ndarray, params: dict):
    n = params['n']
    m = params['m']
    a = params['a']
    b = params['b']
    c = params['c']
    
    carga_agentes = np.zeros(m)
    custo_total = 0.0
    
    for j in range(n):
        agente_i = int(solucao[j])
        carga_agentes[agente_i] += a[agente_i, j]
        custo_total += c[agente_i, j]
    
    f1 = custo_total
    f2 = np.max(carga_agentes) - np.min(carga_agentes)
    penalidade_total = PENALIDADE_COEFICIENTE * np.sum(np.maximum(0, carga_agentes - b))
    
    return f1, f2, penalidade_total

def get_fitness_epsilon(solucao: np.ndarray,
                        params: dict,
                        epsilon_f2: float) -> float:
    f1, f2, pen_cap = calcular_valores_biojetivos(solucao, params)
    penalidade_epslon = PENALIDADE_COEFICIENTE * max(0.0, f2 - epsilon_f2)
    
    F_e = f1 + pen_cap + penalidade_epslon
    return F_e

def rodar_abordagem_soma_ponderada(params: dict,
                    k_max: int,
                    max_iter_sem_melhora: int,
                    w1: float,
                    w2: float,
                    norm_data: dict):
    probdata = ProbData(params)
    
    peso_custo_heuristica = w1
    sol_inicial = heuristica_construtiva_com_aleatoriedade(probdata,
                                                           peso_custo=peso_custo_heuristica)
    
    func_fitness = lambda sol: get_fitness_pw(sol, params, w1, w2, norm_data)
    
    f_id_vnd = 1 if w1 >= w2 else 2
    
    label = f"Pw(w1={w1:.2f},w2={w2:.2f})"
    
    melhor_sol, historico = gvns_abordagem_escalar(
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora,
        func_fitness=func_fitness,
        probdata=probdata,
        solucao_inicial=sol_inicial,
        f_id_vnd=f_id_vnd,
        label=label
    )
    
    f1, f2, pen_cap = calcular_valores_biojetivos(melhor_sol, params)
    return melhor_sol, f1, f2, pen_cap, historico

def rodar_abordagem_epsilon(params: dict,
                         k_max: int,
                         max_iter_sem_melhora: int,
                         epsilon_f2: float):
    probdata = ProbData(params)
    
    sol_inicial = heuristica_construtiva_com_aleatoriedade(probdata,
                                                           peso_custo=0.8)
    
    func_fitness = lambda sol: get_fitness_epsilon(sol, params, epsilon_f2)
    
    f_id_vnd = 1
    label = f"Pe(eps={epsilon_f2:.2f})"
    
    melhor_sol, historico = gvns_abordagem_escalar(
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora,
        func_fitness=func_fitness,
        probdata=probdata,
        solucao_inicial=sol_inicial,
        f_id_vnd=f_id_vnd,
        label=label
    )
    
    f1, f2, pen_cap = calcular_valores_biojetivos(melhor_sol, params)
    return melhor_sol, f1, f2, pen_cap, historico

def filtrar_nao_dominadas(pontos: list):
    nao_dom = []
    for i, p in enumerate(pontos):
        dominado = False
        for j, q in enumerate(pontos):
            if j == i:
                continue

            if (q['f1'] <= p['f1'] and q['f2'] <= p['f2'] and
                (q['f1'] < p['f1'] or q['f2'] < p['f2'])):
                dominado = True
                break
        if not dominado:
            nao_dom.append(p)
    return nao_dom


def selecionar_bem_distribuidas(pontos: list, max_pontos: int = 20):
    if len(pontos) <= max_pontos:
        return pontos
    
    pontos_ordenados = sorted(pontos, key=lambda p: p['f1'])
    indices_continuos = np.linspace(0, len(pontos_ordenados) - 1, max_pontos)
    indices = sorted(set(int(round(idx)) for idx in indices_continuos))
    
    selecionados = [pontos_ordenados[i] for i in indices]
    return selecionados

def gerar_lista_pesos(num_pesos: int = 10):
    pesos = []
    for i in range(num_pesos + 1):
        w1 = i / num_pesos
        w2 = 1.0 - w1
        pesos.append((w1, w2))
    return pesos


def gerar_lista_epsilons(norm_data: dict, num_eps: int = 10):
    f2_min = norm_data['f2_min']
    f2_max = norm_data['f2_max']
    
    if f2_max <= f2_min:
        return [f2_min]
    
    epsilons = list(np.linspace(f2_min, f2_max, num_eps))
    return epsilons


def gvns_abordagem_escalar(params: dict,
                k_max: int,
                max_iter_sem_melhora: int,
                func_fitness: Callable[[np.ndarray], float],
                probdata: ProbData,
                solucao_inicial: np.ndarray,
                f_id_vnd: int,
                label: str = "escalar"):

    melhor_solucao = solucao_inicial.copy()
    
    f1, f2, pen_cap = calcular_valores_biojetivos(melhor_solucao, params)
    best_fitness = func_fitness(melhor_solucao)
    
    print(f"\n[GVNS-{label}] Solução inicial: "
          f"F={best_fitness:.4f} (f1={f1:.2f}, f2={f2:.2f}, P_cap={pen_cap:.2f})")
    
    iter_sem_melhora = 0
    iter_atual = 0
    historico_convergencia = [(iter_atual, best_fitness, f1, f2, pen_cap)]
    
    while iter_sem_melhora < max_iter_sem_melhora:
        k = 1
        
        while k <= k_max:
            solucao_shake = shake(melhor_solucao, k, probdata)
            
            solucao_vnd = vnd_hibrido(solucao_shake, probdata, func_fitness, f_id_vnd)
            
            f1_second, f2_second, pen_cap_second = calcular_valores_biojetivos(solucao_vnd, params)
            fitness_vnd = func_fitness(solucao_vnd)
            
            if fitness_vnd < best_fitness:
                melhor_solucao = solucao_vnd.copy()
                best_fitness = fitness_vnd
                f1 = f1_second
                f2 = f2_second
                pen_cap = pen_cap_second
                
                print(f"[GVNS-{label}] Iter {iter_atual}: Novo melhor! "
                      f"F={best_fitness:.4f} (f1={f1:.2f}, f2={f2:.2f}, P_cap={pen_cap:.2f}) (k={k})")
                
                k = 1
                iter_sem_melhora = 0
                historico_convergencia.append((iter_atual, best_fitness, f1, f2, pen_cap))
            else:
                k += 1
        
        iter_sem_melhora += 1
        iter_atual += 1
        historico_convergencia.append((iter_atual, best_fitness, f1, f2, pen_cap))
    
    print(f"[GVNS-{label}] Fim. Melhor F={best_fitness:.4f} "
          f"(f1={f1:.2f}, f2={f2:.2f}, P_cap={pen_cap:.2f})")
    
    return melhor_solucao, historico_convergencia

def executar_experimentos_multiobjetivo(params: dict,
                                        k_max: int,
                                        max_iter_sem_melhora: int,
                                        n_runs: int = 5):
    norm_data = estimar_intervalos_normalizacao(params, k_max, max_iter_sem_melhora)
    
    lista_pesos = gerar_lista_pesos(num_pesos=10) # combinações de (w1,w2)
    lista_eps   = gerar_lista_epsilons(norm_data, num_eps=10) # valores de ε
    
    fronteiras_pw_runs = []
    fronteiras_pe_runs = []
    
    for r in range(n_runs):
        print(f"\n===== [Pw] Execução {r+1} / {n_runs} =====")
        pontos_pw = []
        
        for (w1, w2) in lista_pesos:
            melhor_sol, f1, f2, pen_cap, _ = rodar_abordagem_soma_ponderada(
                params=params,
                k_max=k_max,
                max_iter_sem_melhora=max_iter_sem_melhora,
                w1=w1,
                w2=w2,
                norm_data=norm_data
            )
            pontos_pw.append({
                'f1': f1,
                'f2': f2,
                'solucao': melhor_sol,
                'w1': w1,
                'w2': w2,
                'pen_cap': pen_cap
            })
        
        nao_dom_pw = filtrar_nao_dominadas(pontos_pw)
        frente_pw = selecionar_bem_distribuidas(nao_dom_pw, max_pontos=20)
        
        fronteiras_pw_runs.append(frente_pw)
    
    for r in range(n_runs):
        print(f"\n===== [Pε] Execução {r+1} / {n_runs} =====")
        pontos_pe = []
        
        for eps in lista_eps:
            melhor_sol, f1, f2, pen_cap, _ = rodar_abordagem_epsilon(
                params=params,
                k_max=k_max,
                max_iter_sem_melhora=max_iter_sem_melhora,
                epsilon_f2=eps
            )
            pontos_pe.append({
                'f1': f1,
                'f2': f2,
                'solucao': melhor_sol,
                'epsilon': eps,
                'pen_cap': pen_cap
            })
        
        nao_dom_pe = filtrar_nao_dominadas(pontos_pe)
        frente_pe = selecionar_bem_distribuidas(nao_dom_pe, max_pontos=20)
        
        fronteiras_pe_runs.append(frente_pe)
    
    plot_fronteiras(
        fronteiras_pw_runs,
        titulo="Fronteiras de Pareto aproximadas - Soma Ponderada (Pw)",
        nome_arquivo="figuras_mo/fronteiras_pw.png"
    )
    
    plot_fronteiras(
        fronteiras_pe_runs,
        titulo="Fronteiras de Pareto aproximadas - Método ε-restrito (Pε)",
        nome_arquivo="figuras_mo/fronteiras_pe.png"
    )
    
    return fronteiras_pw_runs, fronteiras_pe_runs

### graficos, load data e main
### //////////////////////

def plot_fronteiras(lista_fronteiras_runs: list,
                    titulo: str,
                    nome_arquivo: str = None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.figure(figsize=(8, 6))

    todas_solucoes = []
    for idx, fronteira in enumerate(lista_fronteiras_runs):
        if not fronteira:
            continue
        f1_vals = [p['f1'] for p in fronteira]
        f2_vals = [p['f2'] for p in fronteira]

        plt.scatter(
            f1_vals,
            f2_vals,
            label=f"Execução {idx+1}",
            alpha=0.7,
            s=45
        )

        todas_solucoes.extend(fronteira)

    if todas_solucoes:
        frente_global = filtrar_nao_dominadas(todas_solucoes)
        frente_global_ordenada = sorted(frente_global, key=lambda p: p['f1'])

        f1_front = [p['f1'] for p in frente_global_ordenada]
        f2_front = [p['f2'] for p in frente_global_ordenada]

        plt.plot(
            f1_front,
            f2_front,
            'k-',
            linewidth=2,
            label='Fronteira Pareto (estimada)'
        )

        plt.scatter(
            f1_front,
            f2_front,
            facecolors='none',
            edgecolors='k',
            s=80,
            linewidths=1.8
        )

    plt.xlabel("f1(x) - Custo total")
    plt.ylabel("f2(x) - Desequilíbrio de carga")
    plt.title(titulo)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    if nome_arquivo is not None:
        pasta = os.path.dirname(nome_arquivo)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        plt.savefig(nome_arquivo, dpi=300)
        print(f"Figura salva em: {nome_arquivo}")

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
    
    K_MAX = 3
    MAX_ITER_SEM_MELHORA = 30
    N_RUNS = 5

    fronteiras_pw, fronteiras_pe = executar_experimentos_multiobjetivo(
        params=params,
        k_max=K_MAX,
        max_iter_sem_melhora=MAX_ITER_SEM_MELHORA,
        n_runs=N_RUNS
    )
