# parte 2 ( algoritmo de solucao )

import numpy as np
import random
from typing import Iterator, Callable, Tuple
import matplotlib.pyplot as plt

PENALIDADE_COEFICIENTE = 10000.0

class ProbData:
    """
    Wrapper para adaptar o dicionário de parâmetros ao formato de objeto
    esperado pelas novas funções de vizinhança.
    """
    def __init__(self, params: dict):
        self.m = params['m']              # nº de agentes
        self.n = params['n']              # nº de tarefas
        self.recurso = params['a']        # a[i,j] = recurso do agente i p/ tarefa j
        self.custo = params['c']          # c[i,j] = custo de atribuir j a i
        self.capacidade = params['b']     # b[i] = capacidade do agente i
        self.b = params['b']              # alias

def estimar_intervalos_normalizacao(params: dict, k_max: int, max_iter_sem_melhora: int):
    """
    Estima intervalos [min, max] para f1 e f2 usando as soluções
    (aproximadas) que minimizam f1 e f2 separadamente via GVNS.

    Esses pontos funcionam como aproximações dos pontos ideal/nadir
    no espaço de objetivos, para efeito de normalização na soma ponderada.
    """
    print("\n[Normalização] Estimando extremos de f1 e f2 com GVNS mono-objetivo...")
    
    # Melhor para f1 (custo)
    sol_f1, _ = gvns(
        f_id=1,
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora
    )
    f1_f1, _ = calculate_objective_and_penalty(sol_f1, 1, params)
    f2_f1, _ = calculate_objective_and_penalty(sol_f1, 2, params)
    
    # Melhor para f2 (equilíbrio)
    sol_f2, _ = gvns(
        f_id=2,
        params=params,
        k_max=k_max,
        max_iter_sem_melhora=max_iter_sem_melhora
    )
    f1_f2, _ = calculate_objective_and_penalty(sol_f2, 1, params)
    f2_f2, _ = calculate_objective_and_penalty(sol_f2, 2, params)
    
    # Intervalos aproximados (ideal/nadir aproximados)
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
    """
    Fitness para o método da soma ponderada (Pw).

    F_pw(x) = w1 * f1_norm(x) + w2 * f2_norm(x) + P_cap(x)

    onde f1_norm e f2_norm são normalizadas usando os intervalos
    estimados em 'norm_data'.
    """
    f1, f2, pen_cap = calcular_valores_biojetivos(solucao, params)
    
    # Evita divisão por zero
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
    """
    Calcula simultaneamente:
      - f1(x): custo total
      - f2(x): desequilíbrio de carga
      - P_cap(x): penalidade por violar capacidade

    Útil para as formulações escalares (Pw e Pε).
    """
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
    
    # f1: custo total
    f1 = custo_total
    
    # f2: diferença entre agente mais e menos carregado
    f2 = np.max(carga_agentes) - np.min(carga_agentes)
    
    # Violação de capacidade: max(0, carga - capacidade)
    violacao = np.maximum(0, carga_agentes - b)
    penalidade_total = PENALIDADE_COEFICIENTE * np.sum(violacao)
    
    return f1, f2, penalidade_total

def get_fitness_epsilon(solucao: np.ndarray,
                        params: dict,
                        epsilon_f2: float) -> float:
    """
    Fitness para o método ϵ-restrito (Pε):

    min f1(x)
    s.a. f2(x) <= epsilon_f2

    Implementado via penalização:
      F_e(x) = f1(x) + P_cap(x) + P_ε(x),
    onde P_ε(x) = PENALIDADE_COEFICIENTE * max(0, f2(x) - epsilon_f2).
    """
    f1, f2, pen_cap = calcular_valores_biojetivos(solucao, params)
    
    viol_epsilon = max(0.0, f2 - epsilon_f2)
    pen_eps = PENALIDADE_COEFICIENTE * viol_epsilon
    
    F_e = f1 + pen_cap + pen_eps
    return F_e

# =============================================================================
# Estruturas de Vizinhança (Geradores)
# =============================================================================

def gerar_vizinhos_shift_completo(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
    """
    Adaptação do 'explore_neighborhood_shift' do Algoritmo 2 para o Algoritmo 1.
    
    Diferença do Smart Shift:
    - NÃO verifica se o destino está menos carregado que a origem.
    - Tenta mover cada tarefa para TODOS os outros agentes.
    - É mais lento (O(n*m)), mas garante exploração completa para redução de custo (f1).
    """
    n_tarefas = probdata.n
    m_agentes = probdata.m
    
    # Itera sobre todas as tarefas (como no Alg 2)
    for j in range(n_tarefas):
        agente_atual = solucao[j]
        
        # Tenta mover para todos os outros agentes
        for novo_agente in range(m_agentes):
            if novo_agente == agente_atual:
                continue
            
            # Gera o vizinho incondicionalmente (a validação de melhora é feita pelo VND)
            yield aplicar_shift(solucao, j, novo_agente)

def aplicar_shift(solucao: np.ndarray, tarefa: int, novo_agente: int) -> np.ndarray:
    """Aplica o movimento Shift: move 'tarefa' para 'novo_agente'."""
    vizinho = solucao.copy()
    vizinho[tarefa] = novo_agente
    return vizinho

def aplicar_swap(solucao: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """Aplica o movimento Swap: troca os agentes das tarefas t1 e t2."""
    vizinho = solucao.copy()
    vizinho[t1], vizinho[t2] = vizinho[t2], vizinho[t1]
    return vizinho

def gerar_vizinhos_smart_shift(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
    """
    Gera vizinhos da estrutura N1 (Shift/Realocação) de forma otimizada e factível.

    Estratégia:
    - Move tarefas dos agentes mais carregados para agentes menos carregados, 
      sempre respeitando o limite de capacidade do agente destino.
    - O único filtro aplicado é para garantir o movimento na direção do equilíbrio
      (agente destino fica com menos carga que agente origem).
    - NÃO restringe movimentos apenas por eficiência local, pois a modelagem valoriza
      o equilíbrio global das cargas (f_E) e não o consumo individual da tarefa no destino.
    - Isso amplia as possibilidades de movimentos e aumenta o potencial exploratório do método,
      buscando sair de ótimos locais e atacar tanto a infactibilidade quanto o desequilíbrio.
    """

    n_agentes = probdata.m

    cargas = np.zeros(n_agentes)
    tarefas_por_agente = [[] for _ in range(n_agentes)]

    for t, agente in enumerate(solucao):
        agente = int(agente)
        cargas[agente] += probdata.recurso[agente][t]
        tarefas_por_agente[agente].append(t)

    agentes_criticos = np.argsort(cargas)[::-1]


    for agente_origem in agentes_criticos:
        if not tarefas_por_agente[agente_origem]:
            continue

        carga_origem = cargas[agente_origem]

        for tarefa in tarefas_por_agente[agente_origem]:
            for novo_agente in range(n_agentes):
                if novo_agente == agente_origem:
                    continue
                
                consumo_no_destino = probdata.recurso[novo_agente][tarefa]
                nova_carga_destino = cargas[novo_agente] + consumo_no_destino

                # Restringe apenas por capacidade e balanceamento
                # Se quiser vizinhança sempre factível, descomenta:
                # if nova_carga_destino > probdata.capacidade[novo_agente]:
                #     continue

                if nova_carga_destino < carga_origem:
                  yield aplicar_shift(solucao, tarefa, novo_agente)

def gerar_vizinhos_swap(solucao: np.ndarray, **kwargs) -> Iterator[np.ndarray]:
    """
    Gera vizinhos da estrutura N2 (Swap/Troca).
    Troca as atribuições de duas tarefas distintas.
    """
    n_tarefas = len(solucao)
    # Podemos adicionar aleatoriedade na ordem de visitação para não ser sempre determinístico
    indices = list(range(n_tarefas))
    random.shuffle(indices) # estocástico
    
    for i in range(len(indices)):
        j1 = indices[i]
        for k in range(i + 1, len(indices)):
            j2 = indices[k]
            
            # Se as tarefas já estão com o mesmo agente, a troca é inútil
            if solucao[j1] != solucao[j2]:
                yield aplicar_swap(solucao, j1, j2)

# =============================================================================
# Estratégias de Busca Local
# =============================================================================

def first_improvement(solucao_atual: np.ndarray, 
                      func_geradora: Callable, 
                      func_obj: Callable, 
                      **kwargs) -> Tuple[np.ndarray, float, bool]:
    """
    Estratégia First Improvement: Retorna na primeira melhoria encontrada.
    
    Args:
        solucao_atual: Vetor da solução atual.
        func_geradora: Função que gera vizinhos (ex: gerar_vizinhos_shift).
        func_obj: Função que calcula o custo/fitness.
        **kwargs: Argumentos extras para o gerador (ex: n_agentes).
    
    Returns:
        (nova_solucao, novo_custo, melhorou)
    """
    custo_atual = func_obj(solucao_atual)
    
    # Itera sobre os vizinhos gerados sob demanda
    for vizinho in func_geradora(solucao_atual, **kwargs):
        custo_vizinho = func_obj(vizinho)
        
        # Critério de aceite: Minimização
        if custo_vizinho < custo_atual:
            return vizinho, custo_vizinho, True
            
    return solucao_atual, custo_atual, False

def best_improvement(solucao_atual: np.ndarray, 
                     func_geradora: Callable, 
                     func_obj: Callable, 
                     **kwargs) -> Tuple[np.ndarray, float, bool]:
    """
    Estratégia Best Improvement: Explora toda a vizinhança e retorna o melhor.
    
    Returns:
        (melhor_vizinho, melhor_custo, melhorou)
    """
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

# =============================================================================
# Métodos Principais: Shake e VND
# =============================================================================

def shake(x: np.ndarray, k: int, probdata: object) -> np.ndarray:
    """
    Função de Perturbação (Shake) com Ruin & Recreate para k >= 3.
    """
    y = x.copy()
    n_tarefas = len(x)
    m_agentes = probdata.m
    
    # Seleciona tarefas aleatoriamente para perturbar
    indices_tarefas = np.random.permutation(n_tarefas)
    
    if k == 1:
        # N1: Realoca 1 tarefa aleatória (Pequeno impacto)
        t = indices_tarefas[0]
        agente_atual = y[t]
        possiveis = [a for a in range(m_agentes) if a != agente_atual]
        if possiveis:
            novo_agente = np.random.choice(possiveis)
            # mesma lógica do VND
            y = aplicar_shift(y, t, novo_agente)
            
    elif k == 2:
        # N2: Swap aleatório de 2 tarefas (Médio impacto)
        t1, t2 = indices_tarefas[0], indices_tarefas[1]
        if y[t1] != y[t2]:
            y = aplicar_swap(y, t1, t2)
        else:
            # Se caíram no mesmo agente, força um shift em uma delas para garantir perturbação
            possiveis = [a for a in range(m_agentes) if a != y[t1]]
            if possiveis:
                y = aplicar_shift(y, t1, np.random.choice(possiveis))

    elif k >= 3:
        # N3+: Ruin and Recreate (Alto impacto, inteligente)
        # "Ruin": Remove K tarefas da solução (desaloca virtualmente)
        # Intensidade: define quantas tarefas serão reatribuídas
        # Ex: k=3 -> 3 tarefas, k=4 -> 4 tarefas (até um limite de ~10-20%)
        qtd_ruin = min(k, int(n_tarefas * 0.2)) 
        tarefas_ruin = indices_tarefas[:qtd_ruin]
        
        # "Recreate": Reinsere essas tarefas de forma Gulosa (Greedy)
        # Para cada tarefa removida, escolhe o agente que oferece o menor custo de atribuição
        # Isso tende a levar a solução para um "vale" diferente, mas promissor.
        for t in tarefas_ruin:
            melhor_agente = -1
            menor_custo_local = float('inf')
            
            # Simplesmente busca o agente mais barato para essa tarefa (Greedy local)
            # Nota: Aqui ignoramos temporariamente a capacidade global para ser rápido,
            # ou podemos checar probdata.custo[a][t]
            for a in range(m_agentes):
                custo = probdata.custo[a][t]
                if custo < menor_custo_local:
                    menor_custo_local = custo
                    melhor_agente = a
            
            # Se quiser adicionar aleatoriedade no Recreate, pode pegar o 2º melhor as vezes
            y[t] = melhor_agente

    return y

def vnd_hibrido(solucao_inicial: np.ndarray, 
                probdata: object, 
                func_obj: Callable,
                f_id: int) -> np.ndarray:
    """
    VND Híbrido:
      - N1: Smart Shift (FI)
      - N2: Swap (FI)
      - Refinamento final: BI em N1
    
    Adaptativo:
      - Se f_id=1 (Custo): Usa Shift Irrestrito (lento mas acha preço bom)
      - Se f_id=2 (Equilíbrio): Usa Smart Shift (rápido e foca em balanceamento)
    """
    x = solucao_inicial.copy()
    k_max = 2 
    k = 1
    
    while k <= k_max:
        melhorou = False
        novo_x = None
        
        if k == 1:
            # N1: Smart Shift (Prioriza agentes cheios)
            if f_id == 1:
                # ESTRATÉGIA PARA CUSTO (F1)
                # Usa a lógica do Algoritmo 2 (Shift Irrestrito)
                # Isso permite mover tarefa de um agente vazio para um cheio se for mais barato.
                novo_x, _, melhorou = first_improvement(
                    x, gerar_vizinhos_shift_completo, func_obj, probdata=probdata
                )
            else:
                # ESTRATÉGIA PARA EQUILÍBRIO (F2)
                # Usa a lógica Smart: Só move de Cheio -> Vazio.
                # Muito mais rápido e direto para este objetivo.
                novo_x, _, melhorou = first_improvement(
                    x, gerar_vizinhos_smart_shift, func_obj, probdata=probdata
                )
        elif k == 2:
            # N2: Swap
            novo_x, _, melhorou = first_improvement(
                x, gerar_vizinhos_swap, func_obj, probdata=probdata
            )
            
        if melhorou:
            x = novo_x
            k = 1 
        else:
            k += 1
            
    # Refinamento Final: Best Improvement no Smart Shift
    # Garante que o ótimo local final é o melhor possível na vizinhança mais inteligente
    x_refinado, _, melhorou_bi = best_improvement(
        x, gerar_vizinhos_smart_shift, func_obj, probdata=probdata
    )
    
    if melhorou_bi:
        x = x_refinado
        
    return x

def heuristica_construtiva_com_aleatoriedade(probdata, peso_custo=0.5):
    """
    Constrói uma solução inicial ponderando Custo (fC) e Equilíbrio de Carga (fE).

    probdata: instância de ProbData
    peso_custo: 1.0 -> foca em custo, 0.0 -> foca em equilíbrio de carga
    
    Args:
        probdata: Dados do problema.
        peso_custo (float): Entre 0.0 e 1.0.
            - 1.0: Foca 100% em minimizar Custo ($f_C$).
            - 0.0: Foca 100% em minimizar Desequilíbrio ($f_E$).
            - 0.5: Meio termo equilibrado.
    """
    m = probdata.m
    n = probdata.n

    # Atribuição final: vetor [0..m-1] de tamanho n (um agente por tarefa)
    solucao = np.empty(n, dtype=int)
    
    # Rastreia a carga atual de cada agente para cálculo rápido
    carga_atual = np.zeros(m)
    capacidade_max = probdata.capacidade
    
    # Normalização: Para somar custo (ex: 20) com carga (ex: 0.5),
    # precisamos colocar tudo na mesma escala (0 a 1 aproximadamente).
    max_custo_global = np.max(probdata.custo) if np.max(probdata.custo) > 0 else 1.0
    
    # 1. Aleatoriedade: Embaralha a ordem de inserção das tarefas
    # Isso é vital para o GVNS não ficar preso sempre no mesmo ponto de partida
    tarefas = list(range(n))
    random.shuffle(tarefas)
    
    for tarefa in tarefas:
        scores = []
        
        for agente in range(m):
            custo_bruto = probdata.custo[agente][tarefa]
            recurso_necessario = probdata.recurso[agente][tarefa]
            
            # --- Cálculo dos Componentes do Score ---
            
            # Componente A: Custo Normalizado (0 a 1)
            # Quanto menor, melhor para f_C
            score_custo = custo_bruto / max_custo_global
            
            # Componente B: Impacto na Carga (0 a 1+)
            # Calcula quão cheio o agente ficará se receber essa tarefa.
            # Quanto menor (menos cheio), melhor para f_E e para viabilidade.
            nova_carga = carga_atual[agente] + recurso_necessario
            score_carga = nova_carga / capacidade_max[agente]
            
            # Penalidade Suave (Soft Constraint)
            # Se estourar a capacidade, aumenta drasticamente o score para evitar esse agente
            penalidade = 0
            if nova_carga > capacidade_max[agente]:
                penalidade = 1000.0  # Valor alto para desencorajar violações
            
            # --- Score Final ---
            # Combina os objetivos baseados no peso
            score_final = (peso_custo * score_custo) + \
                          ((1 - peso_custo) * score_carga) + \
                          penalidade
            
            # Adiciona um "ruído" aleatório muito pequeno (0.1%) para desempatar
            # agentes idênticos de forma variada
            score_final += random.uniform(0, 0.001)
            
            scores.append(score_final)
        
        # Escolhe o agente com o MENOR score (menor custo/impacto combinado)
        melhor_agente = np.argmin(scores)
        
        # Atribui
        solucao[tarefa] = melhor_agente
        carga_atual[melhor_agente] += probdata.recurso[melhor_agente][tarefa]
        
    return solucao

def calculate_objective_and_penalty(solucao: np.ndarray, f_id: int, params: dict):
    """
    Calcula f1 (custo) OU f2 (equilíbrio) + penalidade por violar capacidade.

    f_id = 1 -> fC (custo total)
    f_id = 2 -> fE (equilíbrio de carga)
    """
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
    
    # f1: custo total
    f1 = custo_total
    
    # f2: diferença entre agente mais e menos carregado
    f2 = np.max(carga_agentes) - np.min(carga_agentes)
    
    # Violação de capacidade: max(0, carga - capacidade)
    violacao = np.maximum(0, carga_agentes - b)
    penalidade_total = PENALIDADE_COEFICIENTE * np.sum(violacao)
    
    if f_id == 1:
        return f1, penalidade_total
    else:
        return f2, penalidade_total


def get_fitness(solucao: np.ndarray, f_id: int, params: dict) -> float:
    """
    Fitness penalizado: F(x) = f(x) + P(x), para usar no GVNS/VND.
    """
    f_obj, penalidade = calculate_objective_and_penalty(solucao, f_id, params)
    return f_obj + penalidade

def gvns(f_id: int, params: dict, k_max: int, max_iter_sem_melhora: int):    
    """
    GVNS mono-objetivo para f_id (1: custo, 2: equilíbrio).
    Usa:
      - heurística construtiva para solução inicial;
      - shake(k);
      - VND híbrido (FI N1/N2 + BI N1) como busca local.
    """
    probdata = ProbData(params)
    
    # Função fitness penalizada F(x) para este f_id
    func_fitness = lambda sol: get_fitness(sol, f_id, params)
    
    # peso_custo = 0.8 if f_id == 1 else 0.2
    # Solução inicial
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
            
            # 1. Shaking
            sol_prime = shake(best_sol, k, probdata)
            
            # 2. VND (busca local)
            sol_second = vnd_hibrido(sol_prime, probdata, func_fitness, f_id)
            
            # 3. Avaliação da solução refinada
            f_obj_second, p_second = calculate_objective_and_penalty(sol_second, f_id, params)
            fitness_second = f_obj_second + p_second

            if fitness_second < best_fitness:
                best_sol = sol_second
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

def rodar_abordagem_soma_ponderada(params: dict,
                    k_max: int,
                    max_iter_sem_melhora: int,
                    w1: float,
                    w2: float,
                    norm_data: dict):
    """
    Executa uma vez o GVNS usando a formulação Pw (soma ponderada),
    para um par de pesos (w1, w2), w1 + w2 = 1.

    Retorna:
      - melhor_sol: solução em termos de X
      - f1, f2, pen_cap: valores biobjetivo da solução
      - historico: histórico de convergência (F_pw, f1, f2, P_cap)
    """
    probdata = ProbData(params)
    
    # Peso da heurística construtiva alinhado com ênfase da soma ponderada:
    peso_custo_heur = w1  # w1 ~ importância relativa de f1 (custo)
    sol_inicial = heuristica_construtiva_com_aleatoriedade(probdata,
                                                           peso_custo=peso_custo_heur)
    
    func_fitness = lambda sol: get_fitness_pw(sol, params, w1, w2, norm_data)
    
    # Se w1 >= w2, focamos a vizinhança no custo; senão, no equilíbrio
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
    """
    Executa uma vez o GVNS usando a formulação Pε (ϵ-restrito):

      min f1(x)
      s.a. f2(x) <= epsilon_f2

    via penalização.

    Retorna:
      - melhor_sol: solução em termos de X
      - f1, f2, pen_cap: valores biobjetivo da solução
      - historico: histórico de convergência
    """
    probdata = ProbData(params)
    
    # Para Pε, f1 é objetivo principal -> heurística puxa mais para custo
    sol_inicial = heuristica_construtiva_com_aleatoriedade(probdata,
                                                           peso_custo=0.8)
    
    func_fitness = lambda sol: get_fitness_epsilon(sol, params, epsilon_f2)
    
    # Aqui o VND é orientado para custo (f1) como objetivo principal
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
    """
    Recebe uma lista de dicionários com pelo menos:
      - 'f1': valor de f1(x)
      - 'f2': valor de f2(x)

    Retorna apenas as soluções não-dominadas (minimização em f1 e f2).
    """
    nao_dom = []
    for i, p in enumerate(pontos):
        dominado = False
        for j, q in enumerate(pontos):
            if j == i:
                continue
            # q domina p se:
            # q.f1 <= p.f1 e q.f2 <= p.f2 e pelo menos uma estrita
            if (q['f1'] <= p['f1'] and q['f2'] <= p['f2'] and
                (q['f1'] < p['f1'] or q['f2'] < p['f2'])):
                dominado = True
                break
        if not dominado:
            nao_dom.append(p)
    return nao_dom


def selecionar_bem_distribuidas(pontos: list, max_pontos: int = 20):
    """
    Recebe uma lista de pontos (dicionários com 'f1' e 'f2') já não-dominados.
    Se o número de pontos excede max_pontos, seleciona aproximadamente
    max_pontos pontos bem distribuídos ao longo da fronteira em f1.

    Estratégia simples:
      - ordena por f1 crescente;
      - pega índices igualmente espaçados.
    """
    if len(pontos) <= max_pontos:
        return pontos
    
    pontos_ordenados = sorted(pontos, key=lambda p: p['f1'])
    indices_continuos = np.linspace(0, len(pontos_ordenados) - 1, max_pontos)
    indices = sorted(set(int(round(idx)) for idx in indices_continuos))
    
    selecionados = [pontos_ordenados[i] for i in indices]
    return selecionados

def gerar_lista_pesos(num_pesos: int = 10):
    """
    Gera uma lista de pares (w1, w2) igualmente espaçados em [0,1],
    com w1 + w2 = 1.

    Ex.: num_pesos = 10 -> 11 pares de pesos.
    """
    pesos = []
    for i in range(num_pesos + 1):
        w1 = i / num_pesos
        w2 = 1.0 - w1
        pesos.append((w1, w2))
    return pesos


def gerar_lista_epsilons(norm_data: dict, num_eps: int = 10):
    """
    Gera uma lista de valores de epsilon para o método Pε,
    igualmente espaçados no intervalo aproximado de f2.

    Usa os valores de f2_min e f2_max estimados em 'estimar_intervalos_normalizacao'.
    """
    f2_min = norm_data['f2_min']
    f2_max = norm_data['f2_max']
    
    if f2_max <= f2_min:
        # Intervalo degenerado -> gera pelo menos um epsilon
        return [f2_min]
    
    epsilons = list(np.linspace(f2_min, f2_max, num_eps))
    return epsilons

def plot_fronteiras(lista_fronteiras_runs: list,
                    titulo: str,
                    nome_arquivo: str = None):
    """
    Plota as fronteiras (f1, f2) de várias execuções sobrepostas.
    Cada elemento de lista_fronteiras_runs é uma lista de pontos (dicts).

    Ex.: lista_fronteiras_runs tem tamanho 5 (5 execuções),
    e cada fronteira tem até 20 pontos.
    """
    plt.figure()
    
    for idx, fronteira in enumerate(lista_fronteiras_runs):
        f1_vals = [p['f1'] for p in fronteira]
        f2_vals = [p['f2'] for p in fronteira]
        plt.scatter(f1_vals, f2_vals, label=f"Execução {idx+1}")
    
    plt.xlabel("f1(x) - Custo total")
    plt.ylabel("f2(x) - Desequilíbrio de carga")
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if nome_arquivo is not None:
        import os
        pasta = os.path.dirname(nome_arquivo)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        plt.savefig(nome_arquivo, dpi=300)
        print(f"Figura salva em: {nome_arquivo}")
    
    # Se quiser visualizar interativamente, descomente:
    # plt.show()

def gvns_abordagem_escalar(params: dict,
                k_max: int,
                max_iter_sem_melhora: int,
                func_fitness: Callable[[np.ndarray], float],
                probdata: ProbData,
                solucao_inicial: np.ndarray,
                f_id_vnd: int,
                label: str = "scalar"):
    """
    GVNS genérico para uma formulação escalar qualquer (Pw ou Pε).

    - func_fitness: função F(x) que será minimizada.
    - f_id_vnd: controla o comportamento do VND híbrido:
        1 -> foca em custo (usa shift completo)
        2 -> foca em equilíbrio (usa smart shift)
    """
    melhor_solucao = solucao_inicial.copy()
    
    # Avalia solução inicial em termos de f1, f2 e fitness escalar
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
            # 1. Shaking
            sol_prime = shake(melhor_solucao, k, probdata)
            
            # 2. VND (busca local) para a formulação escalar
            sol_second = vnd_hibrido(sol_prime, probdata, func_fitness, f_id_vnd)
            
            # 3. Avaliação da solução refinada
            f1_second, f2_second, pen_cap_second = calcular_valores_biojetivos(sol_second, params)
            fitness_second = func_fitness(sol_second)
            
            if fitness_second < best_fitness:
                melhor_solucao = sol_second
                best_fitness = fitness_second
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
    """
    Executa os experimentos multiobjetivo para as abordagens escalares:
      - Pw (soma ponderada)
      - Pε (epsilon restrito)

    Atende aos itens (b), (c) e (d):

      (b) Usa o GVNS/VND do item (ii) como motor mono-objetivo.
      (c) Executa 5 vezes para cada abordagem.
      (d) Em cada execução, extrai no máximo 20 soluções não-dominadas.
    """
    # 1) Estima intervalos de normalização e pontos âncora para f1 e f2
    norm_data = estimar_intervalos_normalizacao(params, k_max, max_iter_sem_melhora)
    
    # 2) Define conjuntos de pesos (Pw) e epsilons (Pε)
    lista_pesos = gerar_lista_pesos(num_pesos=10)     # -> 11 combinações de (w1,w2)
    lista_eps   = gerar_lista_epsilons(norm_data, num_eps=10)  # -> 10 valores de ε
    
    fronteiras_pw_runs = []
    fronteiras_pe_runs = []
    
    # ------------------------------
    # Abordagem Pw (Soma Ponderada)
    # ------------------------------
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
        
        # Filtra não-dominadas e reduz para <= 20 pontos
        nao_dom_pw = filtrar_nao_dominadas(pontos_pw)
        frente_pw = selecionar_bem_distribuidas(nao_dom_pw, max_pontos=20)
        
        fronteiras_pw_runs.append(frente_pw)
    
    # --------------------------------------
    # Abordagem Pε (método epsilon-restrito)
    # --------------------------------------
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
    
    # 3) Geração dos gráficos com as 5 fronteiras sobrepostas
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
    
    # EXecucacao (soma ponderada e epsilon-restrito) (Pw, Pε)
    fronteiras_pw, fronteiras_pe = executar_experimentos_multiobjetivo(
        params=params,
        k_max=K_MAX,
        max_iter_sem_melhora=MAX_ITER_SEM_MELHORA,
        n_runs=N_RUNS
    )
    
    # Se quiser, aqui você pode salvar as frentes em CSV
    # ou imprimir algumas soluções exemplo para discutir no relatório.
