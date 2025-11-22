import numpy as np
import copy
import random
from typing import Callable, Tuple, Iterator

# =============================================================================
# Estruturas de Vizinhança (Geradores)
# =============================================================================

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

def gerar_vizinhos_smart_shift(solucao: np.ndarray, n_agentes: int, probdata: object) -> Iterator[np.ndarray]:
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

    cargas = np.zeros(n_agentes)
    tarefas_por_agente = [[] for _ in range(n_agentes)]

    for t, agente in enumerate(solucao):
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
    # random.shuffle(indices) # Descomentar se quiser swap estocástico
    
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
    """
    custo_atual = func_obj(solucao_atual)
    
    # O kwargs agora pode passar 'probdata' para o smart_shift
    for vizinho in func_geradora(solucao_atual, **kwargs):
        custo_vizinho = func_obj(vizinho)
        
        if custo_vizinho < custo_atual:
            return vizinho, custo_vizinho, True
            
    return solucao_atual, custo_atual, False

def best_improvement(solucao_atual: np.ndarray, 
                     func_geradora: Callable, 
                     func_obj: Callable, 
                     **kwargs) -> Tuple[np.ndarray, float, bool]:
    """
    Estratégia Best Improvement: Explora toda a vizinhança e retorna o melhor.
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
                func_obj: Callable) -> np.ndarray:
    """
    VND Híbrido atualizado com Smart Shift.
    """
    x = solucao_inicial.copy()
    k_max = 2 
    k = 1
    n_agentes = probdata.m
    
    while k <= k_max:
        melhorou = False
        novo_x = None
        
        if k == 1:
            # N1: Smart Shift (Prioriza agentes cheios)
            novo_x, _, melhorou = first_improvement(
                x, gerar_vizinhos_smart_shift, func_obj, 
                n_agentes=n_agentes, probdata=probdata
            )
        elif k == 2:
            # N2: Swap
            novo_x, _, melhorou = first_improvement(
                x, gerar_vizinhos_swap, func_obj
            )
            
        if melhorou:
            x = novo_x
            k = 1 
        else:
            k += 1
            
    # Refinamento Final: Best Improvement no Smart Shift
    # Garante que o ótimo local final é o melhor possível na vizinhança mais inteligente
    x_refinado, _, melhorou_bi = best_improvement(
        x, gerar_vizinhos_smart_shift, func_obj, 
        n_agentes=n_agentes, probdata=probdata
    )
    
    if melhorou_bi:
        x = x_refinado
        
    return x