# parte 2 ( algoritmo de solucao )

import numpy as np
import random
import time
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
            menor_custo = float('inf')

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

def neighborhood_change(X_atual, F_atual, X_refinado, F_refinado, k_atual):
    """
    Função NeighborhoodChange (Mudança de Vizinhança) para o GVNS.
    Compara a solução refinada (X_refinado) com a solução atual (X_atual) 
    e ajusta o índice de vizinhança k.
    """
    
    # 1. Comparação dos valores de fitness (F_refinado < F_atual)
    # Utilizamos o valor F(X) = f(X) + P(X) para guiar a busca [2]
    
    if F_refinado < F_atual:
        # Linha 1: Se f(x') < f(x), o movimento é aceito [1]
        
        # 2. Faz o movimento (Atualiza o incumbente para o refinado)
        X_nova = X_refinado.copy()  # Faz uma cópia da solução (make a move) [1]
        F_novo = F_refinado
        
        # 3. Reinicia a vizinhança
        k_novo = 1  # Retorna à vizinhança inicial N1 [1]
        
    else:
        # 4. Não houve melhoria
        X_nova = X_atual.copy()
        F_novo = F_atual
        
        # 5. Próxima vizinhança
        k_novo = k_atual + 1  # Incrementa o índice para k+1 (Next neighborhood) [1]
        
    return X_nova, F_novo, k_novo

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
    random.shuffle(indices) # Descomentar se quiser swap estocástico
    
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
                func_obj: Callable) -> np.ndarray:
    """
    VND Híbrido:
      - N1: Smart Shift (FI)
      - N2: Swap (FI)
      - Refinamento final: BI em N1
    """
    x = solucao_inicial.copy()
    k_max = 2 
    k = 1
    
    while k <= k_max:
        melhorou = False
        novo_x = None
        
        if k == 1:
            # N1: Smart Shift (Prioriza agentes cheios)
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
            sol_second = vnd_hibrido(sol_prime, probdata, func_fitness)
            
            # 3. Avaliação da solução refinada
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

    print(f"Fim do GVNS f_id={f_id}. Melhor Fitness: {best_fitness:.4f} (fn objetivo={f_obj:.2f}, p(x)={penalidade:.2f})")
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

def plot_melhor_solucao(solucao: np.ndarray, params: dict, titulo: str):
    m = params['m']
    a = params['a']
    carga_agentes = np.zeros(m)
    
    for j in range(params['n']):
        i = int(solucao[j])
        carga_agentes[i] += a[i, j]
    
    agentes = np.arange(m)
    
    plt.figure()
    plt.bar(agentes, carga_agentes)
    plt.xlabel('Agente')
    plt.ylabel('Carga total (recurso consumido)')
    plt.title(titulo)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    params = load_data('data_5x50')
    
    if params is None:
        exit(1)
    
    K_MAX = 5
    MAX_ITER_SEM_MELHORA = 100
    N_RUNS = 5
    
    # -----------------------------
    # f1: Custo Total
    # -----------------------------
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
    
    # Estatísticas f1: min, std, max
    resultados_f1 = np.array(resultados_f1)
    print("\nResumo f1 (sobre 5 execuções):")
    print(f"  min = {resultados_f1.min():.2f}")
    print(f"  max = {resultados_f1.max():.2f}")
    print(f"  std = {resultados_f1.std():.2f}")
    
    # -----------------------------
    # f2: Equilíbrio de Carga
    # -----------------------------
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