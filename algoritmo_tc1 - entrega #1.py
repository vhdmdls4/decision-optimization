import numpy as np
import random
import time
from typing import Iterator, Callable, Tuple
import matplotlib.pyplot as plt

PENALIDADE_COEFICIENTE = 10000.0

class DadosProblema:
    def __init__(self, params: dict):
        self.n_agentes = int(params['m'])
        self.n_tarefas = int(params['n'])
        self.recurso = params['a']
        self.custo = params['c']
        self.capacidade = params['b']
        self.b = params['b']

def neighborhood_change(solucao_atual: np.ndarray, fitness_atual: float, solucao_refinada: np.ndarray, fitness_refinada: float, k_atual: int):
    if fitness_refinada < fitness_atual:
        solucao_nova = solucao_refinada.copy()
        fitness_novo = fitness_refinada
        k_novo = 1
    else:
        solucao_nova = solucao_atual.copy()
        fitness_novo = fitness_atual
        k_novo = k_atual + 1
    return solucao_nova, fitness_novo, k_novo

def gerar_vizinhos_shift_completo(solucao: np.ndarray, dados: DadosProblema) -> Iterator[np.ndarray]:
    n_tarefas = dados.n_tarefas
    n_agentes = dados.n_agentes
    for j in range(n_tarefas):
        agente_atual = int(solucao[j])
        for novo_agente in range(n_agentes):
            if novo_agente == agente_atual:
                continue
            yield aplicar_shift(solucao, j, novo_agente)

def gerar_vizinhos_swap_completo(solucao: np.ndarray, dados: DadosProblema) -> Iterator[np.ndarray]:
    n_tarefas = dados.n_tarefas
    for j1 in range(n_tarefas):
        for j2 in range(j1 + 1, n_tarefas):
            if solucao[j1] == solucao[j2]:
                continue
            yield aplicar_swap(solucao, j1, j2)

def aplicar_shift(solucao: np.ndarray, tarefa: int, novo_agente: int) -> np.ndarray:
    vizinho = solucao.copy()
    vizinho[tarefa] = novo_agente
    return vizinho

def aplicar_swap(solucao: np.ndarray, t1: int, t2: int) -> np.ndarray:
    vizinho = solucao.copy()
    vizinho[t1], vizinho[t2] = vizinho[t2], vizinho[t1]
    return vizinho

def gerar_vizinhos_smart_shift(solucao: np.ndarray, dados: DadosProblema) -> Iterator[np.ndarray]:
    n_agentes = dados.n_agentes
    cargas = np.zeros(n_agentes)
    tarefas_por_agente = [[] for _ in range(n_agentes)]
    for t, agente in enumerate(solucao):
        agente = int(agente)
        cargas[agente] += dados.recurso[agente][t]
        tarefas_por_agente[agente].append(t)
    agentes_ordenados = np.argsort(cargas)[::-1]
    for agente_origem in agentes_ordenados:
        if not tarefas_por_agente[agente_origem]:
            continue
        carga_origem = cargas[agente_origem]
        for tarefa in tarefas_por_agente[agente_origem]:
            for novo_agente in range(n_agentes):
                if novo_agente == agente_origem:
                    continue
                consumo_destino = dados.recurso[novo_agente][tarefa]
                nova_carga_destino = cargas[novo_agente] + consumo_destino
                if nova_carga_destino < carga_origem:
                    yield aplicar_shift(solucao, tarefa, novo_agente)

def gerar_vizinhos_swap(solucao: np.ndarray, dados: DadosProblema) -> Iterator[np.ndarray]:
    n_tarefas = len(solucao)
    indices = list(range(n_tarefas))
    random.shuffle(indices)
    for i in range(len(indices)):
        j1 = indices[i]
        for k in range(i + 1, len(indices)):
            j2 = indices[k]
            if solucao[j1] != solucao[j2]:
                yield aplicar_swap(solucao, j1, j2)

def first_improvement(solucao_atual: np.ndarray, func_geradora: Callable, func_obj: Callable, **kwargs) -> Tuple[np.ndarray, float, bool]:
    custo_atual = func_obj(solucao_atual)
    for vizinho in func_geradora(solucao_atual, **kwargs):
        custo_vizinho = func_obj(vizinho)
        if custo_vizinho < custo_atual:
            return vizinho, custo_vizinho, True
    return solucao_atual, custo_atual, False

def best_improvement(solucao_atual: np.ndarray, func_geradora: Callable, func_obj: Callable, **kwargs) -> Tuple[np.ndarray, float, bool]:
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

def shake(solucao: np.ndarray, k: int, dados: DadosProblema) -> np.ndarray:
    y = solucao.copy()
    n_tarefas = len(solucao)
    n_agentes = dados.n_agentes
    indices_tarefas = np.random.permutation(n_tarefas)
    if k == 1:
        t = indices_tarefas[0]
        agente_atual = y[t]
        possiveis = [a for a in range(n_agentes) if a != agente_atual]
        if possiveis:
            novo_agente = np.random.choice(possiveis)
            y = aplicar_shift(y, t, novo_agente)
    elif k == 2:
        t1, t2 = indices_tarefas[0], indices_tarefas[1]
        if y[t1] != y[t2]:
            y = aplicar_swap(y, t1, t2)
        else:
            possiveis = [a for a in range(n_agentes) if a != y[t1]]
            if possiveis:
                y = aplicar_shift(y, t1, np.random.choice(possiveis))
    elif k >= 3:
        qtd_ruin = min(k, int(n_tarefas * 0.2))
        tarefas_ruin = indices_tarefas[:qtd_ruin]
        for t in tarefas_ruin:
            melhor_agente = -1
            menor_custo_local = float('inf')
            for a in range(n_agentes):
                custo = dados.custo[a][t]
                if custo < menor_custo_local:
                    menor_custo_local = custo
                    melhor_agente = a
            y[t] = melhor_agente
    return y

def vnd_hibrido(solucao_inicial: np.ndarray, dados: DadosProblema, func_fitness: Callable, f_id: int) -> np.ndarray:
    solucao = solucao_inicial.copy()
    k_max = 2
    k = 1
    while k <= k_max:
        melhorou = False
        novo = None
        if k == 1:
            if f_id == 1:
                novo, _, melhorou = first_improvement(solucao, gerar_vizinhos_shift_completo, func_fitness, dados=dados)
            else:
                novo, _, melhorou = first_improvement(solucao, gerar_vizinhos_smart_shift, func_fitness, dados=dados)
        elif k == 2:
            novo, _, melhorou = first_improvement(solucao, gerar_vizinhos_swap, func_fitness, dados=dados)
        if melhorou:
            solucao = novo
            k = 1
        else:
            k += 1
    solucao_refinada, _, melhorou_bi = best_improvement(solucao, gerar_vizinhos_smart_shift, func_fitness, dados=dados)
    if melhorou_bi:
        solucao = solucao_refinada
    return solucao

def heuristica_construtiva_aleatoria(dados: DadosProblema, peso_custo=0.5):
    m = dados.n_agentes
    n = dados.n_tarefas
    solucao = np.empty(n, dtype=int)
    carga_atual = np.zeros(m)
    capacidade_max = dados.capacidade
    max_custo_global = np.max(dados.custo) if np.max(dados.custo) > 0 else 1.0
    tarefas = list(range(n))
    random.shuffle(tarefas)
    for tarefa in tarefas:
        scores = []
        for agente in range(m):
            custo_bruto = dados.custo[agente][tarefa]
            recurso_necessario = dados.recurso[agente][tarefa]
            score_custo = custo_bruto / max_custo_global
            nova_carga = carga_atual[agente] + recurso_necessario
            score_carga = nova_carga / capacidade_max[agente]
            penalidade = 0
            if nova_carga > capacidade_max[agente]:
                penalidade = 1000.0
            score_final = (peso_custo * score_custo) + ((1 - peso_custo) * score_carga) + penalidade
            score_final += random.uniform(0, 0.001)
            scores.append(score_final)
        melhor_agente = np.argmin(scores)
        solucao[tarefa] = melhor_agente
        carga_atual[melhor_agente] += dados.recurso[melhor_agente][tarefa]
    return solucao

def calcular_objetivo_penalidade(solucao: np.ndarray, f_id: int, params: dict):
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

def obter_fitness(solucao: np.ndarray, f_id: int, params: dict) -> float:
    f_obj, penalidade = calcular_objetivo_penalidade(solucao, f_id, params)
    return f_obj + penalidade

def gvns(f_id: int, params: dict, k_max: int, max_iter_sem_melhora: int):
    dados = DadosProblema(params)
    func_fitness = lambda sol: obter_fitness(sol, f_id, params)
    melhor_solucao = heuristica_construtiva_aleatoria(dados, peso_custo=0.5)
    f_obj, penalidade = calcular_objetivo_penalidade(melhor_solucao, f_id, params)
    melhor_fitness = f_obj + penalidade
    print(f"Solucao Inicial (f_id={f_id}): Fitness = {melhor_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f})")
    iter_sem_melhora = 0
    iter_atual = 0
    historico_convergencia = [(iter_atual, melhor_fitness, f_obj, penalidade)]
    while iter_sem_melhora < max_iter_sem_melhora:
        k = 1
        while k <= k_max:
            solucao_agitada = shake(melhor_solucao, k, dados)
            solucao_refinada = vnd_hibrido(solucao_agitada, dados, func_fitness, f_id)
            f_obj_ref, p_ref = calcular_objetivo_penalidade(solucao_refinada, f_id, params)
            fitness_ref = f_obj_ref + p_ref
            if fitness_ref < melhor_fitness:
                melhor_solucao = solucao_refinada
                melhor_fitness = fitness_ref
                f_obj = f_obj_ref
                penalidade = p_ref
                print(f"Iter {iter_atual}: Novo Melhor! Fitness = {melhor_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f}) (k={k})")
                k = 1
                iter_sem_melhora = 0
                historico_convergencia.append((iter_atual, melhor_fitness, f_obj, penalidade))
            else:
                k += 1
        iter_sem_melhora += 1
        iter_atual += 1
        historico_convergencia.append((iter_atual, melhor_fitness, f_obj, penalidade))
    print(f"Fim do GVNS f_id={f_id}. Melhor Fitness: {melhor_fitness:.4f} (fn objetivo={f_obj:.2f}, p(x)={penalidade:.2f})")
    return melhor_solucao, historico_convergencia

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
    resultados_f1 = []
    print("\n------ GVNS para f1 (Custo Total) ------")
    for r in range(N_RUNS):
        print(f"\n>>> Execução {r+1} / {N_RUNS} (f1)")
        start_time = time.time()
        melhor_solucao_f1, convergencia_f1 = gvns(f_id=1, params=params, k_max=K_MAX, max_iter_sem_melhora=MAX_ITER_SEM_MELHORA)
        tempo = time.time() - start_time
        f1_final, p1_final = calcular_objetivo_penalidade(melhor_solucao_f1, 1, params)
        f2_de_f1, _ = calcular_objetivo_penalidade(melhor_solucao_f1, 2, params)
        print(f"Tempo de execucao (f1, run {r+1}): {tempo:.2f} s")
        print(f"  -> Custo (f1)={f1_final:.2f}, equilibrio (f2)={f2_de_f1:.2f}, Penalidade (p(x))={p1_final:.2f}")
        resultados_f1.append(f1_final)
    resultados_f1 = np.array(resultados_f1)
    print("\nResumo f1 (sobre 5 execuções):")
    print(f"  min = {resultados_f1.min():.2f}")
    print(f"  max = {resultados_f1.max():.2f}")
    print(f"  std = {resultados_f1.std():.2f}")
    resultados_f2 = []
    print("\n------ GVNS para f2 (Equilíbrio) ------")
    for r in range(N_RUNS):
        print(f"\n>>> Execução {r+1} / {N_RUNS} (f2)")
        start_time = time.time()
        melhor_solucao_f2, convergencia_f2 = gvns(f_id=2, params=params, k_max=K_MAX, max_iter_sem_melhora=MAX_ITER_SEM_MELHORA)
        tempo = time.time() - start_time
        f1_de_f2, _ = calcular_objetivo_penalidade(melhor_solucao_f2, 1, params)
        f2_final, p2_final = calcular_objetivo_penalidade(melhor_solucao_f2, 2, params)
        print(f"Tempo de execucao (f2, run {r+1}): {tempo:.2f} s")
        print(f"  -> Custo (f1)={f1_de_f2:.2f}, equilibrio (f2)={f2_final:.2f}, Penalidade={p2_final:.2f}")
        resultados_f2.append(f2_final)
    resultados_f2 = np.array(resultados_f2)
    print("\nResumo f2 (sobre 5 execuções):")
    print(f"  min = {resultados_f2.min():.2f}")
    print(f"  max = {resultados_f2.max():.2f}")
    print(f"  std = {resultados_f2.std():.2f}")
