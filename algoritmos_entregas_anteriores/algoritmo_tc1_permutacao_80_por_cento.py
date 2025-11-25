import numpy as np
import random
import time
from typing import Iterator, Callable, Tuple
import matplotlib.pyplot as plt

PENALIDADE_COEFICIENTE = 10000.0

class ProbData:
  def __init__(self, params: dict):
    self.m = params['m']
    self.n = params['n']
    self.recurso = params['a']
    self.custo = params['c']
    self.capacidade = params['b']
    self.b = params['b']

def gerar_vizinhos_shift_completo(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
  n_tarefas = probdata.n
  m_agentes = probdata.m

  for j in range(n_tarefas):
    agente_atual = solucao[j]

    for novo_agente in range(m_agentes):

      if novo_agente == agente_atual:
        continue

      yield aplicar_shift(solucao, j, novo_agente)

def gerar_vizinhos_swap_completo(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
  n_tarefas = probdata.n

  for j1 in range(n_tarefas):
    for j2 in range(j1 + 1, n_tarefas):

      if solucao[j1] == solucao[j2]:
        continue

      yield aplicar_swap(solucao, j1, j2)

def aplicar_permutacao_bloco(solucao: np.ndarray,
               frac: float = 0.20) -> np.ndarray:
  n = len(solucao)

  if n == 0:
    return solucao.copy()
  
  n_bloco = max(2, int(n * frac))
  n_bloco = min(n_bloco, n)

  vizinho = solucao.copy()
  indices = np.random.permutation(n)[:n_bloco]
  agentes_bloco = vizinho[indices].copy()

  np.random.shuffle(agentes_bloco)
  vizinho[indices] = agentes_bloco

  return vizinho

def aplicar_shift(solucao: np.ndarray, tarefa: int, novo_agente: int) -> np.ndarray:
  vizinho = solucao.copy()
  vizinho[tarefa] = novo_agente
  return vizinho

def aplicar_swap(solucao: np.ndarray, t1: int, t2: int) -> np.ndarray:
  vizinho = solucao.copy()
  vizinho[t1], vizinho[t2] = vizinho[t2], vizinho[t1]
  return vizinho

def gerar_vizinhos_smart_shift(solucao: np.ndarray, probdata: ProbData) -> Iterator[np.ndarray]:
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

        if nova_carga_destino < carga_origem:
          yield aplicar_shift(solucao, tarefa, novo_agente)

def gerar_vizinhos_swap(solucao: np.ndarray, **kwargs) -> Iterator[np.ndarray]:
  n_tarefas = len(solucao)
  indices = list(range(n_tarefas))
  random.shuffle(indices)

  for i in range(len(indices)):
    j1 = indices[i]

    for k in range(i + 1, len(indices)):
      j2 = indices[k]

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
    t = indices_tarefas[0]
    agente_atual = y[t]
    possiveis = [a for a in range(m_agentes) if a != agente_atual]

    if possiveis:
      novo_agente = np.random.choice(possiveis)
      y = aplicar_shift(y, t, novo_agente)

  elif k == 2:
    t1, t2 = indices_tarefas[0], indices_tarefas[1]

    if y[t1] != y[t2]:
      y = aplicar_swap(y, t1, t2)

    else:
      possiveis = [a for a in range(m_agentes) if a != y[t1]]

      if possiveis:
        y = aplicar_shift(y, t1, np.random.choice(possiveis))

  elif k >= 3:
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

  else:
    frac_bloco = 0.80
    y = aplicar_permutacao_bloco(y, frac=frac_bloco)

  return y

def vnd_hibrido(solucao_inicial: np.ndarray, 
        probdata: object, 
        func_obj: Callable,
        f_id: int) -> np.ndarray:

  x = solucao_inicial.copy()
  k_max = 2 
  k = 1

  while k <= k_max:
    melhorou = False
    novo_x = None

    if k == 1:
      if f_id == 1:
        novo_x, _, melhorou = first_improvement(
          x, gerar_vizinhos_shift_completo, func_obj, probdata=probdata
        )

      else:
        novo_x, _, melhorou = first_improvement(
          x, gerar_vizinhos_smart_shift, func_obj, probdata=probdata
        )

    elif k == 2:
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
  m = probdata.m
  n = probdata.n

  solucao = np.empty(n, dtype=int)
  carga_atual = np.zeros(m)
  capacidade_max = probdata.capacidade
  max_custo_global = np.max(probdata.custo) if np.max(probdata.custo) > 0 else 1.0
  tarefas = list(range(n))

  random.shuffle(tarefas)

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
  f_obj, penalidade = calculate_objective_and_penalty(solucao, f_id, params)

  return f_obj + penalidade

def gvns(f_id: int, params: dict, k_max: int, max_iter_sem_melhora: int):    
  probdata = ProbData(params)

  func_fitness = lambda sol: get_fitness(sol, f_id, params)

  peso_custo = 0.8 if f_id == 1 else 0.2
  melhor_solucao = heuristica_construtiva_com_aleatoriedade(probdata, peso_custo=peso_custo)

  f_obj, penalidade = calculate_objective_and_penalty(melhor_solucao, f_id, params)
  melhor_fitness = f_obj + penalidade

  print(f"Solucao Inicial (f_id={f_id}): Fitness = {melhor_fitness:.4f} (f={f_obj:.2f}, p={penalidade:.2f})")

  iter_sem_melhora = 0
  iter_atual = 0

  historico_convergencia = [(iter_atual, melhor_fitness, f_obj, penalidade)]

  while iter_sem_melhora < max_iter_sem_melhora:
    k = 1

    while k <= k_max:
      #shake
      solucao_shake = shake(melhor_solucao, k, probdata)

      #vnd
      solucao_refinada = vnd_hibrido(solucao_shake, probdata, func_fitness, f_id)

      #avaliacao 
      f_obj_second, p_second = calculate_objective_and_penalty(solucao_refinada, f_id, params)
      fitness_second = f_obj_second + p_second

      # verifica melhoria e neighborhood change
      if fitness_second < melhor_fitness:
        melhor_solucao = solucao_refinada
        melhor_fitness = fitness_second

        f_obj = f_obj_second
        penalidade = p_second

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
