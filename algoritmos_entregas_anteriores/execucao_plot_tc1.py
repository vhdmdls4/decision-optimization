# script do plot e da execucao dos testes conforme enunciado
# executar_tc1.py

import algoritmo_tc1_primeira_entrega_corrigido as vns
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def plot_convergence(results_list: list, f_id: int):
    print(f"Plotando curvas de convergencia (ZOOM 10 iter) para f{f_id}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    
    ax1.set_title(f'Curvas de Convergencia (Fitness Total F = f + P) - f{f_id} [ZOOM 10 Iteraçoes]')
    ax1.set_ylabel('Valor do Fitness (Escala Log)')
    ax1.set_yscale('log')
    
    ax2.set_title(f'Curvas de Convergencia (Objetivo Puro f{f_id}) - f{f_id} [ZOOM 10 Iteraçoes]')
    ax2.set_xlabel('Iteracoes (GVNS)')
    ax2.set_ylabel('Valor do Objetivo (Escala Linear)')
    
    for i, run_data in enumerate(results_list):
        convergence_history = run_data['convergencia']
        
        iters = [h[0] for h in convergence_history]
        fitness_F = [h[1] for h in convergence_history] 
        fitness_f = [h[2] for h in convergence_history] 
        
        ax1.plot(iters, fitness_F, 'o-', label=f'Execucao {i+1}') 
        ax2.plot(iters, fitness_f, 'o-', label=f'Execucao {i+1}') 

    ax1.set_xlim(-0.5, 100)
    ax2.set_xlim(-0.5, 100)

    xticks = [0, 5] + list(range(0, 101, 10))
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)

    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    filename = os.path.join('figuras', f'convergencia_f{f_id}_zoom.png') 
    plt.savefig(filename)
    print(f"Grafico de convergencia (zoom) salvo em: {filename}")
    plt.close()

def get_solution_characteristics(solucao: np.ndarray, params: dict):
    m = params['m']
    n = params['n']
    a = params['a']
    c = params['c']
    
    carga_por_agente = np.zeros(m)
    custo_por_agente = np.zeros(m)
    
    for j in range(n):
        agente_i = int(solucao[j])
        carga_por_agente[agente_i] += a[agente_i, j]
        custo_por_agente[agente_i] += c[agente_i, j]
        
    return carga_por_agente, custo_por_agente

def plot_solution_figures(best_run_data: dict, f_id: int, params: dict):
    print(f"Plotando caracteristicas da melhor solucao para f{f_id}...")
    
    best_sol = best_run_data['solucao']
    params = best_run_data['params']
    b = params['b'] 
    m = params['m']
    
    carga_agentes, custo_agentes = get_solution_characteristics(best_sol, params)
    
    agentes_idx = np.arange(m) 
    
    plt.figure(figsize=(12, 7))
    
    bars_carga = plt.bar(agentes_idx, carga_agentes, width=0.6, label='Carga (Recursos Usados)', color='blue')
    
    plt.bar_label(bars_carga, fmt='%.2f') 
    
    plt.plot(agentes_idx, b, 'r-o', label='Capacidade (b_i)', markersize=8)
    
    plt.title(f'Melhor Solucao para f{f_id}: Carga de Recursos vs. Capacidade por Agente')
    plt.xlabel('Agente')
    plt.ylabel('Uso de Recursos')
    plt.xticks(agentes_idx, [f'Agente {i+1}' for i in agentes_idx])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    
    filename_carga = os.path.join('figuras', f'solucao_f{f_id}_carga.png')
    plt.savefig(filename_carga)
    print(f"Grafico de carga salvo em: {filename_carga}")
    plt.close()

    plt.figure(figsize=(12, 7))
    
    bars_custo = plt.bar(agentes_idx, custo_agentes, width=0.6, label='Custo de Alocacao', color='green')
    
    plt.bar_label(bars_custo, fmt='%.2f')
    
    plt.title(f'Melhor Solucao para f{f_id}: Custo por Agente')
    plt.xlabel('Agente')
    plt.ylabel('Custo Total')
    plt.xticks(agentes_idx, [f'Agente {i+1}' for i in agentes_idx])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    
    filename_custo = os.path.join('figuras', f'solucao_f{f_id}_custo.png')
    plt.savefig(filename_custo)
    print(f"Grafico de custo salvo em: {filename_custo}")
    plt.close()

if __name__ == "__main__":
    
    print("--- INICIANDO EXECUCAO DA PARTE III DO TC1 ---")
    
    if not os.path.exists('figuras'):
        os.makedirs('figuras')
    
    params = vns.load_data('data_5x50')
    if params is None:
        print("Falha ao carregar os dados. Abortando.")
        exit()
        
    K_MAX = 3
    MAX_ITER_SEM_MELHORA = 30
    NUM_EXECUCOES = 5
    
    results_f1 = []
    results_f2 = []

    for f_id in [1, 2]:
        print(f"\n--- Processando Objetivo f{f_id} ({NUM_EXECUCOES} execucoes) ---")
        
        current_results = []
        
        for i in range(NUM_EXECUCOES):
            print(f"\nExecucao {i+1}/{NUM_EXECUCOES} para f{f_id}...")
            start_time = time.time()
            
            solucao_final, convergencia = vns.gvns(
                f_id=f_id, 
                params=params, 
                k_max=K_MAX, 
                max_iter_sem_melhora=MAX_ITER_SEM_MELHORA
            )
            
            end_time = time.time()
            
            f_obj = convergencia[-1][2]
            penalidade = convergencia[-1][3]
            fitness_final = convergencia[-1][1]
            
            run_data = {
                'execucao': i+1,
                'f_id': f_id,
                'solucao': solucao_final,
                'fitness': fitness_final,
                'f_obj_final': f_obj,
                'penalidade_final': penalidade,
                'convergencia': convergencia,
                'tempo_exec': end_time - start_time,
                'params': params
            }
            
            current_results.append(run_data)
            print(f"Execucao {i+1} finalizada. Tempo: {run_data['tempo_exec']:.2f}s, Fitness: {run_data['fitness']:.4f}")

        if f_id == 1:
            results_f1 = current_results
        else:
            results_f2 = current_results

    print("\n\n--- (iii.b) RESULTADOS ESTATICOS ---")
    
    fitness_finais_f1 = [r['fitness'] for r in results_f1]
    f_obj_finais_f1 = [r['f_obj_final'] for r in results_f1]
    penal_finais_f1 = [r['penalidade_final'] for r in results_f1]
    
    print("\nEstatisticas para f1 (Fitness = f1 + P):")
    print(f"  Resultados (Fitness): {[round(f, 2) for f in fitness_finais_f1]}")
    print(f"  Resultados (f_obj):   {[round(f, 2) for f in f_obj_finais_f1]}")
    print(f"  Resultados (Penal.):  {[round(f, 2) for f in penal_finais_f1]}")
    print(f"  Minimo (Fitness): {np.min(fitness_finais_f1):.4f}")
    print(f"  Maximo (Fitness): {np.max(fitness_finais_f1):.4f}")
    print(f"  Desvio Padrao (Fitness): {np.std(fitness_finais_f1):.4f}")
    
    fitness_finais_f2 = [r['fitness'] for r in results_f2]
    f_obj_finais_f2 = [r['f_obj_final'] for r in results_f2]
    penal_finais_f2 = [r['penalidade_final'] for r in results_f2]
    
    print("\nEstatisticas para f2 (Fitness = f2 + P):")
    print(f"  Resultados (Fitness): {[round(f, 2) for f in fitness_finais_f2]}")
    print(f"  Resultados (f_obj):   {[round(f, 2) for f in f_obj_finais_f2]}")
    print(f"  Resultados (Penal.):  {[round(f, 2) for f in penal_finais_f2]}")
    print(f"  Minimo (Fitness): {np.min(fitness_finais_f2):.4f}")
    print(f"  Maximo (Fitness): {np.max(fitness_finais_f2):.4f}")
    print(f"  Desvio Padrao (Fitness): {np.std(fitness_finais_f2):.4f}")

    print("\n\n--- (iii.c) GERANDO GRAFICOS DE CONVERGENCIA ---")
    plot_convergence(results_f1, f_id=1)
    plot_convergence(results_f2, f_id=2)

    print("\n\n--- (iii.d) GERANDO GRAFICOS DA MELHOR SOLUCAO ---")
    
    best_run_f1 = min(results_f1, key=lambda x: x['fitness'])
    plot_solution_figures(best_run_f1, f_id=1, params=params)
    
    best_run_f2 = min(results_f2, key=lambda x: x['fitness'])
    plot_solution_figures(best_run_f2, f_id=2, params=params)
    
    print("\n\n--- EXECUCAO DA PARTE III CONCLUIDA ---")
    print("Todos os dados estatisticos foram impressos e os 6 graficos foram salvos.")