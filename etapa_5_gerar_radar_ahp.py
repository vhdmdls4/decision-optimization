import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
PESOS = {'f1': 0.3891, 'f2': 0.3170, 'f3': 0.1724, 'f4': 0.1215}

def load_instance(prefix="data_5x50"):
    try:
        a = np.loadtxt(f"{prefix}_a.csv", delimiter=",")
        b = np.loadtxt(f"{prefix}_b.csv", delimiter=",")
        return {"a": a, "b": b, "m": len(b), "n": a.shape[1]}
    except: return None

def parse_sol(s):
    clean = str(s).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
    return np.array([int(x) for x in clean.split(",") if x.strip() != ""])

# Funções de Cálculo (Mantidas iguais)
def calc_metrics(sol, params):
    # f3: Robustez (Média)
    cargas = np.zeros(params['m'])
    for t, ag in enumerate(sol): cargas[ag] += params['a'][ag, t]
    
    with np.errstate(divide='ignore'): 
        margens = (params['b'] - cargas) / params['b']
        margens = np.nan_to_num(margens, nan=0.0)
    
    f3 = margens.mean() * 100.0
    
    # f4: Estabilidade (Carga)
    total_carga = np.sum(cargas)
    if total_carga > 0: p = cargas / total_carga
    else: p = np.zeros(params['m'])
    
    p_pos = p[p > 0]
    f4 = -np.sum(p_pos * np.log(p_pos))
    
    return f3, f4

# ==========================================
# 2. PREPARAÇÃO DOS DADOS
# ==========================================
def preparar_dados():
    params = load_instance()
    if not params: return None, None

    try:
        df = pd.concat([
            pd.read_csv("fronteiras_pe.csv").assign(metodo='Pe'),
            pd.read_csv("fronteiras_pw.csv").assign(metodo='Pw')
        ], ignore_index=True)
    except: 
        print("Erro: CSVs não encontrados.")
        return None, None

    # Calcular Métricas
    f3s, f4s = [], []
    for s in df['solucao']:
        val_f3, val_f4 = calc_metrics(parse_sol(s), params)
        f3s.append(val_f3)
        f4s.append(val_f4)
    df['f3'] = f3s
    df['f4'] = f4s
    
    # Remover duplicatas
    df = df.drop_duplicates(subset=['solucao']).copy()
    
    # --- CÁLCULO DO SCORE (NORMALIZAÇÃO GLOBAL) ---
    # Usamos o Global para decidir quem são as Top 5 de verdade
    ranges_global = {}
    for col in ['f1', 'f2', 'f3', 'f4']:
        ranges_global[col] = (df[col].min(), df[col].max())
    
    # Funções auxiliares para normalizar
    def n_min(x, c): return (ranges_global[c][1] - x) / (ranges_global[c][1] - ranges_global[c][0])
    def n_max(x, c): return (x - ranges_global[c][0]) / (ranges_global[c][1] - ranges_global[c][0])
    
    df['n_f1'] = n_min(df['f1'], 'f1')
    df['n_f2'] = n_min(df['f2'], 'f2')
    df['n_f3'] = n_max(df['f3'], 'f3')
    df['n_f4'] = n_min(df['f4'], 'f4')
    
    df['Score'] = (PESOS['f1']*df['n_f1'] + PESOS['f2']*df['n_f2'] + 
                   PESOS['f3']*df['n_f3'] + PESOS['f4']*df['n_f4'])
    
    return df.sort_values('Score', ascending=False).head(5), ranges_global

# ==========================================
# 3. PLOTAGEM COM ZOOM (NORMALIZAÇÃO RELATIVA)
# ==========================================
def plot_radar_zoom(df_top5):
    categories = ['Custo (f1)', 'Equilíbrio (f2)', 'Robustez (f3)', 'Estabilidade (f4)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # --- RE-NORMALIZAÇÃO LOCAL (ZOOM) ---
    # Recalcula a escala apenas considerando o universo das TOP 5
    # Isso faz as diferenças pequenas parecerem maiores no gráfico
    df_plot = df_top5.copy()
    
    ranges_local = {}
    for col in ['f1', 'f2', 'f3', 'f4']:
        ranges_local[col] = (df_top5[col].min(), df_top5[col].max())

    # Função de escala local com margem de segurança
    # Mapeia o pior do grupo para 0.20 e o melhor para 1.00
    def scale_local(val, col, mode):
        mn, mx = ranges_local[col]
        if mx == mn: return 1.0 # Se forem iguais, preenche tudo
        
        if mode == 'min': # Menor é melhor
            norm = (mx - val) / (mx - mn)
        else:             # Maior é melhor
            norm = (val - mn) / (mx - mn)
            
        return 0.2 + (0.8 * norm) # Escala de 0.2 a 1.0

    df_plot['p_f1'] = df_plot['f1'].apply(lambda x: scale_local(x, 'f1', 'min'))
    df_plot['p_f2'] = df_plot['f2'].apply(lambda x: scale_local(x, 'f2', 'min'))
    df_plot['p_f3'] = df_plot['f3'].apply(lambda x: scale_local(x, 'f3', 'max'))
    df_plot['p_f4'] = df_plot['f4'].apply(lambda x: scale_local(x, 'f4', 'min'))

    # Plotagem
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='black', size=11, weight='bold')
    ax.set_rlabel_position(0)
    # Remove as labels numéricas radiais pq a escala é relativa agora
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [], color="grey", size=7) 
    plt.ylim(0, 1.05)
    
    cores = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    estilos = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    
    rank = 1
    for idx, row in df_plot.iterrows():
        values = [row['p_f1'], row['p_f2'], row['p_f3'], row['p_f4']]
        values += values[:1]
        
        # Monta label com valores reais para referência
        label = (f"#{rank} ({row['metodo']})\n"
                 f"Score Global: {row['Score']:.3f}")
        
        ax.plot(angles, values, linewidth=2, linestyle=estilos[rank-1], label=label, color=cores[rank-1])
        ax.fill(angles, values, color=cores[rank-1], alpha=0.05)
        rank += 1
        
    plt.title("Comparativo Relativo - Top 5 Soluções\n(Escala Ajustada para Evidenciar Diferenças)", size=14, y=1.1)
    
    # Legenda fora
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig("grafico_radar_top5_zoom.png", dpi=300)
    print("Gráfico salvo: grafico_radar_top5_zoom.png")

if __name__ == "__main__":
    df_top5, _ = preparar_dados()
    if df_top5 is not None:
        print("Top 5 Soluções:")
        print(df_top5[['f1', 'f2', 'f3', 'f4', 'Score']])
        plot_radar_zoom(df_top5)