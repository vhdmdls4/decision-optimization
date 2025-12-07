import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

# ==========================================
# 1. CONFIGURAÇÕES & CARGA (Igual ao Radar)
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

def calc_metrics(sol, params):
    # f3: Robustez (Média) -> Maximizar
    cargas = np.zeros(params['m'])
    for t, ag in enumerate(sol): cargas[ag] += params['a'][ag, t]
    
    with np.errstate(divide='ignore'): 
        margens = (params['b'] - cargas) / params['b']
        margens = np.nan_to_num(margens, nan=0.0)
    f3 = margens.mean() * 100.0
    
    # f4: Estabilidade (Carga) -> Minimizar
    total_carga = np.sum(cargas)
    if total_carga > 0: p = cargas / total_carga
    else: p = np.zeros(params['m'])
    p_pos = p[p > 0]
    f4 = -np.sum(p_pos * np.log(p_pos))
    
    return f3, f4

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (Top 8)
# ==========================================
def preparar_dados_top8():
    params = load_instance()
    if not params: 
        print("Erro: Dados data_5x50 não encontrados.")
        return None

    try:
        df = pd.concat([
            pd.read_csv("fronteiras_pe.csv").assign(metodo='Pe'),
            pd.read_csv("fronteiras_pw.csv").assign(metodo='Pw')
        ], ignore_index=True)
    except: 
        print("Erro: CSVs de fronteira não encontrados.")
        return None

    # Calcular Métricas
    f3s, f4s = [], []
    for s in df['solucao']:
        val_f3, val_f4 = calc_metrics(parse_sol(s), params)
        f3s.append(val_f3)
        f4s.append(val_f4)
    df['f3'] = f3s
    df['f4'] = f4s
    
    df = df.drop_duplicates(subset=['solucao']).copy()
    
    # --- FILTRAGEM (K-Means para pegar 20 representativas) ---
    if len(df) > 20:
        # Normaliza temp para cluster
        subset = df[['f1', 'f2']].copy()
        subset = (subset - subset.min()) / (subset.max() - subset.min())
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(subset)
        # Pega a melhor de cada cluster (menor f1 como desempate)
        df = df.sort_values('f1').groupby('cluster').head(1).reset_index(drop=True)

    # --- NORMALIZAÇÃO DE UTILIDADE (0=Pior, 1=Melhor) ---
    ranges = {}
    for col in ['f1', 'f2', 'f3', 'f4']:
        ranges[col] = (df[col].min(), df[col].max())
    
    # Normaliza: 1.0 é sempre BOM (para o gráfico ficar alinhado)
    # f1, f2, f4 são Minimização -> Invertemos
    # f3 é Maximização -> Direto
    def norm_min(x, c): return (ranges[c][1] - x) / (ranges[c][1] - ranges[c][0]) if ranges[c][1]!=ranges[c][0] else 1.0
    def norm_max(x, c): return (x - ranges[c][0]) / (ranges[c][1] - ranges[c][0]) if ranges[c][1]!=ranges[c][0] else 1.0
    
    df['n_f1'] = norm_min(df['f1'], 'f1')
    df['n_f2'] = norm_min(df['f2'], 'f2')
    df['n_f3'] = norm_max(df['f3'], 'f3')
    df['n_f4'] = norm_min(df['f4'], 'f4')
    
    # Score Global
    df['Score'] = (PESOS['f1']*df['n_f1'] + PESOS['f2']*df['n_f2'] + 
                   PESOS['f3']*df['n_f3'] + PESOS['f4']*df['n_f4'])
    
    return df.sort_values('Score', ascending=False).head(8).reset_index(drop=True)

# ==========================================
# 3. PLOTAGEM PARALELA
# ==========================================
def plot_paralelo(df_top):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colunas para plotar (Normalizadas)
    cols_plot = ['n_f1', 'n_f2', 'n_f3', 'n_f4']
    labels_eixo = [
        "Custo ($f_1$)\n(Minimizar)", 
        "Equilíbrio ($f_2$)\n(Minimizar)", 
        "Robustez ($f_3$)\n(Maximizar)", 
        "Estabilidade ($f_4$)\n(Minimizar)"
    ]
    
    x_coords = list(range(len(cols_plot)))
    
    # Plota linhas (Itera reverso para o Rank 1 ficar por cima)
    for idx in reversed(df_top.index):
        row = df_top.loc[idx]
        y_values = [row[c] for c in cols_plot]
        
        # Estilo por método
        if row['metodo'] == 'Pw':
            cor = 'tab:orange'
            alpha = 0.6
            zorder = 2
        else: # Pe
            cor = 'tab:blue'
            alpha = 0.6
            zorder = 2
            
        # Destaque Vencedora
        if idx == 0: # Como ordenamos por Score, índice 0 é a vencedora
            # Contorno preto
            ax.plot(x_coords, y_values, color='black', linewidth=4, alpha=0.8, zorder=10)
            # Linha colorida
            ax.plot(x_coords, y_values, color=cor, linewidth=2.5, zorder=11)
            ax.text(x_coords[0], y_values[0] + 0.02, "Vencedora", ha='center', va='bottom', fontweight='bold')
        else:
            ax.plot(x_coords, y_values, color=cor, linewidth=1.2, alpha=alpha, zorder=zorder)

    # Eixos Verticais
    for x in x_coords:
        ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.3)
        
    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels_eixo, fontsize=11, fontweight='bold')
    
    ax.set_ylabel("Utilidade Normalizada (1.0 = Melhor, 0.0 = Pior)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Coordenadas Paralelas: Top 8 Soluções AHP", fontsize=14, pad=15)
    
    # Legenda
    legend_elements = [
        Line2D([0], [0], color="tab:orange", lw=2, label="Soluções Pw"),
        Line2D([0], [0], color="tab:blue",   lw=2, label="Soluções Pe"),
        Line2D([0], [0], color="black",      lw=3, label="Melhor Solução")
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig("grafico_paralelo_top8.png", dpi=300)
    print("Gráfico salvo: grafico_paralelo_top8.png")

if __name__ == "__main__":
    df_top8 = preparar_dados_top8()
    if df_top8 is not None:
        print("Top 8 para conferência:")
        print(df_top8[['f1', 'f2', 'f3', 'f4', 'Score', 'metodo']].head())
        plot_paralelo(df_top8)