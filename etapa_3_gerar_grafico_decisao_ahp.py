import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================================================
# 0. IMPLEMENTAÇÃO MANUAL DO AHP (Simples e Robusta)
# =========================================================
def ahp_eigen(matrix):
    """Calcula pesos AHP pelo método do autovalor (igual ao pyDecision)"""
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals)
    max_eigval = eigvals[max_idx].real
    eigvec = eigvecs[:, max_idx].real
    weights = eigvec / np.sum(eigvec)
    
    # Consistência
    ci = (max_eigval - n) / (n - 1)
    ri_dict = {3: 0.58, 4: 0.90, 5: 1.12} # Tabela Saaty
    ri = ri_dict.get(n, 1.12)
    cr = ci / ri
    return weights, cr

# =========================================================
# 1. CARREGAMENTO E CÁLCULO DE CRITÉRIOS
# =========================================================
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
    # --- f3: Robustez (Média) ---
    cargas = np.zeros(params['m'])
    for t, ag in enumerate(sol): 
        cargas[ag] += params['a'][ag, t]
        
    with np.errstate(divide='ignore'): 
        margens = (params['b'] - cargas) / params['b']
        margens = np.nan_to_num(margens, nan=0.0)
    
    # Lógica Correta: MÉDIA (Mean)
    # Multiplicamos por 100 aqui apenas para o gráfico ficar em % (0-100)
    f3 = margens.mean() * 100.0 
    
    # --- f4: Estabilidade (Carga) ---
    total_carga = np.sum(cargas)
    if total_carga > 0:
        p = cargas / total_carga
    else:
        p = np.zeros(params['m'])
        
    p_pos = p[p > 0]
    # Lógica Correta: Entropia da CARGA
    f4 = -np.sum(p_pos * np.log(p_pos))
    
    return f3, f4

def gerar_dataframe_unificado():
    """Lê os CSVs originais, calcula f3/f4 e une com tag de método."""
    params = load_instance()
    if not params: 
        print("Erro: Arquivos de dados (data_5x50) não encontrados.")
        return pd.DataFrame()
    
    dfs = []
    # Tenta carregar cada arquivo
    for fname, metodo in [('fronteiras_pe.csv', 'Pe'), ('fronteiras_pw.csv', 'Pw')]:
        try:
            d = pd.read_csv(fname)
            d['metodo'] = metodo # Identificador crucial
            
            # Calcula critérios extras na hora
            f3s, f4s = [], []
            for s in d['solucao']:
                val_f3, val_f4 = calc_metrics(parse_sol(s), params)
                f3s.append(val_f3)
                f4s.append(val_f4)
            d['f3_robustez'] = f3s
            d['f4_estabilidade'] = f4s
            dfs.append(d)
        except Exception as e: 
            print(f"Aviso: {fname} não encontrado ou erro ({e}).")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# =========================================================
# 2. CONFIGURAÇÃO DOS PESOS (MATRIZ AHP)
# =========================================================
# Custo > Equilíbrio > Robustez > Estabilidade
matrix_ahp = np.array([
  [ 1,   1,   3,   3 ], 
  [ 1,   1,   2,   2 ], 
  [1/3, 1/2,  1,   2 ], 
  [1/3, 1/2, 1/2,  1 ] 
])
weights, rc = ahp_eigen(matrix_ahp)
PESOS = {'f1': weights[0], 'f2': weights[1], 'f3': weights[2], 'f4': weights[3]}
print(f"Pesos AHP Calculados: {PESOS} (CR={rc:.4f})")

# =========================================================
# 3. FILTRAGEM (REPRESENTATIVAS) E SCORE
# =========================================================
def filtrar_representativas(df):
    if df.empty or len(df) <= 20: return df
    
    # Clusteriza baseado em f1 e f2 normalizados
    subset = df[['f1', 'f2']].copy()
    subset = (subset - subset.min()) / (subset.max() - subset.min())
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(subset)
    
    # Seleciona 1 por cluster (menor custo como desempate)
    return df.sort_values('f1').groupby('cluster').head(1).reset_index(drop=True)

def calcular_score(df):
    d = df.copy()
    # Ranges globais para normalização
    ranges = {}
    for c in ['f1', 'f2', 'f3_robustez', 'f4_estabilidade']:
        ranges[c] = (d[c].min(), d[c].max())
    
    def norm(x, c, mode): # mode='min' ou 'max'
        denom = ranges[c][1] - ranges[c][0] if ranges[c][1] != ranges[c][0] else 1.0
        return (ranges[c][1] - x)/denom if mode=='min' else (x - ranges[c][0])/denom

    # Score Soma Ponderada
    d['Score'] = (PESOS['f1']*norm(d['f1'], 'f1', 'min') + 
                  PESOS['f2']*norm(d['f2'], 'f2', 'min') + 
                  PESOS['f3']*norm(d['f3_robustez'], 'f3_robustez', 'max') + 
                  PESOS['f4']*norm(d['f4_estabilidade'], 'f4_estabilidade', 'min'))
    return d.sort_values('Score', ascending=False)

# =========================================================
# 4. PLOTAGEM DO GRÁFICO FINAL
# =========================================================
if __name__ == "__main__":
    df_full = gerar_dataframe_unificado()
    
    if not df_full.empty:
        df_rep = filtrar_representativas(df_full)
        df_ranked = calcular_score(df_rep)
        
        # Top 1
        best_sol = df_ranked.iloc[0]
        
        plt.figure(figsize=(10, 7))
        
        # --- DEFINIÇÃO DE ESTILO ---
        style = {
            'Pe': {'c': 'tab:blue',   'm': 'o', 'lbl': 'Pe (Epsilon)'},
            'Pw': {'c': 'tab:orange', 'm': '^', 'lbl': 'Pw (Soma Pond.)'}
        }
        
        # 1. Plotar FUNDO (Todas as soluções, transparente)
        for m in ['Pe', 'Pw']:
            subset = df_full[df_full['metodo'] == m]
            if not subset.empty:
                plt.scatter(subset['f1'], subset['f2'], 
                           c=style[m]['c'], marker=style[m]['m'], 
                           alpha=0.15, s=30, label=f'Todas {style[m]["lbl"]}')

        # 2. Plotar REPRESENTATIVAS (Mais forte, borda preta)
        for m in ['Pe', 'Pw']:
            subset = df_ranked[df_ranked['metodo'] == m]
            if not subset.empty:
                plt.scatter(subset['f1'], subset['f2'], 
                           c=style[m]['c'], marker=style[m]['m'], 
                           s=80, edgecolors='k', alpha=1.0, 
                           label=f'Analisadas {style[m]["lbl"]}')

        # 3. Plotar VENCEDORA (Estrela Dourada)
        m_win = best_sol['metodo']
        plt.scatter(best_sol['f1'], best_sol['f2'], 
                   c='gold', marker='*', s=450, edgecolors='k', linewidth=1.5, zorder=10,
                   label=f"Vencedora ({m_win})\nScore: {best_sol['Score']:.3f}")

        plt.xlabel('Custo Total ($f_1$)', fontsize=12)
        plt.ylabel('Desequilíbrio ($f_2$)', fontsize=12)
        plt.title('Seleção Final AHP: Diferenciação Pe vs Pw', fontsize=14)
        
        # Legenda fora do gráfico para não poluir
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('grafico_decisao_ahp_diferenciado.png', dpi=300)
        print("Sucesso! Gráfico salvo como 'grafico_decisao_ahp_diferenciado.png'")
        
        # Salva CSV final para conferência
        df_ranked.to_csv('tabela_final_com_metodo.csv', index=False)
    else:
        print("Não foi possível gerar o gráfico (sem dados).")