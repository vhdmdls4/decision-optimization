import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

PESOS = {'f1': 0.3891, 'f2': 0.3170, 'f3': 0.1724, 'f4': 0.1215}

def load_instance(prefix="data_5x50"):
    try:
        a = np.loadtxt(f"{prefix}_a.csv", delimiter=",")
        b = np.loadtxt(f"{prefix}_b.csv", delimiter=",")
        return {"a": a, "b": b, "m": len(b), "n": a.shape[1]}
    except: return None

def parse_solution_string(s):
    clean = str(s).replace("[", "").replace("]", "").replace('"', '').replace("'", "")
    return np.array([int(x) for x in clean.split(",") if x.strip() != ""])

def calc_f3_robustness(sol, params):
    cargas = np.zeros(params['m'])
    for t, ag in enumerate(sol):
        cargas[ag] += params['a'][ag, t]
        
    with np.errstate(divide='ignore'):
        margens = (params['b'] - cargas) / params['b']
        margens = np.nan_to_num(margens, nan=0.0)
        
    return float(margens.mean() * 100.0)

def calc_f4_stability(sol, params):
    cargas = np.zeros(params['m'])
    for t, ag in enumerate(sol):
        cargas[ag] += params['a'][ag, t]
        
    total_carga = np.sum(cargas)
    if total_carga > 0:
        p = cargas / total_carga
    else:
        p = np.zeros(params['m'])
        
    p_pos = p[p > 0]
    return float(-np.sum(p_pos * np.log(p_pos)))

def main():
    params = load_instance()
    if not params: return
    
    try:
        df = pd.concat([
            pd.read_csv("fronteiras_pe.csv").assign(origem='Pe'),
            pd.read_csv("fronteiras_pw.csv").assign(origem='Pw')
        ], ignore_index=True)
    except: return

    df['f3_raw'] = [calc_f3_robustness(parse_solution_string(s), params) for s in df['solucao']]
    print("\n=== DISTRIBUIÇÃO DE f3 BRUTO ===")
    print(df['f3_raw'].describe())
    print("\nValores únicos:", sorted(df['f3_raw'].unique()))
    df['f4_raw'] = [calc_f4_stability(parse_solution_string(s), params) for s in df['solucao']]
    
    df_u = df.drop_duplicates(subset=['solucao']).copy()
    subset = (df_u[['f1', 'f2']] - df_u[['f1', 'f2']].min()) / (df_u[['f1', 'f2']].max() - df_u[['f1', 'f2']].min())
    kmeans = KMeans(n_clusters=min(20, len(df_u)), random_state=42, n_init=10)
    df_u['cluster'] = kmeans.fit_predict(subset)
    df_rep = df_u.sort_values('f1').groupby('cluster').head(1).reset_index(drop=True)
    
    ranges = {}
    for col in ['f1', 'f2', 'f3_raw', 'f4_raw']:
        ranges[col] = (df_rep[col].min(), df_rep[col].max())
        
    def norm_min(x, col): return (ranges[col][1] - x) / (ranges[col][1] - ranges[col][0])
    def norm_max(x, col): return (x - ranges[col][0]) / (ranges[col][1] - ranges[col][0])

    df_rep['n_f1'] = norm_min(df_rep['f1'], 'f1')
    df_rep['n_f2'] = norm_min(df_rep['f2'], 'f2')
    df_rep['n_f3'] = norm_max(df_rep['f3_raw'], 'f3_raw')
    df_rep['n_f4'] = norm_min(df_rep['f4_raw'], 'f4_raw')
    
    df_rep['Score_Global'] = (PESOS['f1']*df_rep['n_f1'] + PESOS['f2']*df_rep['n_f2'] + 
                              PESOS['f3']*df_rep['n_f3'] + PESOS['f4']*df_rep['n_f4'])
    
    df_rep['Robustez (Norm 0-100)'] = df_rep['n_f3'] * 100
    df_rep['Estabilidade (Norm 0-100)'] = df_rep['n_f4'] * 100
    
    final_cols = ['f1', 'f2', 'Robustez (Norm 0-100)', 'Estabilidade (Norm 0-100)', 'Score_Global', 'origem']
    rename_map = {'f1': 'Custo Total', 'f2': 'Equilibrio'}
    
    tabela = df_rep.sort_values('Score_Global', ascending=False)[final_cols].rename(columns=rename_map)
    tabela.to_csv("tabela_decisao_final_normalizada.csv", index=False)
    
    print(tabela.head(5).to_string(index=False, float_format="%.2f"))
    print("\nTabela salva: tabela_decisao_final_normalizada.csv")

if __name__ == "__main__":
    main()