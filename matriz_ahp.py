import numpy as np

def calcular_ahp(matriz):
    n = matriz.shape[0] 
    
    # calc dos pesos
    col_sums = matriz.sum(axis=0)
    matriz_norm = matriz / col_sums
    pesos = matriz_norm.mean(axis=1)
    
    # calc da consistencia (lambda_max)
    Aw = np.dot(matriz, pesos)
    lambda_max = np.mean(Aw / pesos)
    
    # calc do indice de consistencia (CI)
    ci = (lambda_max - n) / (n - 1)
    
    # calc da razao de consistencia (CR)
    # Valores de RI para matrizes de ordem 1 a 10 
    ri_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = ri_dict.get(n, 1.49)
    
    cr = ci / ri if ri > 0 else 0
    
    return pesos, cr

# matriz das fn: [f1: Custo, f2: Equilibrio, f3: Robustez, f4: Risco]
matriz_julgamento = np.array([
    [1.0, 5.0, 2.0, 7.0],  # f1 vs 
    [1/5, 1.0, 1/3, 2.0],  # f2 vs 
    [1/2, 3.0, 1.0, 5.0],  # f3 vs 
    [1/7, 1/2, 1/5, 1.0]   # f4 vs 
])

pesos, cr = calcular_ahp(matriz_julgamento)

print("Resultados do AHP")
criterios = ["f1 (Custo)", "f2 (Equilíbrio)", "f3 (Robustez)", "f4 (Risco)"]
for i in range(len(pesos)):
    print(f"{criterios[i]}: {pesos[i]:.4f} ({pesos[i]*100:.1f}%)")

print(f"\nRazão de Consistência (CR): {cr:.4f}")

if cr <= 0.10:
    print(" !!! Show, consistente, vlwss.")
else:
    print(f"> n e consistente (CR={cr:.3f} > 0.1).")